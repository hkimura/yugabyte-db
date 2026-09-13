// Copyright (c) YugabyteDB, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied.  See the License for the specific language governing permissions and limitations
// under the License.
//
// Tests for the PITR restore patch (RestorePatch in restore_util.{h,cc}), which rewinds a tablet
// by diffing a restoring state against the live state and writing the difference. Every case here
// is about what the patch does when the restoring row is a PACKED row: the patch reads columns out
// of the packed row through the packing that the row bytes name, and several of its decisions are
// wrong when that packing no longer describes the current schema.
//
// The restoring state is a second, independent tablet, which is the shape the production callers
// hand the patch as well: both materialize a physical snapshot into a separate RocksDB and diff
// that against the live DB.

#include <gtest/gtest.h>

#include "yb/common/ql_protocol_util.h"
#include "yb/common/schema.h"

#include "yb/docdb/consensus_frontier.h"
#include "yb/docdb/doc_read_context.h"
#include "yb/docdb/doc_write_batch.h"
#include "yb/docdb/docdb.h"
#include "yb/docdb/ql_rowwise_iterator_interface.h"

#include "yb/dockv/reader_projection.h"
#include "yb/dockv/schema_packing.h"

#include "yb/qlexpr/ql_expr.h"

#include "yb/rocksdb/db.h"

#include "yb/tablet/local_tablet_writer.h"
#include "yb/tablet/restore_util.h"
#include "yb/tablet/tablet-test-util.h"
#include "yb/tablet/tablet.h"
#include "yb/tablet/tablet_metadata.h"

#include "yb/util/result.h"
#include "yb/util/status_log.h"
#include "yb/util/test_macros.h"
#include "yb/util/test_util.h"

DECLARE_bool(ycql_enable_packed_row);
DECLARE_bool(TEST_dcheck_for_missing_schema_packing);

namespace yb::tablet {

namespace {

// The two production subclasses both add their own filtering; this one keeps everything and does
// nothing extra, so what the tests below observe is the behavior of RestorePatch itself.
class PlainRestorePatch : public RestorePatch {
 public:
  using RestorePatch::RestorePatch;

  Status Finish() override { return Status::OK(); }

 private:
  Result<bool> ShouldSkipEntry(const Slice& key, const Slice& value) override { return false; }
};

}  // namespace

class RestoreUtilTest : public YBTabletTest {
 public:
  RestoreUtilTest() : YBTabletTest(BaseSchema(), TableType::YQL_TABLE_TYPE) {}

  void SetUp() override {
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_ycql_enable_packed_row) = true;
    YBTabletTest::SetUp();
  }

 protected:
  // key is the hash column, c1/c2 are values. Column ids come from
  // Schema::InitColumnIdsByDefault, so key = kFirstColumnId, c1 = +1, c2 = +2.
  static Schema BaseSchema() {
    return Schema({ ColumnSchema("key", DataType::INT32, ColumnKind::HASH),
                    ColumnSchema("c1", DataType::INT32, ColumnKind::VALUE, Nullable::kTrue),
                    ColumnSchema("c2", DataType::INT32, ColumnKind::VALUE, Nullable::kTrue) });
  }

  // Second, independent tablet playing the role of the restoring (snapshot) state. It always uses
  // the unaltered base schema, so a packed row it writes names schema version 0 and the live
  // tablet resolves that version to the very same packing.
  Result<TabletPtr> CreateRestoringTablet() {
    TabletTestHarness::Options opts(GetTestPath("fs_root_restoring"));
    opts.enable_metrics = true;
    opts.table_type = TableType::YQL_TABLE_TYPE;
    opts.tablet_id = "restoring_tablet_id";
    Schema schema = BaseSchema();
    schema.InitColumnIdsByDefault();
    restoring_harness_ = std::make_unique<TabletTestHarness>(schema, opts);
    RETURN_NOT_OK(restoring_harness_->Create(/* first_time= */ true));
    RETURN_NOT_OK(restoring_harness_->Open());
    return restoring_harness_->tablet();
  }

  // Writes every non-key column, which is what makes YCQL emit a packed row rather than a liveness
  // column plus individual entries.
  static void InsertPackedRow(
      const TabletPtr& tablet, int32_t key, const std::vector<int32_t>& values) {
    LocalTabletWriter writer(tablet);
    QLWriteRequestPB req;
    QLAddInt32HashValue(&req, key);
    for (size_t i = 0; i < values.size(); ++i) {
      QLAddInt32ColumnValue(&req, kFirstColumnId + narrow_cast<int>(i) + 1, values[i]);
    }
    ASSERT_OK(writer.Write(&req));
  }

  static void UpdateColumn(
      const TabletPtr& tablet, int32_t key, int column_index, int32_t value) {
    LocalTabletWriter writer(tablet);
    QLWriteRequestPB req;
    req.set_type(QLWriteRequestPB::QL_STMT_UPDATE);
    QLAddInt32HashValue(&req, key);
    QLAddInt32ColumnValue(&req, kFirstColumnId + column_index, value);
    ASSERT_OK(writer.Write(&req));
  }

  static Status FlushTablet(const TabletPtr& tablet) {
    return tablet->Flush(FlushMode::kSync, rocksdb::FlushReason::kTestOnly);
  }

  // Like DumpTablet, but reads at ReadHybridTime::Max(). The patch goes straight to RocksDB and
  // never through MVCC, so the tablet's safe time does not advance past it and a safe-time read
  // would not see the restore at all.
  static Result<std::vector<std::string>> DumpRows(const TabletPtr& tablet) {
    const auto& schema = *tablet->schema();
    dockv::ReaderProjection projection(schema);
    auto iter = VERIFY_RESULT(tablet->NewRowIterator(projection, ReadHybridTime::Max()));
    std::vector<std::string> rows;
    qlexpr::QLTableRow row;
    while (VERIFY_RESULT(iter->FetchNext(&row))) {
      rows.push_back(row.ToString(schema));
    }
    return rows;
  }

  // Runs the restore patch of `restoring` onto the fixture's tablet and applies the resulting
  // batch, the way TabletSnapshots::RestorePartialRows does.
  Status RunRestorePatch(const TabletPtr& restoring) {
    LOG(INFO) << "Existing DocDB:\n" << tablet()->TEST_DocDBDumpStr();
    LOG(INFO) << "Restoring DocDB:\n" << restoring->TEST_DocDBDumpStr();
    auto pending_op = ScopedRWOperation::TEST_Create();
    docdb::DocWriteBatch write_batch(
        tablet()->doc_db(), docdb::InitMarkerBehavior::kOptional, pending_op, nullptr);

    FetchState existing_state(tablet()->doc_db(), ReadHybridTime::Max());
    RETURN_NOT_OK(existing_state.SetPrefix(""));
    FetchState restoring_state(restoring->doc_db(), ReadHybridTime::Max());
    RETURN_NOT_OK(restoring_state.SetPrefix(""));

    PlainRestorePatch patch(
        &existing_state, &restoring_state, &write_batch,
        tablet()->metadata()->primary_table_info().get());
    RETURN_NOT_OK(patch.PatchCurrentStateFromRestoringState());
    RETURN_NOT_OK(patch.Finish());
    last_patch_tickers_ = patch.TickersToString();
    last_patch_inserts_ = patch.GetTicker(RestoreTicker::kInserts);
    last_patch_updates_ = patch.GetTicker(RestoreTicker::kUpdates);
    last_patch_deletes_ = patch.GetTicker(RestoreTicker::kDeletes);
    LOG(INFO) << "Restore patch: " << last_patch_tickers_;

    WriteToRocksDB(
        &write_batch, clock()->Now(), OpId(1, 1), tablet().get(), /* restore_kv= */ std::nullopt);
    LOG(INFO) << "Existing DocDB after patch:\n" << tablet()->TEST_DocDBDumpStr();
    return Status::OK();
  }

  // The census RegularRocksDbListener::OldSchemaGC computes before trimming old packings.
  std::optional<SchemaVersion> MinCensusedSchemaVersion() {
    std::unordered_map<Uuid, SchemaVersion> versions;
    auto* db = tablet()->regular_db();
    auto smallest = db->GetInMemoryFrontier(storage::UpdateUserValueType::kSmallest);
    if (smallest) {
      down_cast<docdb::ConsensusFrontier&>(*smallest).MakeExternalSchemaVersionsAtMost(&versions);
    }
    for (const auto& file : db->GetLiveFilesMetaData()) {
      if (!file.smallest.user_frontier) {
        continue;
      }
      down_cast<docdb::ConsensusFrontier&>(*file.smallest.user_frontier)
          .MakeExternalSchemaVersionsAtMost(&versions);
    }
    auto it = versions.find(Uuid::Nil());
    if (it == versions.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  std::unique_ptr<TabletTestHarness> restoring_harness_;
  std::string last_patch_tickers_;
  size_t last_patch_inserts_ = 0;
  size_t last_patch_updates_ = 0;
  size_t last_patch_deletes_ = 0;
};

// -------------------------------------------------------------------------------------------
// A standalone column delta that is byte-identical on both sides is never rewritten by the patch,
// while the packed row underneath it is rewritten at the patch hybrid time whenever its bytes
// differ. The rewritten root record is then newer than the delta, so the delta is shadowed and the
// restore silently rolls that column back past the restore point.
//
// RestorePatch::ProcessCommonEntry only writes when the values differ
// (restore_util.cc:197-208), a byte-identical entry just advances both iterators
// (:290-295), and the readers reject column entries at or below the row write time
// (doc_reader.cc:1847-1852 for the flat reader, :1554 for the non-flat one), so the surviving
// delta becomes invisible; the next compaction then drops it for good.
//
// The enabling byte difference is SYNTHESIZED here: the two sides are simply given packed rows
// that differ in c1. In production it comes from a live-side compaction between snapshot capture
// and restore, because both callers diff a physical snapshot materialized into a temp RocksDB
// against the live DB rather than one DB at two read times. The closest production analogue of
// what this test hands the patch is the "fold of another column's below-cutoff delta" variant:
// packed row P(c1=a, c2=b), a c1 delta at or below the history cutoff, and a c2 delta above it.
// After the snapshot, the live side folds the c1 delta, so its packed row reads (a', b) while the
// snapshot's still reads (a, b) - byte-different - and the c2 delta is byte-identical on both
// sides. The patch input is then this test's input plus a restoring-only c1 delta, which
// ProcessRestoringOnlyEntry re-inserts later in the same batch and which therefore survives; the
// c2 leg, the one this test asserts on, behaves identically. So the mechanism is exercised
// faithfully, but the test does not by itself prove the production trigger.
TEST_F(RestoreUtilTest, CommonEqualDeltaShadowedByRewrittenPackedRow) {
  auto restoring = ASSERT_RESULT(CreateRestoringTablet());
  ASSERT_NO_FATALS(InsertPackedRow(restoring, 1, {10, 20}));
  ASSERT_NO_FATALS(UpdateColumn(restoring, 1, /* column_index= */ 2, /* value= */ 21));
  ASSERT_OK(FlushTablet(restoring));

  ASSERT_NO_FATALS(InsertPackedRow(tablet(), 1, {11, 20}));
  ASSERT_NO_FATALS(UpdateColumn(tablet(), 1, /* column_index= */ 2, /* value= */ 21));
  ASSERT_OK(FlushTablet(tablet()));

  // Preconditions: the packed rows differ, the c2 deltas are identical. Column ids are relative
  // to kFirstColumnId, which is 0 in normal builds and 10 under ASAN/TSAN.
  const auto existing_dump = tablet()->TEST_DocDBDumpStr();
  const auto restoring_dump = restoring->TEST_DocDBDumpStr();
  const auto c2_str = Format("ColumnId($0)", kFirstColumnId + 2);
  ASSERT_STR_CONTAINS(
      existing_dump, Format("{ $0: 11 $1: 20 }", kFirstColumnId + 1, kFirstColumnId + 2));
  ASSERT_STR_CONTAINS(
      restoring_dump, Format("{ $0: 10 $1: 20 }", kFirstColumnId + 1, kFirstColumnId + 2));
  ASSERT_STR_CONTAINS(existing_dump, c2_str);
  ASSERT_STR_CONTAINS(restoring_dump, c2_str);

  // What the restore has to reproduce, read off the restoring state rather than hard-coded: c2 is
  // 21 there, because the delta is the newest entry for that column on that side too.
  auto restoring_rows = ASSERT_RESULT(DumpRows(restoring));
  ASSERT_EQ(restoring_rows.size(), 1);
  LOG(INFO) << "Restoring state row: " << restoring_rows[0];
  ASSERT_STR_CONTAINS(restoring_rows[0], "int32_value: 21");

  ASSERT_OK(RunRestorePatch(restoring));
  // The packed row was rewritten at the patch hybrid time; that is the precondition the whole
  // case rests on. Only a lower bound, so a fix that also rewrites the delta still satisfies it.
  ASSERT_GE(last_patch_updates_, 1);

  auto rows = ASSERT_RESULT(DumpRows(tablet()));
  ASSERT_EQ(rows.size(), 1);
  LOG(INFO) << "Row after restore: " << rows[0];
  // c1 = 10 shows the packed row itself was restored; c2 must be 21, not the 20 that lives inside
  // that restored packed row.
  ASSERT_STR_CONTAINS(rows[0], "int32_value: 10");
  ASSERT_STR_CONTAINS(rows[0], "int32_value: 21");
}

}  // namespace yb::tablet

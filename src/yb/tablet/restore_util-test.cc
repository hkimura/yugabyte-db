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
// A column that exists only in the live state and is absent from the restoring row's packing is
// neither re-inserted nor tombstoned, so its post-restore-time value survives the rewind.
//
// RestorePatch::ProcessExistingOnlyEntry looks the column up in the packing named by the restoring
// packed row's bytes and returns Status::OK() on kSkippedColumnIdx (restore_util.cc:254-258),
// which leaves the tombstone below it (restore_util.cc:271-276) unreachable. kSkippedColumnIdx is
// IdMapping::kNoEntry (schema_packing.h:80), so the branch fires not only for genuinely skipped
// columns but for every column that packing has never heard of -- i.e. every column ADDED after
// its schema version. The live caller is PgCatalogRestorePatch
// (restore_sys_catalog_state.cc:188-266), which for every pg system table other than
// pg_yb_catalog_version delegates straight to these base methods and whose ShouldSkipEntry then
// returns false -- which is exactly what PlainRestorePatch above does.
TEST_F(RestoreUtilTest, ExistingOnlyAddedColumnSurvivesRestore) {
  auto restoring = ASSERT_RESULT(CreateRestoringTablet());
  ASSERT_NO_FATALS(InsertPackedRow(restoring, 1, {10, 20}));
  ASSERT_OK(FlushTablet(restoring));

  // The same packed row at the same schema version on the live side.
  ASSERT_NO_FATALS(InsertPackedRow(tablet(), 1, {10, 20}));
  ASSERT_OK(FlushTablet(tablet()));

  // A column added after the restore point, with a value written into it.
  SchemaBuilder builder(*tablet()->metadata()->schema());
  ASSERT_OK(builder.AddNullableColumn("c3", DataType::INT32));
  ASSERT_NO_FATALS(AlterSchema(builder.Build()));
  ASSERT_NO_FATALS(UpdateColumn(tablet(), 1, /* column_index= */ 3, /* value= */ 99));
  ASSERT_OK(FlushTablet(tablet()));

  // Preconditions. The packed rows must be byte-identical, otherwise ProcessCommonEntry rewrites
  // the row at the patch hybrid time and the added column would end up shadowed rather than kept
  // -- the defect would be masked by a different one. Column ids are relative to kFirstColumnId,
  // which is 0 in normal builds and 10 under ASAN/TSAN.
  const auto existing_dump = tablet()->TEST_DocDBDumpStr();
  const auto restoring_dump = restoring->TEST_DocDBDumpStr();
  const auto packed_row_str =
      Format("{ $0: 10 $1: 20 }", kFirstColumnId + 1, kFirstColumnId + 2);
  const auto added_column_str = Format("ColumnId($0)", kFirstColumnId + 3);
  ASSERT_STR_CONTAINS(existing_dump, packed_row_str);
  ASSERT_STR_CONTAINS(restoring_dump, packed_row_str);
  ASSERT_STR_CONTAINS(existing_dump, added_column_str);
  ASSERT_STR_NOT_CONTAINS(restoring_dump, added_column_str);

  ASSERT_OK(RunRestorePatch(restoring));
  // The byte-identity precondition, restated as what the patch actually observed: it did not
  // rewrite the packed row. This still holds after a fix, which adds a tombstone for the added
  // column (a delete) and leaves the update counter alone.
  ASSERT_EQ(last_patch_updates_, 0);

  auto rows = ASSERT_RESULT(DumpRows(tablet()));
  ASSERT_EQ(rows.size(), 1);
  LOG(INFO) << "Row after restore: " << rows[0];
  // 99 was written after the restore point, so the rewind must remove it. On the defect the patch
  // does not even count the entry -- the kSkippedColumnIdx early return leaves all three tickers
  // at zero, which is why they are logged above rather than asserted on: a fix adds a tombstone
  // and moves them.
  ASSERT_STR_NOT_CONTAINS(rows[0], "99");
}

}  // namespace yb::tablet

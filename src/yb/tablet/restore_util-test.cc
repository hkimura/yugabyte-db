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

  // The census RegularRocksDbListener::OldSchemaGC computes before trimming old packings, for the
  // primary (nil-cotable) table. This mirrors Tablet::FillMinSchemaVersion (tablet.cc:715-729) for
  // the regular DB only; production also folds in the intents DB and FillMinXClusterSchemaVersion,
  // both of which can only lower the result, and this fixture has neither.
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
// The restore patch writes packed rows through a writer that has no schema-version census, so the
// versions those rows depend on are invisible to schema-packing GC and get trimmed out from under
// them.
//
// yb::WriteToRocksDB (restore_util.cc:346-365) builds a NonTransactionalWriter whose frontiers
// carry op id and hybrid time only; the other three apply paths install a
// FrontierSchemaVersionUpdater. RegularRocksDbListener::OldSchemaGC (tablet.cc:650-676) then
// aggregates frontier schema versions (Tablet::FillMinSchemaVersion, tablet.cc:715-729 - a
// frontier with no entries imposes no constraint, ConsensusFrontier
// ::MakeExternalSchemaVersionsAtMost, consensus_frontier.cc:413-421) and trims every packing below
// the minimum, so a restored row at an older version becomes unreadable ("Schema packing not
// found").
TEST_F(RestoreUtilTest, RestoredPackedRowIsNotCensused) {
  // The live tablet carries schema version 1 only: it is altered before anything is written.
  SchemaBuilder builder(*tablet()->metadata()->schema());
  ASSERT_OK(builder.AddNullableColumn("c3", DataType::INT32));
  ASSERT_NO_FATALS(AlterSchema(builder.Build()));
  ASSERT_EQ(tablet()->metadata()->primary_table_schema_version(), 1);

  ASSERT_NO_FATALS(InsertPackedRow(tablet(), 1, {10, 20, 30}));
  ASSERT_OK(FlushTablet(tablet()));
  ASSERT_EQ(MinCensusedSchemaVersion(), 1);

  // The restoring state holds a row the live state does not have, packed at version 0.
  auto restoring = ASSERT_RESULT(CreateRestoringTablet());
  ASSERT_NO_FATALS(InsertPackedRow(restoring, 2, {40, 50}));
  ASSERT_OK(FlushTablet(restoring));
  // Column ids are relative to kFirstColumnId, which is 0 in normal builds and 10 under ASAN/TSAN.
  ASSERT_STR_CONTAINS(
      restoring->TEST_DocDBDumpStr(),
      Format("{ $0: 40 $1: 50 }", kFirstColumnId + 1, kFirstColumnId + 2));

  ASSERT_OK(RunRestorePatch(restoring));
  // The restoring-only row really was inserted, so what follows is not vacuous.
  ASSERT_GE(last_patch_inserts_, 1);
  ASSERT_OK(FlushTablet(tablet()));

  auto rows_before_gc = ASSERT_RESULT(DumpRows(tablet()));
  ASSERT_EQ(rows_before_gc.size(), 1) << "the restored row replaced the live one as expected";

  // The tablet now stores a version 0 packed row, so the census must see version 0. EXPECT rather
  // than ASSERT so the consequence below is still exercised.
  EXPECT_EQ(MinCensusedSchemaVersion(), 0)
      << "restored packed row is invisible to the schema-version census";

  // OldSchemaGC's trim floor is min(census, MinActiveVersion()) (tablet_metadata.cc:2177-2181), so
  // the census gap only bites if MinActiveVersion() is not itself holding v0 down. It is not, and
  // that is not an artifact of this fixture: each TableInfo lineage owns its own
  // SchemaPackingRegistry (tablet_metadata.cc:190-191, 220-222, shared through the TableInfo copy
  // constructors), so the registry tracks only this table's live packing storages, and once the
  // pre-ALTER TableInfo is released the minimum is the current version. A pg catalog cotable on
  // the master's sys catalog tablet is in exactly that state after a migration ALTERs it - its
  // registry is its own, not shared with the hundreds of never-altered cotables beside it.
  // Asserted so this test cannot pass the GC step for a fixture-only reason.
  ASSERT_EQ(
      tablet()->metadata()->primary_table_info()
          ->doc_read_context->schema_packing_storage.registry().MinActiveVersion(),
      1);

  // Consequence: schema packing GC trims every packing below the census minimum, and the restored
  // row's packing goes with it. Turn the debug CHECK for a missing packing into a plain NotFound
  // so the failure is reportable instead of aborting the process (a release build returns the
  // status either way; TEST_dcheck_for_missing_schema_packing defaults true and the CHECK is
  // compiled in whenever NDEBUG is not, schema_packing.cc:38 and :865-878).
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_TEST_dcheck_for_missing_schema_packing) = false;
  std::unordered_map<Uuid, SchemaVersion> versions;
  versions[Uuid::Nil()] = MinCensusedSchemaVersion().value_or(0);
  ASSERT_OK(tablet()->metadata()->OldSchemaGC(versions));

  auto rows_after_gc = DumpRows(tablet());
  // A failure here is the restored row having become unreadable after schema packing GC.
  ASSERT_OK(rows_after_gc);
  ASSERT_EQ(*rows_after_gc, rows_before_gc);
}

}  // namespace yb::tablet

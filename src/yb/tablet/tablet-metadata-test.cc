// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
// The following only applies to changes made to this file as part of YugabyteDB development.
//
// Portions Copyright (c) YugabyteDB, Inc.
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

#include <cstddef>
#include <set>

#include "yb/util/logging.h"

#include "yb/common/opid.h"
#include "yb/common/ql_protocol_util.h"
#include "yb/common/schema.h"
#include "yb/common/wire_protocol-test-util.h"

#include "yb/fs/fs_manager.h"

#include "yb/gutil/ref_counted.h"

#include "yb/docdb/doc_read_context.h"

#include "yb/dockv/schema_packing.h"

#include "yb/qlexpr/index.h"

#include "yb/tablet/local_tablet_writer.h"
#include "yb/tablet/operations/snapshot_operation.h"
#include "yb/tablet/tablet-test-harness.h"
#include "yb/tablet/tablet-test-util.h"
#include "yb/tablet/tablet.h"
#include "yb/tablet/tablet_metadata.h"
#include "yb/tablet/tablet_snapshots.h"

#include "yb/util/env.h"
#include "yb/util/path_util.h"
#include "yb/util/pb_util.h"
#include "yb/util/result.h"
#include "yb/util/status_log.h"

using std::string;

DECLARE_bool(TEST_dcheck_for_missing_schema_packing);

namespace yb {
namespace tablet {

class TestRaftGroupMetadata : public YBTabletTest {
 public:
  TestRaftGroupMetadata()
      : YBTabletTest(GetSimpleTestSchema()) {
  }

  void SetUp() override {
    YBTabletTest::SetUp();
    writer_.reset(new LocalTabletWriter(harness_->tablet()));
  }

  void BuildPartialRow(int key, int intval, const char* strval,
                       QLWriteRequestPB* req);

  // Builds a standalone FsManager backed by the given (path -> tier) data disks, so tests can
  // exercise multi-drive behaviour independently of the single-disk test harness.
  std::unique_ptr<FsManager> MakeMultiDriveFsManager(
      const std::vector<std::pair<std::string, std::string>>& disks) {
    FsManagerOpts opts;
    for (const auto& [path, tier] : disks) {
      CHECK_OK(env_->CreateDir(path));
      opts.data_paths.push_back(path);
      opts.wal_paths.push_back(path);
      opts.tier_by_path[path] = tier;
    }
    opts.server_type = "tserver_test";
    auto fs = std::make_unique<FsManager>(env_.get(), opts);
    CHECK_OK(fs->CreateInitialFileSystemLayout());
    CHECK_OK(fs->CheckAndOpenFileSystemRoots());
    return fs;
  }

  // Creates a fresh tablet on the given FsManager, homed on the first data root of `home_tier`.
  Result<RaftGroupMetadataPtr> CreateTabletOn(
      FsManager* fs, const std::string& home_tier, const RaftGroupId& tablet_id) {
    auto home_roots = fs->GetDataRootDirsForTier(home_tier);
    CHECK(!home_roots.empty()) << "No data roots for home tier " << home_tier;
    // RaftGroupMetadata::Load() resolves the metadata path via FsManager's tablet->data-root map.
    // In production this is managed by TSTabletManager; tests creating metadata directly must
    // register it explicitly.
    fs->SetTabletPathByDataPath(tablet_id, home_roots[0]);
    auto partition = CreateDefaultPartition(schema_);
    auto table_info = TableInfo::TEST_Create(
        "table_md", "test_ns", "table_md", YQL_TABLE_TYPE, schema_, partition.first);
    return RaftGroupMetadata::CreateNew(
        RaftGroupMetadataData{
            .fs_manager = fs,
            .table_info = table_info,
            .raft_group_id = tablet_id,
            .partition = partition.second,
            .tablet_data_state = TABLET_DATA_READY,
            .snapshot_schedules = {},
            .hosted_services = {},
        },
        home_roots[0], home_roots[0]);
  }

 protected:
  std::unique_ptr<LocalTabletWriter> writer_;
};

void TestRaftGroupMetadata::BuildPartialRow(int key, int intval, const char* strval,
                                         QLWriteRequestPB* req) {
  req->Clear();
  QLAddInt32HashValue(req, key);
  QLAddInt32ColumnValue(req, kFirstColumnId + 1, intval);
  QLAddStringColumnValue(req, kFirstColumnId + 2, strval);
}

// The two tests below pin down the CURRENT behavior when a tablet's data is not at the absolute
// path its superblock records -- the state a swapped data mount or a manual mv/cp of tablet
// directories produces. Both outcomes are silent, which is what motivates the data-root pin
// check (fs/fs_root_pin.h): nothing on the open path validates the recorded path against where
// the metadata was found.

// Data moved away from the recorded path: the tablet reopens EMPTY, creating fresh directories
// at the recorded path, and serves zero rows with no error anywhere. Acknowledged data becomes
// unreadable without a single complaint, while the bytes sit intact at the moved location.
//
// The harness's logical clock restarts at kInitial on every reopen, which would hide old-HT rows
// from reads and make this test pass vacuously; each reopen therefore restores the clock, and the
// first reopen is a control proving a plain reopen does serve the row.
TEST_F(TestRaftGroupMetadata, MovedRocksDbDirReopensEmptyWithoutComplaint) {
  QLWriteRequestPB req;
  BuildPartialRow(0, 0, "foo", &req);
  ASSERT_OK(writer_->Write(&req));
  ASSERT_OK(harness_->tablet()->Flush(tablet::FlushMode::kSync, rocksdb::FlushReason::kTestOnly));

  const auto recorded_dir = harness_->tablet()->metadata()->rocksdb_dir();
  const auto pre_shutdown_ht = clock()->Now();
  harness_->tablet()->StartShutdown(DisableFlushOnShutdown::kFalse, AbortOps::kFalse);
  harness_->tablet()->CompleteShutdown();
  // The writer holds a ref to the old tablet; drop it so TabletReOpen fully replaces the tablet.
  writer_.reset();

  // Reopens the tablet the way this harness allows and makes prior data readable again: the
  // harness clock restarts at kInitial and the fresh MvccManager only advances with traffic, so
  // restore the clock and write one sentinel row; safe time then covers everything written
  // before the shutdown.
  const auto reopen_and_write_sentinel = [&]() {
    TabletReOpen();
    clock()->Update(pre_shutdown_ht);
    writer_.reset(new LocalTabletWriter(harness_->tablet()));
    QLWriteRequestPB sentinel;
    BuildPartialRow(100, 100, "sentinel", &sentinel);
    ASSERT_OK(writer_->Write(&sentinel));
  };

  // Control: a plain reopen, data untouched, serves the acknowledged row beside the sentinel.
  ASSERT_NO_FATALS(reopen_and_write_sentinel());
  std::vector<std::string> rows;
  ASSERT_OK(DumpTablet(*harness_->tablet(), &rows));
  ASSERT_EQ(2, rows.size());

  harness_->tablet()->StartShutdown(DisableFlushOnShutdown::kFalse, AbortOps::kFalse);
  harness_->tablet()->CompleteShutdown();
  writer_.reset();

  // The operator relocates the data (or the mount swaps): every byte still exists, just not at
  // the recorded absolute path.
  const auto moved_dir = recorded_dir + ".moved-by-operator";
  ASSERT_OK(Env::Default()->RenameFile(recorded_dir, moved_dir));

  // Reopen succeeds and takes new writes; nothing notices.
  ASSERT_NO_FATALS(reopen_and_write_sentinel());

  // The acknowledged rows are gone, a fresh database (holding only the new sentinel) sits at the
  // recorded path, and the real data is stranded at the moved path.
  ASSERT_OK(DumpTablet(*harness_->tablet(), &rows));
  ASSERT_EQ(1, rows.size());
  ASSERT_STR_CONTAINS(rows[0], "sentinel");
  ASSERT_TRUE(Env::Default()->FileExists(recorded_dir));
  ASSERT_TRUE(Env::Default()->FileExists(moved_dir));
}

// An OLDER copy of the same tablet at the recorded path (what one half of a swapped mount pair
// looks like, and what a stale manual cp leaves): the tablet opens the stale copy and serves it,
// silently un-happening every write acknowledged since the copy was taken.
TEST_F(TestRaftGroupMetadata, StaleCopyAtRecordedPathIsServedWithoutComplaint) {
  QLWriteRequestPB req;
  BuildPartialRow(0, 0, "v1", &req);
  ASSERT_OK(writer_->Write(&req));
  ASSERT_OK(harness_->tablet()->Flush(tablet::FlushMode::kSync, rocksdb::FlushReason::kTestOnly));

  const auto recorded_dir = harness_->tablet()->metadata()->rocksdb_dir();
  harness_->tablet()->StartShutdown(DisableFlushOnShutdown::kFalse, AbortOps::kFalse);
  harness_->tablet()->CompleteShutdown();
  writer_.reset();

  // Preserve the v1 incarnation, as a copied-away tablet directory or the other volume of a
  // swapped pair would.
  const auto stale_copy = recorded_dir + ".stale-copy";
  ASSERT_OK(Env::Default()->RenameFile(recorded_dir, stale_copy));

  // Second incarnation: overwrite row 0 and add row 1, i.e. writes the client saw acknowledged.
  ASSERT_NO_FATALS(TabletReOpen());
  writer_.reset(new LocalTabletWriter(harness_->tablet()));
  BuildPartialRow(0, 1, "v2", &req);
  ASSERT_OK(writer_->Write(&req));
  BuildPartialRow(1, 1, "v2-second-row", &req);
  ASSERT_OK(writer_->Write(&req));
  ASSERT_OK(harness_->tablet()->Flush(tablet::FlushMode::kSync, rocksdb::FlushReason::kTestOnly));
  const auto pre_shutdown_ht = clock()->Now();
  harness_->tablet()->StartShutdown(DisableFlushOnShutdown::kFalse, AbortOps::kFalse);
  harness_->tablet()->CompleteShutdown();
  writer_.reset();

  // The stale incarnation lands back at the recorded path.
  ASSERT_OK(Env::Default()->DeleteRecursively(recorded_dir));
  ASSERT_OK(Env::Default()->RenameFile(stale_copy, recorded_dir));

  // Reopen succeeds and serves the stale data with no error anywhere: the second row never
  // existed and row 0 is back at "v1". (Clock restored and a sentinel written for the same
  // reason as the previous test: the harness clock and MvccManager restart on reopen.)
  ASSERT_NO_FATALS(TabletReOpen());
  clock()->Update(pre_shutdown_ht);
  writer_.reset(new LocalTabletWriter(harness_->tablet()));
  BuildPartialRow(100, 100, "sentinel", &req);
  ASSERT_OK(writer_->Write(&req));

  std::vector<std::string> rows;
  ASSERT_OK(DumpTablet(*harness_->tablet(), &rows));
  ASSERT_EQ(2, rows.size());
  ASSERT_STR_CONTAINS(yb::ToString(rows), "v1");
  ASSERT_STR_NOT_CONTAINS(yb::ToString(rows), "v2-second-row");
}

// Test that loading & storing the superblock results in an equivalent file.
TEST_F(TestRaftGroupMetadata, TestLoadFromSuperBlock) {
  // Write some data to the tablet and flush.
  QLWriteRequestPB req;
  BuildPartialRow(0, 0, "foo", &req);
  ASSERT_OK(writer_->Write(&req));
  ASSERT_OK(harness_->tablet()->Flush(tablet::FlushMode::kSync, rocksdb::FlushReason::kTestOnly));

  // Create one more row. Write and flush.
  BuildPartialRow(1, 1, "bar", &req);
  ASSERT_OK(writer_->Write(&req));
  ASSERT_OK(harness_->tablet()->Flush(tablet::FlushMode::kSync, rocksdb::FlushReason::kTestOnly));

  // Shut down the tablet.
  harness_->tablet()->StartShutdown(DisableFlushOnShutdown::kFalse, AbortOps::kFalse);
  harness_->tablet()->CompleteShutdown();

  RaftGroupMetadata* meta = harness_->tablet()->metadata();

  // Dump the superblock to a PB. Save the PB to the side.
  RaftGroupReplicaSuperBlockPB superblock_pb_1;
  meta->ToSuperBlock(&superblock_pb_1);

  // Load the superblock PB back into the RaftGroupMetadata.
  ASSERT_OK(meta->ReplaceSuperBlock(superblock_pb_1));

  // Dump the tablet metadata to a superblock PB again, and save it.
  RaftGroupReplicaSuperBlockPB superblock_pb_2;
  meta->ToSuperBlock(&superblock_pb_2);

  // Compare the 2 dumped superblock PBs.
  ASSERT_EQ(superblock_pb_1.SerializeAsString(),
            superblock_pb_2.SerializeAsString())
    << superblock_pb_1.DebugString()
    << superblock_pb_2.DebugString();

  LOG(INFO) << "Superblocks match:\n"
            << superblock_pb_1.DebugString();
}

TEST_F(TestRaftGroupMetadata, TestDeleteTabletDataClearsDisk) {
  auto tablet = harness_->tablet();

  // Write some data to the tablet and flush.
  QLWriteRequestPB req;
  BuildPartialRow(0, 0, "foo", &req);
  ASSERT_OK(writer_->Write(&req));
  ASSERT_OK(tablet->Flush(tablet::FlushMode::kSync, rocksdb::FlushReason::kTestOnly));

  // Create one more row. Write and flush.
  BuildPartialRow(1, 1, "bar", &req);
  ASSERT_OK(writer_->Write(&req));
  ASSERT_OK(tablet->Flush(tablet::FlushMode::kSync, rocksdb::FlushReason::kTestOnly));

  const string snapshotId = "0123456789ABCDEF0123456789ABCDEF";
  tserver::TabletSnapshotOpRequestPB request;
  request.set_snapshot_id(snapshotId);
  tablet::SnapshotOperation operation(tablet);
  operation.AllocateRequest()->CopyFrom(request);
  operation.set_hybrid_time(tablet->clock()->Now());
  operation.set_op_id(OpId(-1, 2));
  ASSERT_OK(tablet->snapshots().Create(&operation));

  // Tiered storage: simulate SSTs living on a second (non-home) tier disk and verify delete
  // tears that directory down too, not just the home dir.
  const auto tier_dir = JoinPathSegments(GetTestPath("tier_disk"), "rocksdb/table-x/tablet-y");
  ASSERT_OK(env_->CreateDirs(tier_dir));
  ASSERT_OK(WriteStringToFile(env_.get(), "sst", JoinPathSegments(tier_dir, "000123.sst")));
  tablet->metadata()->TEST_SetTierPaths(
      {{.path_id = 0, .tier = "ssd", .path = tablet->metadata()->rocksdb_dir()},
       {.path_id = 1, .tier = "hdd", .path = tier_dir}});

  ASSERT_TRUE(env_->DirExists(tablet->metadata()->rocksdb_dir()));
  ASSERT_TRUE(env_->DirExists(tablet->metadata()->intents_rocksdb_dir()));
  ASSERT_TRUE(env_->DirExists(tablet->metadata()->snapshots_dir()));
  ASSERT_TRUE(env_->DirExists(tier_dir));

  CHECK_OK(tablet->metadata()->DeleteTabletData(
    TabletDataState::TABLET_DATA_DELETED, operation.op_id()));

  ASSERT_FALSE(env_->DirExists(tablet->metadata()->rocksdb_dir()));
  ASSERT_FALSE(env_->DirExists(tablet->metadata()->intents_rocksdb_dir()));
  ASSERT_FALSE(env_->DirExists(tablet->metadata()->snapshots_dir()));
  ASSERT_FALSE(env_->DirExists(tier_dir));
}

TEST_F(TestRaftGroupMetadata, NamespaceIdPreservedAcrossSchemaChanges) {
  // Verify that namespace_id is preserved across schema updates and packed schema insertions.
  auto tablet = harness_->tablet();
  auto* meta = tablet->metadata();

  // Simulate namespace backfill.
  const NamespaceId kNamespaceId = "0123456789abcdef0123456789abcdef";
  ASSERT_OK(meta->set_namespace_id(kNamespaceId));
  ASSERT_EQ(meta->primary_table_info()->namespace_id, kNamespaceId);

  auto initial_table_info = meta->primary_table_info();
  const Schema initial_schema = initial_table_info->schema();
  const auto initial_version = initial_table_info->schema_version;
  const qlexpr::IndexMap& index_map = *initial_table_info->index_map;
  const TableId& table_id = initial_table_info->table_id;

  // Perform schema changes and verify that namespace_id is preserved.

  // 1. SetSchema.
  meta->SetSchema(
      initial_schema, index_map, /* deleted_cols */ {}, initial_version + 1, OpId() /* op_id */,
      table_id);
  ASSERT_EQ(meta->primary_table_info()->namespace_id, kNamespaceId);

  // 2. InsertPackedSchemaForXClusterTarget.
  meta->InsertPackedSchemaForXClusterTarget(
      initial_schema, index_map, initial_version + 3, OpId() /* op_id */, table_id);
  ASSERT_EQ(meta->primary_table_info()->namespace_id, kNamespaceId);
}

// Verify that index_map() returns NotFound for a nonexistent colocated table rather than crashing.
// Before the fix, index_map() used CHECK_RESULT(GetTableInfo(table_id)) which would FATAL on a
// missing table. This test would crash (not just fail) without the fix.
TEST_F(TestRaftGroupMetadata, IndexMapReturnsNotFoundForMissingTable) {
  auto* meta = harness_->tablet()->metadata();

  // Primary table (empty table_id) should always succeed.
  auto primary_result = meta->index_map();
  ASSERT_OK(primary_result);
  ASSERT_NE(*primary_result, nullptr);

  // A nonexistent table_id should return NotFound, not crash.
  const TableId kNonexistentTableId = "deadbeefdeadbeefdeadbeefdeadbeef";
  auto missing_result = meta->index_map(kNonexistentTableId);
  ASSERT_NOK(missing_result);
  ASSERT_TRUE(missing_result.status().IsNotFound()) << missing_result.status();
}

// Tiered storage: verify that tier_paths survives a superblock round-trip, and that an
// old / non-tiered superblock (no tier_paths) synthesizes a single home entry on load.
TEST_F(TestRaftGroupMetadata, TierPathsRoundTripAndSynthesis) {
  auto* meta = harness_->tablet()->metadata();
  auto* fs = meta->fs_manager();
  const auto raft_group_id = meta->raft_group_id();

  // The test harness uses a single data dir, so tier_paths starts as a synthesized home entry.
  const auto& initial = meta->tier_paths();
  ASSERT_EQ(initial.size(), 1u);
  ASSERT_EQ(initial[0].path_id, 0u);
  ASSERT_EQ(initial[0].path, meta->rocksdb_dir());

  // Persist a two-entry tier_paths (home ssd + slow hdd) via ReplaceSuperBlock.
  // Note: ReplaceSuperBlock reloads with local_superblock=false (remote superblock semantics),
  // so in-memory tier_paths are not refreshed from the pb - verify on disk and via Load().
  RaftGroupReplicaSuperBlockPB sb;
  meta->ToSuperBlock(&sb);
  auto* kv = sb.mutable_kv_store();
  kv->clear_tier_paths();
  {
    auto* home = kv->add_tier_paths();
    home->set_path_id(0);
    home->set_tier("ssd");
    home->set_path(meta->rocksdb_dir());
  }
  {
    auto* slow = kv->add_tier_paths();
    slow->set_path_id(1);
    slow->set_tier("hdd");
    slow->set_path("/some/hdd/path");
  }
  ASSERT_OK(meta->ReplaceSuperBlock(sb));

  RaftGroupReplicaSuperBlockPB on_disk;
  ASSERT_OK(meta->ReadSuperBlockFromDisk(&on_disk));
  const auto& kv_disk = on_disk.kv_store();
  ASSERT_EQ(kv_disk.tier_paths_size(), 2);
  ASSERT_EQ(kv_disk.tier_paths(0).path_id(), 0u);
  ASSERT_EQ(kv_disk.tier_paths(0).tier(), "ssd");
  ASSERT_EQ(kv_disk.tier_paths(0).path(), meta->rocksdb_dir());
  ASSERT_EQ(kv_disk.tier_paths(1).path_id(), 1u);
  ASSERT_EQ(kv_disk.tier_paths(1).tier(), "hdd");
  ASSERT_EQ(kv_disk.tier_paths(1).path(), "/some/hdd/path");

  // Reload metadata from disk (same as tserver restart): local_superblock=true applies tier_paths.
  auto reloaded = ASSERT_RESULT(RaftGroupMetadata::Load(fs, raft_group_id));
  const auto& loaded = reloaded->tier_paths();
  ASSERT_EQ(loaded.size(), 2u);
  ASSERT_EQ(loaded[0].path_id, 0u);
  ASSERT_EQ(loaded[0].tier, "ssd");
  ASSERT_EQ(loaded[0].path, meta->rocksdb_dir());
  ASSERT_EQ(loaded[1].path_id, 1u);
  ASSERT_EQ(loaded[1].tier, "hdd");
  ASSERT_EQ(loaded[1].path, "/some/hdd/path");

  // ToSuperBlock on reloaded metadata must serialize the same two entries.
  RaftGroupReplicaSuperBlockPB sb2;
  reloaded->ToSuperBlock(&sb2);
  ASSERT_EQ(sb2.kv_store().tier_paths_size(), 2);

  // Upgrade of an old (pre-tiered-storage) superblock: with no tier_paths on disk, Load() must
  // rebuild them from the node's disks (BuildTierPaths) AND persist them (the single-data-dir
  // harness yields exactly the home entry).
  sb2.mutable_kv_store()->clear_tier_paths();
  ASSERT_OK(reloaded->ReplaceSuperBlock(sb2));
  auto reloaded2 = ASSERT_RESULT(RaftGroupMetadata::Load(fs, raft_group_id));
  const auto& migrated = reloaded2->tier_paths();
  ASSERT_EQ(migrated.size(), 1u);
  ASSERT_EQ(migrated[0].path_id, 0u);
  ASSERT_EQ(migrated[0].path, reloaded2->rocksdb_dir());

  // The upgrade must be persisted, not just synthesized in-memory: the on-disk superblock now
  // carries tier_paths, so a subsequent load does not need to migrate again.
  RaftGroupReplicaSuperBlockPB after_migration;
  ASSERT_OK(reloaded2->ReadSuperBlockFromDisk(&after_migration));
  ASSERT_EQ(after_migration.kv_store().tier_paths_size(), 1);
  ASSERT_EQ(after_migration.kv_store().tier_paths(0).path(), reloaded2->rocksdb_dir());
}

// Tiered storage: a tablet created on a multi-drive tserver must record one tier_paths entry per
// disk (every drive across every tier), with path_id 0 == home rocksdb_dir, and that list must be
// persisted to the on-disk superblock (not just held in memory).
TEST_F(TestRaftGroupMetadata, NewTabletOnMultiDriveHasAllTierPaths) {
  // 3-disk tserver: one SSD (home tier) + two HDDs.
  const auto ssd = GetTestPath("md_ssd");
  const auto hdd_a = GetTestPath("md_hdd_a");
  const auto hdd_b = GetTestPath("md_hdd_b");
  auto fs = MakeMultiDriveFsManager({{ssd, "ssd"}, {hdd_a, "hdd"}, {hdd_b, "hdd"}});

  const RaftGroupId kTabletId = "0123456789abcdef0123456789abcde1";
  auto meta = ASSERT_RESULT(CreateTabletOn(fs.get(), "ssd", kTabletId));

  // All three disks are represented; path_id 0 is the home dir on the ssd tier.
  const auto& tier_paths = meta->tier_paths();
  ASSERT_EQ(tier_paths.size(), 3u);
  ASSERT_EQ(tier_paths[0].path_id, 0u);
  ASSERT_EQ(tier_paths[0].tier, "ssd");
  ASSERT_EQ(tier_paths[0].path, meta->rocksdb_dir());

  // path_ids are dense 0..N-1, both tiers appear, and every slot carries this tablet's suffix.
  std::set<uint32_t> path_ids;
  std::set<std::string> tiers_seen;
  for (const auto& tp : tier_paths) {
    path_ids.insert(tp.path_id);
    tiers_seen.insert(tp.tier);
    ASSERT_NE(tp.path.find(kTabletId), std::string::npos) << tp.path;
  }
  ASSERT_EQ(path_ids, (std::set<uint32_t>{0, 1, 2}));
  ASSERT_EQ(tiers_seen, (std::set<std::string>{"ssd", "hdd"}));

  // The tier_paths must be durable: a fresh Load() (like a restart) sees all three, and the raw
  // on-disk superblock carries them.
  auto reloaded = ASSERT_RESULT(RaftGroupMetadata::Load(fs.get(), kTabletId));
  ASSERT_EQ(reloaded->tier_paths().size(), 3u);
  RaftGroupReplicaSuperBlockPB on_disk;
  ASSERT_OK(reloaded->ReadSuperBlockFromDisk(&on_disk));
  ASSERT_EQ(on_disk.kv_store().tier_paths_size(), 3);
}

// Tiered storage: an old (pre-tiered-storage) superblock with no tier_paths, loaded on a
// multi-drive tserver, must be upgraded to record every disk and the upgrade must be persisted, so
// a subsequent load does not need to migrate again.
TEST_F(TestRaftGroupMetadata, MigrateOldSuperblockOnMultiDrivePopulatesAllDisks) {
  const auto ssd = GetTestPath("mig_ssd");
  const auto hdd = GetTestPath("mig_hdd");
  auto fs = MakeMultiDriveFsManager({{ssd, "ssd"}, {hdd, "hdd"}});

  const RaftGroupId kTabletId = "0123456789abcdef0123456789abcde2";
  auto meta = ASSERT_RESULT(CreateTabletOn(fs.get(), "ssd", kTabletId));
  ASSERT_EQ(meta->tier_paths().size(), 2u);

  // Simulate a pre-tiered-storage superblock by clearing tier_paths and writing that to disk.
  RaftGroupReplicaSuperBlockPB sb;
  meta->ToSuperBlock(&sb);
  sb.mutable_kv_store()->clear_tier_paths();
  ASSERT_OK(meta->ReplaceSuperBlock(sb));
  {
    RaftGroupReplicaSuperBlockPB cleared;
    ASSERT_OK(meta->ReadSuperBlockFromDisk(&cleared));
    ASSERT_EQ(cleared.kv_store().tier_paths_size(), 0);
  }

  // Loading the tablet (as on tserver restart) must repopulate tier_paths for every disk...
  auto reloaded = ASSERT_RESULT(RaftGroupMetadata::Load(fs.get(), kTabletId));
  ASSERT_EQ(reloaded->tier_paths().size(), 2u);
  ASSERT_EQ(reloaded->tier_paths()[0].path_id, 0u);
  ASSERT_EQ(reloaded->tier_paths()[0].path, reloaded->rocksdb_dir());

  // ...and persist the upgrade so the next load finds tier_paths already present.
  RaftGroupReplicaSuperBlockPB after_migration;
  ASSERT_OK(reloaded->ReadSuperBlockFromDisk(&after_migration));
  ASSERT_EQ(after_migration.kv_store().tier_paths_size(), 2);
}

// Test for a colocated-restore: RaftGroupMetadata::LoadFromPath() (used by
// TabletSnapshots::GetCotableIdsMap() to read a snapshot's superblock file during
// RESTORE_ON_TABLET) constructs its RaftGroupMetadata with an empty raft_group_id_, since the
// object isn't a locally-owned, FsManager-registered tablet.
TEST_F(TestRaftGroupMetadata, LoadFromPathDoesNotMigrateOrFlushOldSuperblock) {
  const auto ssd = GetTestPath("lfp_ssd");
  const auto hdd = GetTestPath("lfp_hdd");
  auto fs = MakeMultiDriveFsManager({{ssd, "ssd"}, {hdd, "hdd"}});

  const RaftGroupId kTabletId = "0123456789abcdef0123456789abcde3";
  const ColocationId kColocationId = 123456789;
  Schema colocated_schema(schema_);
  colocated_schema.set_colocation_id(kColocationId);

  auto home_roots = fs->GetDataRootDirsForTier("ssd");
  ASSERT_FALSE(home_roots.empty());
  fs->SetTabletPathByDataPath(kTabletId, home_roots[0]);
  auto partition = CreateDefaultPartition(colocated_schema);
  auto table_info = TableInfo::TEST_Create(
      "table_md", "test_ns", "table_md", YQL_TABLE_TYPE, colocated_schema, partition.first);
  auto meta = ASSERT_RESULT(RaftGroupMetadata::CreateNew(
      RaftGroupMetadataData{
          .fs_manager = fs.get(),
          .table_info = table_info,
          .raft_group_id = kTabletId,
          .partition = partition.second,
          .tablet_data_state = TABLET_DATA_READY,
          .colocated = true,
          .snapshot_schedules = {},
          .hosted_services = {},
      },
      home_roots[0], home_roots[0]));
  ASSERT_EQ(meta->tier_paths().size(), 2u);
  ASSERT_EQ(meta->GetColocatedTableInfos().size(), 1u);
  const auto original_rocksdb_dir = meta->rocksdb_dir();

  // Build a pre-tiered-storage superblock (no tier_paths) and write it directly to a standalone
  // file, simulating a snapshot's "tablet.metadata" file.
  RaftGroupReplicaSuperBlockPB sb;
  meta->ToSuperBlock(&sb);
  sb.mutable_kv_store()->clear_tier_paths();
  const auto snapshot_metadata_file = GetTestPath("simulated_snapshot_tablet.metadata");
  ASSERT_OK(pb_util::WritePBContainerToPath(
      env_.get(), snapshot_metadata_file, sb, pb_util::OVERWRITE, pb_util::SYNC));

  auto loaded = ASSERT_RESULT(RaftGroupMetadata::LoadFromPath(fs.get(), snapshot_metadata_file));

  // Empty id confirms this object is (correctly) not treated as a locally-owned tablet.
  ASSERT_TRUE(loaded->raft_group_id().empty());

  // Migration must not have run: tier_paths is the single synthesized home entry from
  // KvStoreInfo::LoadFromPB's backward-compatibility fallback, not BuildTierPaths' 2-disk result.
  // This fallback is purely an in-memory default applied while parsing the PB into the KvStoreInfo
  // struct below -- it never touches the serialized bytes, so it has no bearing on whether the
  // file on disk was rewritten.
  ASSERT_EQ(loaded->tier_paths().size(), 1u);
  ASSERT_EQ(loaded->tier_paths()[0].path, original_rocksdb_dir);

  // The migration's Flush() must not have run either: the standalone snapshot file's raw
  // serialized bytes must still show 0 tier_paths, exactly as written above. If Flush() had run,
  // this would instead read back 2 (BuildTierPaths' result for this 2-disk FsManager).
  RaftGroupReplicaSuperBlockPB on_disk;
  ASSERT_OK(RaftGroupMetadata::ReadSuperBlockFromDisk(
      env_.get(), snapshot_metadata_file, &on_disk));
  ASSERT_EQ(on_disk.kv_store().tier_paths_size(), 0);

  // The actual purpose of LoadFromPath() in the restore path -- reading colocated table info to
  // build the cotable-ids map -- must still work, unaffected by any of the above.
  auto loaded_colocated = loaded->GetColocatedTableInfos();

  ASSERT_EQ(loaded_colocated.size(), 1u);
  ASSERT_EQ(loaded_colocated[0]->schema().colocation_id(), kColocationId);
}

// ---------------------------------------------------------------------------
// A YCQL out-of-cluster restore can rebind the tablet's CURRENT schema version to the snapshot's
// packing, and the binding does not even survive a superblock round trip.
//
// Export side: RepackSnapshotsForBackup repairs the skew between the snapshotted
// SysTablesEntryPB.version (frozen when the master collected entries) and the schema version the
// tablets later reached because an ALTER raced CREATE_ON_TABLET. That repair
// (src/yb/master/catalog_manager_ext.cc:744-765) sits INSIDE
// `if (l->table_type() == PGSQL_TABLE_TYPE && !IsColocationParentTableId(entry.id()))` (:695), so
// it never runs for a YCQL (YQL_TABLE_TYPE) table.
//
// Import side: ImportTableEntry recreates the table from the snapshot entry's (pre-ALTER) schema
// and then, for a YCQL main table, takes `else if (meta.version() >= table->...pb.version())
// schema_version = meta.version() + 1` (:2569-2578) -- the YCQL arm above it (:2559-2560) only
// carves out CQL *index* tables. With the unrepaired meta.version() = N, the restore side's
// current version becomes N+1, which is exactly the version the snapshot's own tablet superblock
// carries.
//
// Restore side: a non-schedule restore merges with OverwriteSchemaPacking::kTrue
// (tablet_snapshots.cc, `is_pitr_restore = !request.schedule_id().empty()`), and
// SchemaPackingStorage::GetMergedSchemaPackings (dockv/schema_packing.cc:939-960) then does
// `new_packings.erase(schema_version)` followed by `AddSchema(table_type_, schema_version,
// temp_schema /* = pb.schema() */)`. With the skew, `schema_version` is the restore side's
// CURRENT version, so the packing the master just committed for the restored table is erased and
// replaced by the source's post-ALTER packing.
//
// The detector is TableInfo::MergeSchemaPackings' second guard
// (src/yb/tablet/tablet_metadata.cc:396-402): LOG_IF(DFATAL, !SchemaContainsPacking(...)) followed
// by `return Status::OK()`. DFATAL is LOG(ERROR) in release (util/logging.h:167-171), so the
// restore reports success. The tests below therefore call DocReadContext::MergeWithRestored -- the
// single line TableInfo::MergeSchemaPackings delegates to at tablet_metadata.cc:387 -- so that a
// fastdebug binary observes the release-build outcome instead of aborting on the detector.
//
// What these tests demonstrate is the destructive half (the merge). The master-side scoping half
// is not exercised here; see the commit message.
namespace {

// The TableInfoPB a snapshot superblock carries for a source tablet that started at
// `old_schema` (version 0) and advanced to `new_schema` at `new_version` -- i.e. what
// TabletSnapshots::Create writes via metadata()->SaveTo(TabletMetadataFile(snapshot_dir)).
// TableInfo::ToPB -> DocReadContext::ToPB skips the CURRENT version's packing, so the result is
// schema_version = new_version, schema = new_schema, old_schema_packings = {0: old_schema}.
TableInfoPB MakeSnapshotSuperblockTableInfo(
    const Schema& old_schema, const Schema& new_schema, SchemaVersion new_version) {
  auto partition = CreateDefaultPartition(old_schema);
  auto source_v0 = TableInfo::TEST_Create(
      "snapshot_src", "test_ns", "snapshot_src", YQL_TABLE_TYPE, old_schema, partition.first);
  auto source_vn = std::make_shared<TableInfo>(
      *source_v0, new_schema, qlexpr::IndexMap(), std::vector<DeletedColumn>(), new_version);
  TableInfoPB pb;
  source_vn->ToPB(&pb);
  return pb;
}

}  // namespace

class YcqlRestoreSchemaSkewTest : public TestRaftGroupMetadata {
 public:
  void SetUp() override {
    // A packing-registry miss CHECK-fails in a debug build while this is on. Off, it surfaces as
    // the Status a release build sees.
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_TEST_dcheck_for_missing_schema_packing) = false;
    TestRaftGroupMetadata::SetUp();
  }

 protected:
  static constexpr SchemaVersion kSourceAlteredVersion = 1;

  // Sets up the restore-side tablet at `restore_side_version` with the pre-ALTER schema (what
  // RecreateTable + SendAlterTableRequest leave behind) and returns the snapshot superblock's
  // TableInfoPB for a source tablet that dropped a column at version kSourceAlteredVersion.
  //
  // restore_side_version == kSourceAlteredVersion is the unrepaired YCQL case: the master's
  // frozen SysTablesEntryPB.version was 0, so ImportTableEntry bumped to 0 + 1 = 1, colliding
  // with the version the source tablet reached.
  // restore_side_version == kSourceAlteredVersion + 1 is what the YSQL repair produces: the
  // exported version was raised to 1, so the bump lands at 2 and nothing collides.
  TableInfoPB PrepareRestoreSide(SchemaVersion restore_side_version) {
    auto* meta = harness_->tablet()->metadata();
    const Schema pre_alter = *meta->schema();
    SchemaBuilder builder(pre_alter);
    CHECK_OK(builder.RemoveColumn("int_val"));
    post_alter_schema_ = builder.Build();
    CHECK(!post_alter_schema_.Equals(pre_alter));

    auto snapshot_pb =
        MakeSnapshotSuperblockTableInfo(pre_alter, post_alter_schema_, kSourceAlteredVersion);
    CHECK_EQ(snapshot_pb.schema_version(), kSourceAlteredVersion);

    meta->SetSchema(
        pre_alter, qlexpr::IndexMap(), /* deleted_cols = */ {}, restore_side_version, OpId());
    return snapshot_pb;
  }

  static std::string PackingAt(const TableInfoPtr& table_info, SchemaVersion version) {
    auto packing = table_info->doc_read_context->schema_packing_storage.GetPacking(version);
    return packing.ok() ? packing->ToString() : packing.status().ToString();
  }

  Schema post_alter_schema_;
};

// The destructive half: the merge must not leave the tablet's current schema version bound to a
// packing that does not describe the tablet's current schema.
TEST_F(YcqlRestoreSchemaSkewTest, MergeDoesNotRebindCurrentSchemaVersion) {
  const auto snapshot_pb = PrepareRestoreSide(kSourceAlteredVersion);
  auto* meta = harness_->tablet()->metadata();
  auto table_info = meta->primary_table_info();
  auto& storage = table_info->doc_read_context->schema_packing_storage;

  // Preconditions.
  // (a) The skew: the snapshot superblock names the same schema version the restore side is
  //     currently committed at. This is what makes GetMergedSchemaPackings' erase+re-add land on
  //     the live version.
  ASSERT_EQ(table_info->schema_version, kSourceAlteredVersion);
  ASSERT_EQ(table_info->schema_version, snapshot_pb.schema_version());
  // (b) The two schemas really differ, so a rebind is observable at all.
  Schema snapshot_schema;
  ASSERT_OK(SchemaFromPB(snapshot_pb.schema(), &snapshot_schema));
  ASSERT_FALSE(snapshot_schema.Equals(table_info->schema()));
  // (c) Before the merge the binding is sound: version N+1 describes the live schema.
  {
    const dockv::SchemaPacking& before =
        ASSERT_RESULT(storage.GetPacking(table_info->schema_version));
    ASSERT_TRUE(before.SchemaContainsPacking(YQL_TABLE_TYPE, table_info->schema()));
  }
  // (d) The first of MergeSchemaPackings' two DFATAL guards, `schema_version < pb.schema_version()`
  //     (tablet_metadata.cc:381-385), does not fire here: the versions are equal. Only the second
  //     one does, and it is fail-open.
  ASSERT_FALSE(table_info->schema_version < snapshot_pb.schema_version());

  const auto merge_status = table_info->doc_read_context->MergeWithRestored(
      snapshot_pb, dockv::OverwriteSchemaPacking::kTrue);
  LOG(INFO) << "MergeWithRestored returned: " << merge_status;

  // THE ASSERTION. Correct behaviour has two acceptable shapes: the merge refuses an input that
  // would redefine the tablet's current schema version, or it leaves that version still
  // describing the live schema. What must not happen -- and what happens at the pin -- is
  // Status::OK() plus a redefined version, which is exactly the state
  // TableInfo::MergeSchemaPackings' own DFATAL declares invalid and then lets through.
  if (merge_status.ok()) {
    const dockv::SchemaPacking& merged =
        ASSERT_RESULT(storage.GetPacking(table_info->schema_version));
    ASSERT_TRUE(merged.SchemaContainsPacking(YQL_TABLE_TYPE, table_info->schema()))
        << "schema version " << table_info->schema_version << " was rebound to "
        << merged.ToString() << " while the tablet's live schema is "
        << table_info->schema().ToString();
  }
}

// The persistence half: whatever the merge binds to the current schema version must survive a
// superblock round trip. TableInfo::ToPB -> DocReadContext::ToPB skips the current version's
// packing (schema_packing.cc ToPB, `skip_schema_version`), and DocReadContext::LoadFromPB
// regenerates it with AddSchema(pb.schema_version(), schema_) from the LIVE schema -- so a
// rebound version silently means one thing before the next tablet open and another after it.
TEST_F(YcqlRestoreSchemaSkewTest, MergedCurrentVersionPackingSurvivesSuperblockRoundTrip) {
  const auto snapshot_pb = PrepareRestoreSide(kSourceAlteredVersion);
  auto* meta = harness_->tablet()->metadata();
  const auto current_version = meta->primary_table_info()->schema_version;
  ASSERT_EQ(current_version, snapshot_pb.schema_version());

  const auto merge_status = meta->primary_table_info()->doc_read_context->MergeWithRestored(
      snapshot_pb, dockv::OverwriteSchemaPacking::kTrue);
  LOG(INFO) << "MergeWithRestored returned: " << merge_status;

  const auto before_reload = PackingAt(meta->primary_table_info(), current_version);

  // The tablet is shut down first for the same reason TestLoadFromSuperBlock does it: the
  // round trip replaces the metadata's live TableInfo objects.
  writer_.reset();
  harness_->tablet()->StartShutdown(DisableFlushOnShutdown::kFalse, AbortOps::kFalse);
  harness_->tablet()->CompleteShutdown();

  RaftGroupReplicaSuperBlockPB superblock;
  meta->ToSuperBlock(&superblock);
  ASSERT_OK(meta->ReplaceSuperBlock(superblock));

  const auto after_reload = PackingAt(meta->primary_table_info(), current_version);

  // THE ASSERTION.
  ASSERT_EQ(before_reload, after_reload);
}

// Control: the geometry the YSQL export-side repair produces. RepackSnapshotsForBackup raises the
// exported SysTablesEntryPB.version to the master's current version, so ImportTableEntry's
// meta.version() + 1 lands one above every version in the snapshot and nothing collides. Passes
// at the pin, and shows the merge machinery is doing real work rather than no-oping: the
// snapshot's own packing for its version is installed, the live version is untouched, and the
// binding survives the round trip.
TEST_F(YcqlRestoreSchemaSkewTest, RepairedSnapshotVersionLeavesCurrentPackingIntact) {
  const auto snapshot_pb = PrepareRestoreSide(kSourceAlteredVersion + 1);
  auto* meta = harness_->tablet()->metadata();
  auto table_info = meta->primary_table_info();
  auto& storage = table_info->doc_read_context->schema_packing_storage;
  const auto current_version = table_info->schema_version;

  ASSERT_EQ(current_version, kSourceAlteredVersion + 1);
  ASSERT_GT(current_version, snapshot_pb.schema_version());
  ASSERT_FALSE(storage.HasVersion(snapshot_pb.schema_version()));

  ASSERT_OK(table_info->doc_read_context->MergeWithRestored(
      snapshot_pb, dockv::OverwriteSchemaPacking::kTrue));

  // The merge really ran: the snapshot's post-ALTER packing is now present at its own version.
  ASSERT_TRUE(storage.HasVersion(snapshot_pb.schema_version()));
  {
    const dockv::SchemaPacking& snapshot_packing =
        ASSERT_RESULT(storage.GetPacking(snapshot_pb.schema_version()));
    ASSERT_TRUE(snapshot_packing.SchemaContainsPacking(YQL_TABLE_TYPE, post_alter_schema_));
  }

  // And the live version still describes the live schema.
  {
    const dockv::SchemaPacking& merged = ASSERT_RESULT(storage.GetPacking(current_version));
    ASSERT_TRUE(merged.SchemaContainsPacking(YQL_TABLE_TYPE, table_info->schema()));
  }

  const auto before_reload = PackingAt(meta->primary_table_info(), current_version);

  writer_.reset();
  harness_->tablet()->StartShutdown(DisableFlushOnShutdown::kFalse, AbortOps::kFalse);
  harness_->tablet()->CompleteShutdown();

  RaftGroupReplicaSuperBlockPB superblock;
  meta->ToSuperBlock(&superblock);
  ASSERT_OK(meta->ReplaceSuperBlock(superblock));

  ASSERT_EQ(before_reload, PackingAt(meta->primary_table_info(), current_version));
}

} // namespace tablet
} // namespace yb

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

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "yb/util/logging.h"
#include <gtest/gtest.h>

#include "yb/common/ql_protocol_util.h"
#include "yb/common/schema.h"

#include "yb/docdb/ql_rowwise_iterator_interface.h"

#include "yb/dockv/partial_row.h"
#include "yb/dockv/reader_projection.h"

#include "yb/gutil/strings/numbers.h"
#include "yb/gutil/strings/substitute.h"

#include "yb/docdb/doc_read_context.h"

#include "yb/rocksdb/db.h"
#include "yb/rocksdb/options.h"

#include "yb/tablet/local_tablet_writer.h"
#include "yb/tablet/tablet-test-harness.h"
#include "yb/tablet/tablet-test-util.h"
#include "yb/tablet/tablet.h"
#include "yb/tablet/tablet_metadata.h"

#include "yb/util/debug.h"
#include "yb/util/env.h"
#include "yb/util/flags.h"
#include "yb/util/scope_exit.h"
#include "yb/util/status_log.h"
#include "yb/util/test_macros.h"
#include "yb/util/test_util.h"

DECLARE_bool(TEST_dcheck_for_missing_schema_packing);
DECLARE_uint64(initial_seqno);

using std::string;
using std::vector;

using strings::Substitute;

namespace yb {
namespace tablet {

class TestTabletSchema : public YBTabletTest {
 public:
  TestTabletSchema()
    : YBTabletTest(CreateBaseSchema(), YQL_TABLE_TYPE) {
  }

  void InsertRows(int32_t first_key, int32_t nrows) {
    for (int32_t i = first_key; i < nrows; ++i) {
      InsertRow(i);
      if (i == (nrows / 2)) {
        ASSERT_OK(tablet()->Flush(tablet::FlushMode::kSync, rocksdb::FlushReason::kTestOnly));
      }
    }
  }

  void InsertRow(int32_t key) {
    LocalTabletWriter writer(tablet());
    QLWriteRequestPB req;
    QLAddInt32HashValue(&req, key);
    QLAddInt32ColumnValue(&req, kFirstColumnId + 1, key);
    ASSERT_OK(writer.Write(&req));
  }

  void DeleteRow(int32_t key) {
    LocalTabletWriter writer(tablet());
    QLWriteRequestPB req;
    req.set_type(QLWriteRequestPB::QL_STMT_DELETE);
    QLAddInt32HashValue(&req, key);
    ASSERT_OK(writer.Write(&req));
  }

  void MutateRow(int32_t key, int32_t col_idx, int32_t new_val) {
    LocalTabletWriter writer(tablet());
    QLWriteRequestPB req;
    QLAddInt32HashValue(&req, key);
    QLAddInt32ColumnValue(&req, kFirstColumnId + col_idx, new_val);
    ASSERT_OK(writer.Write(&req));
  }

  void VerifyTabletRows(const std::vector<std::pair<string, string> >& keys) {
    typedef std::pair<string, string> StringPair;

    vector<string> rows;
    ASSERT_OK(DumpTablet(*tablet(), &rows));
    std::sort(rows.begin(), rows.end());
    for (const string& row : rows) {
      bool found = false;
      for (const StringPair& k : keys) {
        if (row.find(k.first) != string::npos) {
          ASSERT_STR_CONTAINS(row, k.second);
          found = true;
          break;
        }
      }
      ASSERT_TRUE(found) << "Row: " << row << ", keys: " << yb::ToString(keys);
    }
  }

 private:
  Schema CreateBaseSchema() {
    return Schema({ ColumnSchema("key", DataType::INT32, ColumnKind::HASH),
                    ColumnSchema("c1", DataType::INT32) });
  }
};

// Verify that RowIterator can still be used safely after schema change.
TEST_F(TestTabletSchema, TestRowIteratorWithAlterSchema) {
  std::atomic<bool> stop(false);
  dockv::ReaderProjection projection(*tablet()->metadata()->schema());
  auto iter = ASSERT_RESULT(tablet()->NewRowIterator(projection));
  std::thread thread([&stop, &iter] {
    // 1. Wait for AlterSchema to finish by sleeping 3 seconds.
    while (!stop.load(std::memory_order_acquire)) {
      SleepFor(MonoDelta::FromMilliseconds(100));
    }
    // 3. Previous schema context should be preserved even after schema change.
    iter->IsFetchedRowStatic();
  });
  SchemaBuilder builder2(*tablet()->metadata()->schema());
  ASSERT_OK(builder2.RenameColumn("c1", "c1_renamed"));
  // 2. Change schema when the other thread is waiting.
  AlterSchema(builder2.Build());
  stop.store(true, std::memory_order_release);
  thread.join();
}

// Write to the table using a projection schema with a renamed field.
TEST_F(TestTabletSchema, TestRenameProjection) {
  std::vector<std::pair<string, string> > keys;

  // Insert with the base schema
  InsertRow(1);

  // Switch schema to s2
  SchemaBuilder builder(*tablet()->metadata()->schema());
  ASSERT_OK(builder.RenameColumn("c1", "c1_renamed"));
  AlterSchema(builder.Build());
  Schema s2 = builder.BuildWithoutIds();

  // Insert with the s2 schema after AlterSchema(s2)
  InsertRow(2);

  // Read and verify using the s2 schema
  keys.clear();
  for (int i = 1; i <= 4; ++i) {
    keys.push_back(std::pair<string, string>(Substitute("{ int32_value: $0", i),
                                             Substitute("int32_value: $0 }", i)));
  }
  VerifyTabletRows(keys);

  // Delete the first two rows
  DeleteRow(/* key= */ 1);

  // Alter the remaining row
  MutateRow(/* key= */ 2, /* col_idx= */ 1, /* new_val= */ 6);

  // Read and verify using the s2 schema
  keys.clear();
  keys.push_back(std::pair<string, string>("{ int32_value: 2", "int32_value: 6 }"));
  VerifyTabletRows(keys);
}

// Verify that removing a column and re-adding it will not result in making old data visible
TEST_F(TestTabletSchema, TestDeleteAndReAddColumn) {
  std::vector<std::pair<string, string> > keys;

  // Insert and Mutate with the base schema
  InsertRow(1);
  MutateRow(/* key= */ 1, /* col_idx= */ 1, /* new_val= */ 2);

  keys.clear();
  keys.push_back(std::pair<string, string>("{ int32_value: 1", "int32_value: 2 }"));
  VerifyTabletRows(keys);

  // Switch schema to s2
  SchemaBuilder builder(*tablet()->metadata()->schema());
  ASSERT_OK(builder.RemoveColumn("c1"));
  // NOTE this new 'c1' will have a different id from the previous one
  //      so the data added to the previous 'c1' will not be visible.
  ASSERT_OK(builder.AddNullableColumn("c1", DataType::INT32));
  AlterSchema(builder.Build());
  Schema s2 = builder.BuildWithoutIds();

  // Verify that the new 'c1' have the default value
  keys.clear();
  keys.push_back(std::pair<string, string>("{ int32_value: 1", "null }"));
  VerifyTabletRows(keys);
}

// packed_row:INV-2 - a schema packing may be deleted only once the census proves no committed row
// still needs it. rocksdb::VersionSet::Import (rocksdb/db/version_set.cc:3296-3300) resets
// smallest.user_frontier / largest.user_frontier on every file it adopts, and
// RegularRocksDbListener::FillMinSchemaVersion (tablet/tablet.cc:726-732) skips any live file
// without a smallest.user_frontier, so packed rows that arrived by Tablet::ImportData
// (tablet.cc:2725-2728, the entry point yb-bulk_load drives) contribute nothing to the census that
// authorises OldSchemaGC. Note the opposite polarity two thousand lines away in the same file:
// Tablet::CompactionHybridTimeConstraints (tablet.cc:5749-5756) treats a frontier-less file as an
// unbounded constraint and names bulk load in its comment.
TEST_F(TestTabletSchema, ImportedFileInvisibleToPackingGc) {
  // The packing this test is about goes missing; without this the observable is a CHECK abort in
  // schema_packing.cc:873 rather than the behaviour under test.
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_TEST_dcheck_for_missing_schema_packing) = false;

  const auto imported_version = tablet()->metadata()->primary_table_schema_version();

  // (1) Build a donor DB the way yb-bulk_load does: a separate tablet with the identical schema in
  // its own fs root, rows packed at `imported_version`, flushed and compacted into SSTs, then shut
  // down so its files are quiescent.
  // The donor's RocksDB sequence numbers must be disjoint from, and below, the importing tablet's:
  // VersionSet::Import rejects a donor whose largest seqno is not below the target's last sequence
  // (version_set.cc:3299-3305) and rejects overlapping seqno ranges (:3325-3337). Every YB RocksDB
  // otherwise starts at the same FLAGS_initial_seqno (1 << 50, docdb_rocksdb_util.cc:170), so two
  // tablets built in one process always collide. This only makes the import constructible; nothing
  // about the census or the frontiers depends on it.
  const auto saved_initial_seqno = FLAGS_initial_seqno;
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_initial_seqno) = 1000;
  auto restore_seqno = ScopeExit([saved_initial_seqno] {
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_initial_seqno) = saved_initial_seqno;
  });

  std::string donor_dir;
  TabletTestHarness::Options donor_opts(GetTestPath("donor_root"));
  donor_opts.table_type = YQL_TABLE_TYPE;
  // A distinct id: Tablet's ctor registers a "tablet-<id>" MemTracker, and reusing the default id
  // would collide with the tablet under test.
  donor_opts.tablet_id = "donor_tablet_id";
  {
    TabletTestHarness donor(schema(), donor_opts);
    ASSERT_OK(donor.Create(/* first_time= */ true));
    ASSERT_OK(donor.Open());
    for (int32_t key = 1; key <= 5; ++key) {
      LocalTabletWriter writer(donor.tablet());
      QLWriteRequestPB req;
      QLAddInt32HashValue(&req, key);
      QLAddInt32ColumnValue(&req, kFirstColumnId + 1, key);
      ASSERT_OK(writer.Write(&req));
    }
    ASSERT_OK(donor.tablet()->Flush(FlushMode::kSync, rocksdb::FlushReason::kTestOnly));
    ASSERT_OK(donor.tablet()->ForceManualRocksDBCompact());
    donor_dir = donor.tablet()->metadata()->rocksdb_dir();
    LOG(INFO) << "donor dir: " << donor_dir;
    for (const auto& file : donor.tablet()->regular_db()->GetLiveFilesMetaData()) {
      LOG(INFO) << "donor file " << file.name_id << " seqno [" << file.smallest.seqno << ", "
                << file.largest.seqno << "] frontier "
                << (file.smallest.user_frontier.get() != nullptr);
    }
    donor.tablet()->StartShutdown(DisableFlushOnShutdown::kTrue, AbortOps::kTrue);
    donor.tablet()->CompleteShutdown();
  }
  // Reopen the donor once, the way BulkLoad::CompactFiles does ("Reopen rocksdb to clean up
  // deleted files", tools/yb-bulk_load.cc:410-411): VersionSet::Import refuses a source manifest
  // that still carries the compaction's file deletions (version_set.cc:3294).
  {
    TabletTestHarness donor(schema(), donor_opts);
    ASSERT_OK(donor.Create(/* first_time= */ false));
    ASSERT_OK(donor.Open());
    for (const auto& file : donor.tablet()->regular_db()->GetLiveFilesMetaData()) {
      LOG(INFO) << "donor file after reopen " << file.name_id << " seqno [" << file.smallest.seqno
                << ", " << file.largest.seqno << "]";
    }
    donor.tablet()->StartShutdown(DisableFlushOnShutdown::kTrue, AbortOps::kTrue);
    donor.tablet()->CompleteShutdown();
  }

  // (2) Advance the importing tablet's sequence numbers past the donor's; Import refuses a donor
  // whose largest seqno is not below the target's last sequence.
  for (int32_t key = 900; key <= 910; ++key) {
    DeleteRow(key);
  }
  ASSERT_OK(tablet()->ImportData(donor_dir));

  // Precondition: the imported rows read back, i.e. the packing for `imported_version` is needed.
  std::vector<std::pair<string, string>> imported_keys;
  for (int32_t key = 1; key <= 5; ++key) {
    imported_keys.emplace_back(
        Substitute("{ int32_value: $0", key), Substitute("int32_value: $0 }", key));
  }
  ASSERT_NO_FATALS(VerifyTabletRows(imported_keys));
  ASSERT_TRUE(tablet()->metadata()->primary_table_info()->doc_read_context
                  ->schema_packing_storage.HasVersion(imported_version));

  // (3) ALTER, so the tablet moves to imported_version + 1.
  SchemaBuilder builder(*tablet()->metadata()->schema());
  ASSERT_OK(builder.AddNullableColumn("c2", DataType::INT32));
  AlterSchema(builder.Build());
  const auto new_version = tablet()->metadata()->primary_table_schema_version();
  ASSERT_GT(new_version, imported_version);

  // (4) Rows packed at the new version, in their own SST, whose frontier records only the new
  // version.
  for (int32_t key = 1000; key <= 1005; ++key) {
    InsertRow(key);
  }
  ASSERT_OK(tablet()->Flush(FlushMode::kSync, rocksdb::FlushReason::kTestOnly));

  // (5) Compact only the files the imported one is not among, so the census that
  // OnCompactionCompleted -> OldSchemaGC runs on never sees the imported rows. A whole-tablet
  // compaction would pull the imported file in and mask the defect.
  std::vector<std::string> non_imported_files;
  size_t num_imported_files = 0;
  for (const auto& file : tablet()->regular_db()->GetLiveFilesMetaData()) {
    LOG(INFO) << "live file " << file.name_id << " imported=" << file.imported << " frontier="
              << (file.smallest.user_frontier.get() != nullptr);
    if (file.imported) {
      ++num_imported_files;
    } else {
      non_imported_files.push_back(file.Name());
    }
  }
  // Preconditions on the geometry: the donor's file is live and adopted, and there is something
  // else to compact without it.
  ASSERT_EQ(num_imported_files, 1);
  ASSERT_FALSE(non_imported_files.empty());

  // Blocked in a debug / fastdebug build before the assertion below can be reached, by a defect
  // next door to the one under test. Tablet::CompactionHybridTimeConstraints has two branches for
  // a live file with no frontier: for compaction *inputs* (tablet.cc:5749-5756) it fails closed
  // and tolerates imported files - LOG_IF(DFATAL, !file->imported), taught to do so by
  // 38009b35be1 (2026-07-18, [#32691]) - but for every *other* live file (tablet.cc:5799-5803) the
  // LOG(DFATAL) is unconditional and got no such guard. The imported file is by construction never
  // an input here, so any compaction on this tablet aborts a debug process:
  //   F tablet.cc:5802] Other file without frontier: { ... name_id: 12 ... imported: 1 ...
  //     smallest: { seqno: 0 user_frontier: <NULL> } ... }
  // DFATAL has no runtime knob (YB_GLOG_SEVERITY_DFATAL is google::GLOG_FATAL at compile time when
  // !NDEBUG, util/logging.h:166-171), so the remainder needs a release build.
  if (kIsDebug) {
    GTEST_SKIP() << "compaction over a tablet holding an imported file aborts a debug build in "
                    "Tablet::CompactionHybridTimeConstraints (tablet.cc:5802); run this case in a "
                    "release build";
  }

  ASSERT_OK(tablet()->regular_db()->CompactFiles(
      rocksdb::CompactionOptions(), non_imported_files, /* output_level= */ 0));

  // The single assertion: the packing the imported rows are encoded at must survive.
  ASSERT_TRUE(tablet()->metadata()->primary_table_info()->doc_read_context
                  ->schema_packing_storage.HasVersion(imported_version))
      << "packing " << imported_version << " was GC'd although imported SSTs still hold rows "
      << "packed at it; storage now holds "
      << tablet()->metadata()->primary_table_info()->doc_read_context
             ->schema_packing_storage.VersionsToString();
}

} // namespace tablet
} // namespace yb

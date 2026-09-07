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

// Correctness of the per-key tombstone drop in partial compactions
// (docdb_reclamation_tombstone_drop): a tombstone past the history cutoff may leave a compaction
// that does not include every live file only when the bloom filters of the live files outside the
// compaction that could hold older data prove its key absent, and only when the memtable and
// running-transaction floor allows it. Every case compacts a single file with the production
// hybrid-time constraints and checks the resulting entries.

#include <gtest/gtest.h>

#include "yb/common/ql_value.h"

#include "yb/docdb/docdb_test_base.h"

#include "yb/dockv/doc_key.h"
#include "yb/dockv/doc_path.h"

#include "yb/rocksdb/db.h"
#include "yb/rocksdb/table_properties.h"

#include "yb/util/flags.h"
#include "yb/util/test_macros.h"

DECLARE_bool(docdb_reclamation_tombstone_drop);
DECLARE_int32(docdb_reclamation_max_probe_files);

namespace yb::docdb {

class ReclamationCompactionTest : public DocDBTestBase {
 public:
  void SetUp() override {
    DocDBTestBase::SetUp();
    if (!KeepUniversalCompactionStyle()) {
      ASSERT_OK(DisableCompactions());
    }
    // Writes carry consensus frontiers only while op_id_ is non-empty.
    op_id_.term = 1;
    op_id_.index = 0;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_tombstone_drop) = true;
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_max_probe_files) = 64;
  }

 protected:
  // DisableCompactions() switches the compaction style to "none", under which a compaction pins
  // its whole version. Tablets run universal compaction, which pins input files only; the
  // universal variant below keeps that style and turns the picker off instead.
  virtual bool KeepUniversalCompactionStyle() const { return false; }

  Schema CreateSchema() override { return Schema(); }

  Status InitRocksDBOptions() override {
    RETURN_NOT_OK(DocDBRocksDBFixture::InitRocksDBOptions());
    UseProductionCompactionHybridTimeConstraints();
    if (KeepUniversalCompactionStyle()) {
      regular_db_options_.disable_auto_compactions = true;
    }
    return Status::OK();
  }

  static dockv::KeyBytes Key(int64_t id) {
    return dockv::DocKey(
        static_cast<dockv::DocKeyHash>(id), {dockv::KeyEntryValue::Int64(id)}).Encode();
  }

  static HybridTime Ht(int64_t micros) { return HybridTime::FromMicros(micros); }

  Status Insert(int64_t id, int64_t ht_micros) {
    return SetPrimitive(
        dockv::DocPath(Key(id), dockv::KeyEntryValue::MakeColumnId(ColumnId(1))),
        QLValue::Primitive("v"), Ht(ht_micros));
  }

  Status DeleteRow(int64_t id, int64_t ht_micros) {
    return DeleteSubDoc(dockv::DocPath(Key(id)), Ht(ht_micros));
  }

  Status DeleteColumn(int64_t id, int64_t ht_micros) {
    return DeleteSubDoc(
        dockv::DocPath(Key(id), dockv::KeyEntryValue::MakeColumnId(ColumnId(1))), Ht(ht_micros));
  }

  Status Flush() { return FlushRocksDbAndWait(rocksdb::FlushReason::kTestOnly); }

  // Compacts the newest file alone, with the history cutoff at `cutoff_micros`.
  Status CompactNewestFile(int64_t cutoff_micros) {
    rocksdb::ColumnFamilyMetaData cf_meta;
    regular_db_->GetColumnFamilyMetaData(&cf_meta);
    const rocksdb::SstFileMetaData* newest = nullptr;
    for (const auto& file : cf_meta.levels[0].files) {
      if (!newest || file.name_id > newest->name_id) {
        newest = &file;
      }
    }
    SCHECK_NOTNULL(newest);
    SetHistoryCutoffHybridTime(Ht(cutoff_micros));
    auto status = regular_db_->CompactFiles(
        rocksdb::CompactionOptions(), {newest->Name()}, /* output_level= */ 0);
    SetHistoryCutoffHybridTime(HybridTime::kMin);
    return status;
  }

  Result<uint64_t> NumEntries() {
    rocksdb::TablePropertiesCollection props;
    RETURN_NOT_OK(regular_db_->GetPropertiesOfAllTables(&props));
    uint64_t result = 0;
    for (const auto& [_, table_props] : props) {
      result += table_props->num_entries;
    }
    return result;
  }

  bool DumpHasTombstone() {
    return DocDBDebugDumpToStr().find("DEL") != std::string::npos;
  }
};

// The pair lives in the newest file and the key exists nowhere else: both halves go.
TEST_F(ReclamationCompactionTest, DropsPairWhenNoOlderDataOutside) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_FALSE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  ASSERT_EQ(ASSERT_RESULT(NumEntries()), 1);
}

// An older version of the key sits in a file outside the compaction: the probe hits and the
// tombstone stays, while the newer value it shadows is still removed.
TEST_F(ReclamationCompactionTest, KeepsTombstoneWhenOlderFileHasKey) {
  ASSERT_OK(Insert(1, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  ASSERT_EQ(ASSERT_RESULT(NumEntries()), 2);
}

// The memtable holds an entry below the tombstone's hybrid time: the non-file floor cannot be
// probed, so the tombstone stays even though no file outside the compaction has the key.
TEST_F(ReclamationCompactionTest, KeepsTombstoneWhenMemtableIsBelowIt) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(3, 500));  // stays in the memtable

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
}

// More candidate files than the probe budget: no probing, tombstone stays.
TEST_F(ReclamationCompactionTest, KeepsTombstoneOverProbeBudget) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_max_probe_files) = 1;
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(3, 1500));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
}

// The flag off restores the coarse rule: any older file outside the compaction keeps it.
TEST_F(ReclamationCompactionTest, FlagOffKeepsTombstone) {
  ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_tombstone_drop) = false;
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
}

// A tombstone over an older tombstone in another file: the bloom filter cannot tell a tombstone
// from data, so the newer tombstone stays (accepted conservatism).
TEST_F(ReclamationCompactionTest, KeepsTombstoneOverOlderTombstone) {
  ASSERT_OK(Insert(1, 900));
  ASSERT_OK(DeleteRow(1, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
}

// A column tombstone uses the same doc-key probe: absent outside, it goes with the value.
TEST_F(ReclamationCompactionTest, DropsColumnTombstoneWhenNoOlderDataOutside) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteColumn(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_FALSE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  ASSERT_EQ(ASSERT_RESULT(NumEntries()), 1);
}

// Within the retention window nothing changes: the pair is too young to touch.
TEST_F(ReclamationCompactionTest, KeepsPairAboveHistoryCutoff) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(1500));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  ASSERT_EQ(ASSERT_RESULT(NumEntries()), 3);
}

class ReclamationCompactionUniversalTest : public ReclamationCompactionTest {
 protected:
  bool KeepUniversalCompactionStyle() const override { return true; }
};

// Under universal compaction the compaction holds no version, only files; the outside files
// must still be available to probe, or the tombstone would be kept for lack of a probe set.
TEST_F(ReclamationCompactionUniversalTest, DropsPairWhenNoOlderDataOutside) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_FALSE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  ASSERT_EQ(ASSERT_RESULT(NumEntries()), 1);
}

// And the probe still sees the older key in the file outside the compaction.
TEST_F(ReclamationCompactionUniversalTest, KeepsTombstoneWhenOlderFileHasKey) {
  ASSERT_OK(Insert(1, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  ASSERT_EQ(ASSERT_RESULT(NumEntries()), 2);
}

}  // namespace yb::docdb

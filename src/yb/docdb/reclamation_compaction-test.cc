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
#include "yb/util/metrics.h"
#include "yb/util/test_macros.h"

DECLARE_bool(docdb_reclamation_tombstone_drop);
DECLARE_int32(docdb_reclamation_max_probe_files);

METRIC_DEFINE_entity(tablet);

namespace yb::docdb {

// The feed's counters, read after a compaction to pin which rule decided.
struct ProbeCounts {
  int64_t dropped = 0;
  int64_t kept_probe_hit = 0;
  int64_t kept_fail_closed = 0;
  int64_t probes = 0;
  int64_t probes_disabled = 0;
};

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
    if (!metric_entity_) {
      MetricEntity::AttributeMap attrs;
      attrs["tablet_id"] = "test";
      attrs["table_name"] = "test";
      attrs["table_id"] = "test";
      metric_entity_ = METRIC_ENTITY_tablet.Instantiate(&metric_registry_, "test", attrs);
      metrics_ = CreateCompactionMetrics(metric_entity_);
    }
    UseProductionCompactionHybridTimeConstraints(metrics_);
    if (KeepUniversalCompactionStyle()) {
      regular_db_options_.disable_auto_compactions = true;
    }
    return Status::OK();
  }

  ProbeCounts Counts() const {
    return ProbeCounts {
      .dropped = metrics_.reclamation_tombstones_dropped->value(),
      .kept_probe_hit = metrics_.reclamation_tombstones_kept_probe_hit->value(),
      .kept_fail_closed = metrics_.reclamation_tombstones_kept_fail_closed->value(),
      .probes = metrics_.reclamation_probes->value(),
      .probes_disabled = metrics_.reclamation_probes_disabled->value(),
    };
  }

  static dockv::KeyBytes Key(int64_t id) {
    return dockv::DocKey(
        static_cast<dockv::DocKeyHash>(id), {dockv::KeyEntryValue::Int64(id)}).Encode();
  }

  // Keys of a colocated table: rows carry the colocation id and a hash part; the table-level
  // tombstone (TRUNCATE) is the colocation id alone.
  static constexpr ColocationId kColocationId = 7;

  static dockv::KeyBytes ColocatedKey(int64_t id) {
    return dockv::DocKey(
        kColocationId, static_cast<dockv::DocKeyHash>(id),
        {dockv::KeyEntryValue::Int64(id)}).Encode();
  }

  Status InsertColocated(int64_t id, int64_t ht_micros) {
    return SetPrimitive(
        dockv::DocPath(ColocatedKey(id), dockv::KeyEntryValue::MakeColumnId(ColumnId(1))),
        QLValue::Primitive("v"), Ht(ht_micros));
  }

  Status TruncateColocated(int64_t ht_micros) {
    return SetPrimitive(
        dockv::DocPath(dockv::DocKey(kColocationId).Encode()),
        ValueRef(dockv::ValueEntryType::kTombstone), Ht(ht_micros));
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

  MetricRegistry metric_registry_;
  scoped_refptr<MetricEntity> metric_entity_;
  CompactionMetrics metrics_;
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
  const auto counts = Counts();
  ASSERT_EQ(counts.dropped, 1);
  ASSERT_EQ(counts.probes, 1);  // one outside file
  ASSERT_EQ(counts.kept_probe_hit, 0);
  ASSERT_EQ(counts.kept_fail_closed, 0);
  ASSERT_EQ(counts.probes_disabled, 0);
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
  const auto counts = Counts();
  ASSERT_EQ(counts.kept_probe_hit, 1);
  ASSERT_EQ(counts.dropped, 0);
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
  ASSERT_EQ(Counts().probes_disabled, 1);
  ASSERT_EQ(Counts().probes, 0);
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

// A table-level tombstone (TRUNCATE of a colocated table) shadows every row of the table, but its
// own filter key is not the rows' filter key, so a probe miss proves nothing: it must stay.
TEST_F(ReclamationCompactionTest, KeepsTableTombstone) {
  ASSERT_OK(InsertColocated(1, 1000));
  ASSERT_OK(InsertColocated(2, 1001));
  ASSERT_OK(Flush());
  ASSERT_OK(TruncateColocated(2000));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  ASSERT_EQ(ASSERT_RESULT(NumEntries()), 3);
  const auto counts = Counts();
  ASSERT_EQ(counts.kept_fail_closed, 1);
  ASSERT_EQ(counts.probes, 0);
  ASSERT_EQ(counts.dropped, 0);
}

// A file's frontier hybrid time is the batch time of its writes; a write applied with an external
// hybrid time (xCluster, index backfill) lands below it. The older version here sits in a file
// whose frontier is newer than the tombstone but not newer than the tablet's last external
// write, so the frontier cannot exclude it and the probe must find it. (An older unrelated file
// anchors the coarse gate; without one the coarse gate alone drops the tombstone, which is a
// pre-existing gap of the frontier-based gate, not of the probes.)
TEST_F(ReclamationCompactionTest, KeepsTombstoneWhenOutsideFileFrontierIsAboveItsEntry) {
  ASSERT_OK(Insert(2, 500));
  ASSERT_OK(Flush());
  SetFrontierHybridTimeOverrideForTests(Ht(5000));
  ASSERT_OK(Insert(1, 1000));
  ASSERT_OK(Flush());
  SetFrontierHybridTimeOverrideForTests(std::nullopt);
  SetExternalWritesUpToForTests(Ht(5000));
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  ASSERT_EQ(ASSERT_RESULT(NumEntries()), 3);
  const auto counts = Counts();
  ASSERT_EQ(counts.kept_probe_hit, 1);
  ASSERT_EQ(counts.dropped, 0);
}

// Without external writes a file's frontier does bound its entries, and a file entirely newer
// than the tombstone need not be probed: newer data cannot be resurrected by dropping it.
TEST_F(ReclamationCompactionTest, SkipsNewerOutsideFileWithoutExternalWrites) {
  ASSERT_OK(Insert(2, 500));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(3, 4000));
  ASSERT_OK(Flush());
  rocksdb::ColumnFamilyMetaData cf_meta;
  regular_db_->GetColumnFamilyMetaData(&cf_meta);
  ASSERT_EQ(cf_meta.levels[0].files.size(), 3);
  // The middle file (by age) holds the pair.
  const auto& pair_file = cf_meta.levels[0].files[1];
  SetHistoryCutoffHybridTime(Ht(5000));
  ASSERT_OK(regular_db_->CompactFiles(
      rocksdb::CompactionOptions(), {pair_file.Name()}, /* output_level= */ 0));
  SetHistoryCutoffHybridTime(HybridTime::kMin);

  ASSERT_FALSE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  const auto counts = Counts();
  ASSERT_EQ(counts.dropped, 1);
  ASSERT_EQ(counts.probes, 1);  // the older file only; the newer one was skipped
}

// The memtable may still hold a write with an external hybrid time: its frontier is not a lower
// bound on what it contains, so per-key drops stay off for the whole compaction.
TEST_F(ReclamationCompactionTest, KeepsTombstoneWhenMemtableMayHoldExternalWrites) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(3, 2500));  // stays in the memtable, frontier 2500
  SetExternalWritesUpToForTests(Ht(2500));

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  const auto counts = Counts();
  ASSERT_EQ(counts.probes_disabled, 1);
  ASSERT_EQ(counts.probes, 0);
  ASSERT_EQ(counts.dropped, 0);
}

// Once every in-memory batch is newer than the last external write, the floor is trusted again.
TEST_F(ReclamationCompactionTest, DropsPairWhenExternalWritesLeftMemory) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(3, 2600));  // stays in the memtable, frontier 2600
  SetExternalWritesUpToForTests(Ht(2500));

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_FALSE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  const auto counts = Counts();
  ASSERT_EQ(counts.probes_disabled, 0);
  ASSERT_EQ(counts.dropped, 1);
}

// A memtable entry exactly at the tombstone's hybrid time is not below it: fail closed.
TEST_F(ReclamationCompactionTest, KeepsTombstoneWhenMemtableFloorEqualsIt) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(3, 2001));  // stays in the memtable

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  const auto counts = Counts();
  ASSERT_EQ(counts.kept_fail_closed, 1);
  ASSERT_EQ(counts.probes, 0);
}

// Index backfill asks compactions to retain delete markers: per-key drops are off entirely.
TEST_F(ReclamationCompactionTest, KeepsTombstoneWhileDeleteMarkersAreRetained) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());
  retention_policy_->SetRetainDeleteMarkersForTests(true);

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_TRUE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  const auto counts = Counts();
  ASSERT_EQ(counts.probes_disabled, 1);
  ASSERT_EQ(counts.probes, 0);
}

class ReclamationCompactionUniversalTest : public ReclamationCompactionTest {
 protected:
  bool KeepUniversalCompactionStyle() const override { return true; }
};

// Under universal compaction the compaction holds no version, only files; the job must hold the
// outside files itself, or the tombstone would be kept for lack of a probe set.
TEST_F(ReclamationCompactionUniversalTest, DropsPairWhenNoOlderDataOutside) {
  ASSERT_OK(Insert(2, 1000));
  ASSERT_OK(Flush());
  ASSERT_OK(Insert(1, 2000));
  ASSERT_OK(DeleteRow(1, 2001));
  ASSERT_OK(Flush());

  ASSERT_OK(CompactNewestFile(3000));
  ASSERT_FALSE(DumpHasTombstone()) << DocDBDebugDumpToStr();
  ASSERT_EQ(ASSERT_RESULT(NumEntries()), 1);
  ASSERT_EQ(Counts().dropped, 1);
  ASSERT_EQ(Counts().probes_disabled, 0);
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

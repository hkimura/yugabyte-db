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

// Microbenchmark: reclamation compaction (compacting only the garbage-heavy files) versus full
// compaction on an identical LSM state, for the queue / outbox pattern (rows inserted, then deleted
// after a lag) sitting on top of cold live data.
//
// Every arm runs its compactions through DB::CompactFiles on the calling thread, so cost is
// measured the same way for all of them: rocksdb tickers plus live-file metadata (CompactFiles
// does not report through the compaction listener). Automatic compactions are disabled so the
// file layout is exactly what the generator built, and the production hybrid-time constraints
// (ComputeCompactionHybridTimeConstraints) govern tombstone removal instead of the fixture's stub.
//
// Sizes default to something CI can run; real runs scale them up through the flags, e.g.
//   YB_EXTRA_GTEST_FLAGS='--reclamation_bench_cold_rows=1000000 --reclamation_bench_csv=/tmp/r.csv'

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <random>

#include <boost/container/small_vector.hpp>
#include <gtest/gtest.h>

#include "yb/common/ql_value.h"
#include "yb/common/schema.h"

#include "yb/docdb/doc_read_context.h"
#include "yb/docdb/docdb_compaction_context.h"
#include "yb/docdb/docdb_test_base.h"
#include "yb/docdb/key_bounds.h"

#include "yb/dockv/doc_key.h"
#include "yb/dockv/doc_path.h"
#include "yb/dockv/packed_row.h"
#include "yb/dockv/schema_packing.h"
#include "yb/dockv/value.h"

#include "yb/gutil/casts.h"

#include "yb/rocksdb/db.h"
#include "yb/rocksdb/db/db_impl.h"
#include "yb/rocksdb/db/filename.h"
#include "yb/rocksdb/listener.h"
#include "yb/rocksdb/statistics.h"
#include "yb/rocksdb/table/block_based_table_reader.h"
#include "yb/rocksdb/table_properties.h"

#include "yb/util/flags.h"
#include "yb/util/format.h"
#include "yb/util/logging.h"
#include "yb/util/metric_entity.h"
#include "yb/util/metrics.h"
#include "yb/util/monotime.h"
#include "yb/util/random_util.h"
#include "yb/util/size_literals.h"
#include "yb/util/status_format.h"
#include "yb/util/test_macros.h"
#include "yb/util/tsan_util.h"

using namespace yb::size_literals;

DECLARE_bool(docdb_reclamation_tombstone_drop);

DEFINE_NON_RUNTIME_int64(reclamation_bench_cold_rows, -1,
    "Rows inserted before the churn phase and never deleted. -1 = build-type default.");
DEFINE_NON_RUNTIME_int32(reclamation_bench_cold_files, 4,
    "Number of flushes the cold rows are spread across.");
DEFINE_NON_RUNTIME_int64(reclamation_bench_wave_rows, -1,
    "Rows inserted per churn wave. -1 = build-type default.");
DEFINE_NON_RUNTIME_int32(reclamation_bench_waves, 8, "Number of churn waves.");
DEFINE_NON_RUNTIME_double(reclamation_bench_delete_fraction, 1.0,
    "Fraction of each wave's rows that is deleted after the lag.");
DEFINE_NON_RUNTIME_string(reclamation_bench_lag_mode, "same_file",
    "Where a wave's deletes land relative to its inserts: same_file, next_file or far.");
DEFINE_NON_RUNTIME_int32(reclamation_bench_payload_bytes, 1024, "Value payload size per row.");
DEFINE_NON_RUNTIME_int32(reclamation_bench_group_files, 2,
    "Maximum number of contiguous candidate files per reclamation compaction.");
DEFINE_NON_RUNTIME_int32(reclamation_bench_point_reads, 200,
    "Point reads of live ids and of deleted ids issued after each arm.");
DEFINE_NON_RUNTIME_string(reclamation_bench_csv, "",
    "If set, one row per arm is appended to this CSV file.");
DEFINE_NON_RUNTIME_int32(reclamation_bench_steady_waves, -1,
    "Waves in the steady-state case. -1 = reclamation_bench_waves.");
DEFINE_NON_RUNTIME_int32(reclamation_bench_full_every, 4,
    "Steady-state case: the periodic full compaction policy compacts every N waves.");
DEFINE_NON_RUNTIME_string(reclamation_bench_steady_csv, "",
    "If set, the steady-state case appends one row per policy and wave to this CSV file.");
DEFINE_NON_RUNTIME_int64(reclamation_bench_seed, 20260906, "Seed for the payload generator.");
DEFINE_NON_RUNTIME_double(reclamation_bench_candidate_ratio, 0.3,
    "A churn file is a reclamation candidate when its reclaimable entries (tombstones plus values "
    "shadowed inside the same file) divided by its entries reach this ratio; a group with no "
    "candidate is not compacted, as the production trigger would not fire on it.");
DEFINE_NON_RUNTIME_string(reclamation_bench_key_shape, "bucket",
    "bucket: hash partition is one of 16 buckets and the id is the range component, so a bucket "
    "scan walks rows in insertion order and the bloom filter key is the bucket. unique: the id "
    "itself is the hashed component, so the bloom filter key is unique per row and the consumer "
    "scan degenerates to a full walk.");
DEFINE_NON_RUNTIME_string(reclamation_bench_value_format, "column",
    "column: a row is one column entry under its doc key. packed: a row is one packed-row entry "
    "at its doc key, the layout YSQL writes, which the compaction feed handles on a separate "
    "path.");
DEFINE_NON_RUNTIME_string(reclamation_bench_selection, "oracle",
    "oracle: the generator knows which files hold churn and their reclaimable fraction. "
    "properties: every file carries the reclaimable fraction a statistics collector computes at "
    "build time (an emulation registered on the fixture), candidates are read from those "
    "properties as the production trigger will, and contiguous candidates form groups.");
DEFINE_NON_RUNTIME_bool(reclamation_bench_auto_compactions, false,
    "Leave the universal compaction picker running while the layout is written, as in a tablet, "
    "instead of freezing the file layout. Requires --reclamation_bench_selection=properties.");
DEFINE_NON_RUNTIME_string(reclamation_bench_churn_kind, "delete",
    "delete: each wave inserts new rows and deletes a fraction of them (the queue pattern). "
    "update: each wave rewrites the same hot rows --reclamation_bench_updates_per_wave times and "
    "deletes nothing, so the garbage is superseded versions rather than tombstones.");
DEFINE_NON_RUNTIME_int32(reclamation_bench_updates_per_wave, 4,
    "update churn: versions written per hot row per wave.");

// The docdb library declares the tablet metric entity and defines counters against it; the
// prototype itself lives in the tablet library, which this test does not link, so define it here
// as the util metric tests do.
METRIC_DEFINE_entity(tablet);

namespace yb::docdb {

namespace {

constexpr int64_t kDefaultColdRows = RegularBuildVsDebugVsSanitizers(20000, 2000, 500);
constexpr int64_t kDefaultWaveRows = RegularBuildVsDebugVsSanitizers(5000, 500, 100);
constexpr uint64_t kIdMixer = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kVersionMixer = 0xC2B2AE3D27D4EB4FULL;
const ColumnId kValueColumn(1);

// Queue-shaped keys: the hash partition is the bucket (id modulo kQueueBuckets) and the id is the
// range component, so within a bucket rows sort in insertion order and a consumer walking a bucket
// meets the oldest rows first. Cold rows get their own buckets (a distinct hashed component), so
// they never interleave with a queue bucket's dead prefix.
constexpr int32_t kQueueBuckets = 16;
constexpr int32_t kColdBucketOffset = 1 << 20;

enum class LagMode { kSameFile, kNextFile, kFar };

Result<LagMode> ParseLagMode(const std::string& mode) {
  if (mode == "same_file") return LagMode::kSameFile;
  if (mode == "next_file") return LagMode::kNextFile;
  if (mode == "far") return LagMode::kFar;
  return STATUS_FORMAT(InvalidArgument, "Unknown lag mode: $0", mode);
}

enum class KeyShape { kBucket, kUnique };

enum class ValueFormat { kColumn, kPacked };

Result<ValueFormat> ParseValueFormat(const std::string& format) {
  if (format == "column") return ValueFormat::kColumn;
  if (format == "packed") return ValueFormat::kPacked;
  return STATUS_FORMAT(InvalidArgument, "Unknown value format: $0", format);
}

Result<KeyShape> ParseKeyShape(const std::string& shape) {
  if (shape == "bucket") return KeyShape::kBucket;
  if (shape == "unique") return KeyShape::kUnique;
  return STATUS_FORMAT(InvalidArgument, "Unknown key shape: $0", shape);
}

enum class Selection { kOracle, kProperties };

Result<Selection> ParseSelection(const std::string& selection) {
  if (selection == "oracle") return Selection::kOracle;
  if (selection == "properties") return Selection::kProperties;
  return STATUS_FORMAT(InvalidArgument, "Unknown selection: $0", selection);
}

enum class ChurnKind { kDelete, kUpdate };

Result<ChurnKind> ParseChurnKind(const std::string& kind) {
  if (kind == "delete") return ChurnKind::kDelete;
  if (kind == "update") return ChurnKind::kUpdate;
  return STATUS_FORMAT(InvalidArgument, "Unknown churn kind: $0", kind);
}

// Emulation of the SST statistics collector at file-build time: counts the entries of a file and
// the subset the collector defines as reclaimable (versions shadowed by a newer entry of the same
// key inside the file, plus every entry of a row whose newest entry is a tombstone), and stores
// both as user properties. Selection can then read per-file statistics from any file, including
// compaction outputs, the way the production trigger will. A row is taken as dead when its first
// entry is a row-level tombstone; row-level entries sort before column entries whatever their
// hybrid times, so a row re-inserted column by column after a row delete would count as dead
// here. The benchmark layouts never reuse a key, so this does not arise.
class BenchStatsCollector : public rocksdb::TablePropertiesCollector {
 public:
  static constexpr const char* kEntriesProperty = "reclamation_bench.entries";
  static constexpr const char* kReclaimableProperty = "reclamation_bench.reclaimable";

  Status AddUserKey(
      const Slice& key, const Slice& value, rocksdb::EntryType, rocksdb::SequenceNumber,
      uint64_t) override {
    ++entries_;
    boost::container::small_vector<size_t, 16> ends;
    RETURN_NOT_OK(dockv::SubDocKey::DecodeDocKeyAndSubKeyEnds(key, &ends));
    if (ends.size() < 2) {
      return Status::OK();  // meta key or unexpected shape: never counted as reclaimable
    }
    // ends[1] is the end of the doc key, ends.back() the end of the last subkey (the hybrid time
    // follows), so the key up to ends.back() identifies the version chain.
    const Slice doc_key(key.data(), ends[1]);
    const Slice key_without_ht(key.data(), ends.back());
    if (doc_key != Slice(current_doc_key_)) {
      current_doc_key_.assign(doc_key.cdata(), doc_key.size());
      prev_key_without_ht_.clear();
      // The first entry of a row is its newest row-level entry when it has one.
      const bool row_level = ends.size() == 2;
      const auto tombstoned = dockv::Value::IsTombstoned(value);
      row_dead_ = row_level && tombstoned.ok() && *tombstoned;
    }
    if (row_dead_ || key_without_ht == Slice(prev_key_without_ht_)) {
      ++reclaimable_;
    }
    prev_key_without_ht_.assign(key_without_ht.cdata(), key_without_ht.size());
    return Status::OK();
  }

  Status Finish(rocksdb::UserCollectedProperties* properties) override {
    (*properties)[kEntriesProperty] = std::to_string(entries_);
    (*properties)[kReclaimableProperty] = std::to_string(reclaimable_);
    return Status::OK();
  }

  rocksdb::UserCollectedProperties GetReadableProperties() const override {
    return {{kEntriesProperty, std::to_string(entries_)},
            {kReclaimableProperty, std::to_string(reclaimable_)}};
  }

  const char* Name() const override { return "ReclamationBenchStatsCollector"; }

 private:
  uint64_t entries_ = 0;
  uint64_t reclaimable_ = 0;
  std::string current_doc_key_;
  std::string prev_key_without_ht_;
  bool row_dead_ = false;
};

class BenchStatsCollectorFactory : public rocksdb::TablePropertiesCollectorFactory {
 public:
  rocksdb::TablePropertiesCollector* CreateTablePropertiesCollector(
      rocksdb::TablePropertiesCollectorFactory::Context) override {
    return new BenchStatsCollector();
  }

  const char* Name() const override { return "ReclamationBenchStatsCollectorFactory"; }
};

// Counts the compactions the universal picker runs on its own (CompactFiles does not report
// through the listener), so their bytes can be told apart from the policy's compactions in the
// automatic-compaction mode.
class AutoCompactionCounter : public rocksdb::EventListener {
 public:
  struct Snapshot {
    uint64_t compactions = 0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;

    Snapshot operator-(const Snapshot& rhs) const {
      return Snapshot {
        .compactions = compactions - rhs.compactions,
        .input_bytes = input_bytes - rhs.input_bytes,
        .output_bytes = output_bytes - rhs.output_bytes,
      };
    }
  };

  void OnCompactionCompleted(rocksdb::DB*, const rocksdb::CompactionJobInfo& info) override {
    compactions_.fetch_add(1);
    input_bytes_.fetch_add(info.stats.total_input_bytes);
    output_bytes_.fetch_add(info.stats.total_output_bytes);
  }

  Snapshot snapshot() const {
    return Snapshot {
      .compactions = compactions_.load(),
      .input_bytes = input_bytes_.load(),
      .output_bytes = output_bytes_.load(),
    };
  }

 private:
  std::atomic<uint64_t> compactions_{0};
  std::atomic<uint64_t> input_bytes_{0};
  std::atomic<uint64_t> output_bytes_{0};
};

// Feed counters of the per-key tombstone drop, read from the benchmark's own metric entity.
struct ProbeCounters {
  int64_t probes = 0;
  int64_t tombstones_dropped = 0;
  int64_t kept_probe_hit = 0;
  int64_t kept_fail_closed = 0;
  // Compactions that ran with the drop flag on but could not probe (budget, hold, floor).
  int64_t probes_disabled = 0;

  ProbeCounters operator-(const ProbeCounters& rhs) const {
    return ProbeCounters {
      .probes = probes - rhs.probes,
      .tombstones_dropped = tombstones_dropped - rhs.tombstones_dropped,
      .kept_probe_hit = kept_probe_hit - rhs.kept_probe_hit,
      .kept_fail_closed = kept_fail_closed - rhs.kept_fail_closed,
      .probes_disabled = probes_disabled - rhs.probes_disabled,
    };
  }
};

// NONE: no compaction, the baseline. FULL: one CompactFiles over every live file, the incumbent.
// RECLAIM: CompactFiles over groups of contiguous churn files only, tombstones governed by the
// coarse gate. RECLAIM_DROP: the same with docdb_reclamation_tombstone_drop on, so tombstones
// whose key is proven absent from the files outside the compaction are dropped too.
// RECLAIM_ANCHORED: RECLAIM_DROP plus, for every group, the file that currently holds the rows
// the group's tombstones delete (oracle selection only). It emulates a probe-guided extension
// applied one trigger at a time, and prices the population-2 layouts (far lag, update churn)
// where the plain arms reclaim nothing. Level-0 compactions are contiguous runs, so each group
// also takes every file between it and its anchor, including earlier groups' outputs.
// RECLAIM_RANGE: the same extension batched: one compaction over the run from the oldest anchor
// of any candidate to the newest candidate, the policy a picker would need to avoid rewriting
// the growing output once per group.
enum class Arm { kNone, kFull, kReclaim, kReclaimDrop, kReclaimAnchored, kReclaimRange };

const char* ArmName(Arm arm) {
  switch (arm) {
    case Arm::kNone: return "NONE";
    case Arm::kFull: return "FULL";
    case Arm::kReclaim: return "RECLAIM";
    case Arm::kReclaimDrop: return "RECLAIM_DROP";
    case Arm::kReclaimAnchored: return "RECLAIM_ANCHORED";
    case Arm::kReclaimRange: return "RECLAIM_RANGE";
  }
  return "?";
}

struct Tickers {
  uint64_t compact_read_bytes = 0;
  uint64_t compact_write_bytes = 0;
  uint64_t keys_dropped_user = 0;
  uint64_t keys_dropped_newer = 0;
  uint64_t keys_dropped_obsolete = 0;
  uint64_t db_next = 0;
  uint64_t db_seek = 0;
  uint64_t bloom_checked = 0;
  uint64_t bloom_useful = 0;
  uint64_t block_cache_miss = 0;
  uint64_t block_cache_filter_miss = 0;

  Tickers operator-(const Tickers& rhs) const {
    return Tickers {
      .compact_read_bytes = compact_read_bytes - rhs.compact_read_bytes,
      .compact_write_bytes = compact_write_bytes - rhs.compact_write_bytes,
      .keys_dropped_user = keys_dropped_user - rhs.keys_dropped_user,
      .keys_dropped_newer = keys_dropped_newer - rhs.keys_dropped_newer,
      .keys_dropped_obsolete = keys_dropped_obsolete - rhs.keys_dropped_obsolete,
      .db_next = db_next - rhs.db_next,
      .db_seek = db_seek - rhs.db_seek,
      .bloom_checked = bloom_checked - rhs.bloom_checked,
      .bloom_useful = bloom_useful - rhs.bloom_useful,
      .block_cache_miss = block_cache_miss - rhs.block_cache_miss,
      .block_cache_filter_miss = block_cache_filter_miss - rhs.block_cache_filter_miss,
    };
  }
};

struct FileState {
  size_t num_files = 0;
  uint64_t total_bytes = 0;
  uint64_t num_entries = 0;
};

struct ScanResult {
  size_t entries_visited = 0;
  int buckets_with_live_row = 0;
};

struct ArmResult {
  Arm arm = Arm::kNone;
  FileState before;
  FileState after;
  size_t compactions = 0;
  int64_t wall_micros = 0;
  int64_t cpu_micros = 0;
  Tickers compaction;
  ProbeCounters probe;
  // Compactions the universal picker ran on its own: while the layout was written, and while the
  // policy ran (zero unless --reclamation_bench_auto_compactions).
  AutoCompactionCounter::Snapshot auto_during_build;
  AutoCompactionCounter::Snapshot auto_during_policy;
  size_t cold_files_left = 0;
  // Consumer-style scan: for every queue bucket, seek to it and advance to the first live row.
  ScanResult scan_result;
  Tickers scan;
  // Point reads: `point_reads` live ids and `point_reads` deleted ids.
  Tickers point_reads_live;
  Tickers point_reads_deleted;
};

int64_t ThreadCpuMicros() {
  rusage usage;
  CHECK_EQ(0, getrusage(RUSAGE_THREAD, &usage));
  return (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000000LL +
         usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
}

}  // namespace

class ReclamationCompactionPerfTest : public DocDBTestBase {
 public:
  void SetUp() override {
    DocDBTestBase::SetUp();
    if (!FLAGS_reclamation_bench_auto_compactions) {
      // The generator decides the file layout; no automatic compaction may rearrange it.
      ASSERT_OK(DisableCompactions());
    }
    InitOpId();
  }

  // The fixture attaches consensus frontiers (op id + hybrid time) to a write batch only while
  // op_id_ is non-empty, and the production constraints computation reads those frontiers from
  // every file. Without them every file looks like a frontier-less import.
  void InitOpId() {
    op_id_.term = 1;
    op_id_.index = 0;
  }

 protected:
  size_t block_cache_size() const override { return 256_MB; }

  // The schema only matters for packed rows: its value column is what the row packer encodes and
  // what the compaction feed's packing provider hands back. Column entries need no schema. The
  // key columns stand in for either key shape (the packing covers value columns only).
  Schema CreateSchema() override {
    if (FLAGS_reclamation_bench_value_format != "packed") {
      return Schema();
    }
    return Schema(
        {ColumnSchema("h", DataType::INT32, ColumnKind::HASH),
         ColumnSchema("r", DataType::INT64, ColumnKind::RANGE_ASC_NULL_FIRST),
         ColumnSchema("v", DataType::STRING, ColumnKind::VALUE, Nullable::kTrue)},
        {ColumnId(10), ColumnId(11), kValueColumn});
  }

  Status InitRocksDBOptions() override {
    RETURN_NOT_OK(DocDBRocksDBFixture::InitRocksDBOptions());
    UseProductionCompactionConstraints();
    // Per-file statistics for property-driven selection, and a counter for the picker's own
    // compactions; both are inert unless the corresponding modes are on.
    regular_db_options_.table_properties_collector_factories.push_back(
        std::make_shared<BenchStatsCollectorFactory>());
    if (!auto_compaction_counter_) {
      auto_compaction_counter_ = std::make_shared<AutoCompactionCounter>();
    }
    regular_db_options_.listeners.push_back(auto_compaction_counter_);
    return Status::OK();
  }

  // In the automatic-compaction mode, lets the picker finish whatever the last flush or
  // compaction triggered, so measurements and manual compactions see a settled file set.
  Status SettleAutoCompactions() {
    if (!FLAGS_reclamation_bench_auto_compactions) {
      return Status::OK();
    }
    return down_cast<rocksdb::DBImpl*>(regular_db_.get())->TEST_WaitForCompact();
  }

  // The fixture's default provider returns a constant for other_min, a test stub that forbids
  // tombstone removal in every partial compaction. Install the production computation instead, so
  // partial compactions are governed by the memtable and live-file frontiers as in a tablet, with
  // real counters so probe activity can be reported. Must be re-applied after any
  // ReinitDBOptions() call, which rebuilds the factory.
  void UseProductionCompactionConstraints() {
    if (!metric_entity_) {
      MetricEntity::AttributeMap attrs;
      attrs["tablet_id"] = "bench";
      attrs["table_name"] = "bench";
      attrs["table_id"] = "bench";
      metric_entity_ = METRIC_ENTITY_tablet.Instantiate(&metric_registry_, "bench", attrs);
      compaction_metrics_ = CreateCompactionMetrics(metric_entity_);
    }
    UseProductionCompactionHybridTimeConstraints(compaction_metrics_);
  }

  ProbeCounters ReadProbeCounters() const {
    return ProbeCounters {
      .probes = compaction_metrics_.reclamation_probes->value(),
      .tombstones_dropped = compaction_metrics_.reclamation_tombstones_dropped->value(),
      .kept_probe_hit = compaction_metrics_.reclamation_tombstones_kept_probe_hit->value(),
      .kept_fail_closed = compaction_metrics_.reclamation_tombstones_kept_fail_closed->value(),
      .probes_disabled = compaction_metrics_.reclamation_probes_disabled->value(),
    };
  }

  // ---- workload -------------------------------------------------------------------------------

  struct Layout {
    std::vector<uint64_t> cold_files;   // file numbers, creation order
    std::vector<uint64_t> churn_files;  // file numbers, creation order
    // Per churn file: reclaimable entries (tombstones plus values shadowed inside the same file)
    // over all entries, the statistic the production trigger will select on.
    std::vector<double> churn_dead_fraction;
    // Per churn file: indexes (into churn_files) of the files holding the rows this file's
    // tombstones delete, or the previous versions its updates supersede. Empty when the pair is
    // inside the file itself.
    std::vector<std::vector<size_t>> churn_anchors;
    int64_t cold_rows = 0;
    int64_t churn_rows = 0;
    int64_t deleted_rows = 0;
    std::vector<int64_t> live_ids;      // sample for point reads
    std::vector<int64_t> deleted_ids;   // sample for point reads
    HybridTime cutoff;                  // "now" once every write is in
  };

  // Cold ids are the first cold_rows_ ids; everything after them is queue churn.
  int32_t BucketOf(int64_t id) const {
    const auto bucket = static_cast<int32_t>(id % kQueueBuckets);
    return id < cold_rows_ ? kColdBucketOffset + bucket : bucket;
  }

  static dockv::DocKeyHash BucketHash(int32_t bucket) {
    return static_cast<dockv::DocKeyHash>((static_cast<uint32_t>(bucket) * 2654435761u) >> 16);
  }

  dockv::KeyBytes EncodedKey(int64_t id) const {
    if (key_shape_ == KeyShape::kUnique) {
      const auto hash = static_cast<dockv::DocKeyHash>(
          (static_cast<uint64_t>(id) * kIdMixer) >> 48);
      return dockv::DocKey(hash, {dockv::KeyEntryValue::Int64(id)}).Encode();
    }
    const auto bucket = BucketOf(id);
    return dockv::DocKey(
        BucketHash(bucket), {dockv::KeyEntryValue::Int32(bucket)},
        {dockv::KeyEntryValue::Int64(id)}).Encode();
  }

  // Encoded prefix shared by every key of a bucket: the hash code and the hashed components.
  static Result<std::string> BucketPrefix(int32_t bucket) {
    const auto encoded =
        dockv::DocKey(BucketHash(bucket), {dockv::KeyEntryValue::Int32(bucket)}).Encode();
    const auto sizes = VERIFY_RESULT(
        dockv::DocKey::EncodedHashPartAndDocKeySizes(encoded.AsSlice()));
    return encoded.AsSlice().Prefix(sizes.hash_part_size).ToBuffer();
  }

  HybridTime NextHybridTime() { return HybridTime::FromMicros(next_ht_micros_++); }

  // A distinct pseudo-random payload per row and per version, so block compression sees no
  // repeats across rows or across the versions of an updated row, and byte counts reflect the
  // payload size rather than a small pool of reused strings. Version 0 is the insert.
  std::string Payload(int64_t id, int64_t version) const {
    std::mt19937_64 rng(
        FLAGS_reclamation_bench_seed ^ (static_cast<uint64_t>(id) * kIdMixer) ^
        (static_cast<uint64_t>(version) * kVersionMixer));
    return RandomHumanReadableString(payload_bytes_, &rng);
  }

  Status InsertRow(int64_t id, int64_t version = 0) {
    if (value_format_ == ValueFormat::kPacked) {
      const auto& packing = VERIFY_RESULT_REF(
          doc_read_context().schema_packing_storage.GetPacking(SchemaVersion(0)));
      dockv::RowPackerV2 packer(
          /* version= */ 0, packing, /* packed_size_limit= */ std::numeric_limits<int64_t>::max(),
          /* control_fields= */ Slice());
      RETURN_NOT_OK(packer.AddValue(kValueColumn, QLValue::Primitive(Payload(id, version))));
      const auto packed_row = VERIFY_RESULT(packer.Complete());
      return SetPrimitive(
          dockv::DocPath(EncodedKey(id)), dockv::ValueControlFields(), ValueRef(packed_row),
          NextHybridTime());
    }
    return SetPrimitive(
        dockv::DocPath(EncodedKey(id), dockv::KeyEntryValue::MakeColumnId(kValueColumn)),
        QLValue::Primitive(Payload(id, version)), NextHybridTime());
  }

  Status DeleteRow(int64_t id) {
    return DeleteSubDoc(dockv::DocPath(EncodedKey(id)), NextHybridTime());
  }

  std::vector<uint64_t> LiveFileNumbers() {
    std::vector<uint64_t> result;
    for (const auto& file : regular_db_->GetLiveFilesMetaData()) {
      result.push_back(file.name_id);
    }
    std::sort(result.begin(), result.end());
    return result;
  }

  // Flushes the memtable and returns the number of the file it produced. In the
  // automatic-compaction mode the picker may already have merged the new file away, in which case
  // the newest file that appeared is returned, or 0 when none did.
  Result<uint64_t> FlushToNewFile() {
    auto before = LiveFileNumbers();
    RETURN_NOT_OK(FlushRocksDbAndWait(rocksdb::FlushReason::kTestOnly));
    RETURN_NOT_OK(SettleAutoCompactions());
    auto after = LiveFileNumbers();
    std::vector<uint64_t> added;
    std::set_difference(
        after.begin(), after.end(), before.begin(), before.end(), std::back_inserter(added));
    if (FLAGS_reclamation_bench_auto_compactions) {
      return added.empty() ? 0 : added.back();
    }
    SCHECK_EQ(added.size(), 1, IllegalState, Format("Flush produced $0 files", added.size()));
    return added.front();
  }

  struct FileStats {
    uint64_t number = 0;
    uint64_t entries = 0;
    uint64_t reclaimable = 0;
  };

  // Live files in number order with the collector emulation's counts.
  Result<std::vector<FileStats>> ReadFileStats() {
    rocksdb::TablePropertiesCollection props;
    RETURN_NOT_OK(regular_db_->GetPropertiesOfAllTables(&props));
    std::vector<FileStats> result;
    for (const auto& [name, table_props] : props) {
      FileStats stats;
      stats.number = rocksdb::TableFileNameToNumber(name);
      stats.entries = table_props->num_entries;
      const auto& user = table_props->user_collected_properties;
      auto it = user.find(BenchStatsCollector::kReclaimableProperty);
      if (it != user.end()) {
        stats.reclaimable = std::stoull(it->second);
      }
      result.push_back(stats);
    }
    std::sort(result.begin(), result.end(), [](const FileStats& a, const FileStats& b) {
      return a.number < b.number;
    });
    return result;
  }

  // Property-driven selection: groups of up to group_files contiguous candidate files in
  // file-number order, a candidate being a file whose reclaimable fraction reaches the ratio.
  // Mirrors the production rule (marked files, adjacent ones batched); unmarked neighbours are
  // not pulled in, so an insert file next to its delete file is joined only by a normal merge.
  Result<std::vector<std::vector<uint64_t>>> SelectGroupsByProperties() {
    std::map<uint64_t, FileStats> stats_by_number;
    for (const auto& s : VERIFY_RESULT(ReadFileStats())) {
      stats_by_number[s.number] = s;
    }
    // Contiguity is in level-0 order (by sequence number, the metadata lists newest first), not
    // by file number: a compaction output has a high number but sits where its inputs were.
    rocksdb::ColumnFamilyMetaData cf_meta;
    regular_db_->GetColumnFamilyMetaData(&cf_meta);
    std::vector<FileStats> stats;
    for (auto it = cf_meta.levels[0].files.rbegin(); it != cf_meta.levels[0].files.rend(); ++it) {
      auto found = stats_by_number.find(it->name_id);
      if (found != stats_by_number.end()) {
        stats.push_back(found->second);
      }
    }
    const size_t group = std::max(1, FLAGS_reclamation_bench_group_files);
    std::vector<std::vector<uint64_t>> groups;
    std::vector<uint64_t> current;
    for (const auto& s : stats) {
      const bool candidate = s.entries > 0 &&
          static_cast<double>(s.reclaimable) / static_cast<double>(s.entries) >=
              FLAGS_reclamation_bench_candidate_ratio;
      if (candidate) {
        current.push_back(s.number);
        if (current.size() == group) {
          groups.push_back(current);
          current.clear();
        }
      } else if (!current.empty()) {
        groups.push_back(current);
        current.clear();
      }
    }
    if (!current.empty()) {
      groups.push_back(current);
    }
    return groups;
  }

  Result<Layout> BuildLayout() {
    Layout layout;
    const int64_t cold_rows = FLAGS_reclamation_bench_cold_rows >= 0
        ? FLAGS_reclamation_bench_cold_rows : kDefaultColdRows;
    const int64_t wave_rows = FLAGS_reclamation_bench_wave_rows >= 0
        ? FLAGS_reclamation_bench_wave_rows : kDefaultWaveRows;
    const int cold_files = std::max(1, FLAGS_reclamation_bench_cold_files);
    const int waves = std::max(1, FLAGS_reclamation_bench_waves);
    const auto lag_mode = VERIFY_RESULT(ParseLagMode(FLAGS_reclamation_bench_lag_mode));
    const double delete_fraction = std::clamp(FLAGS_reclamation_bench_delete_fraction, 0.0, 1.0);
    const size_t sample = std::max(1, FLAGS_reclamation_bench_point_reads);

    int64_t next_id = 0;
    cold_rows_ = cold_rows;

    // Cold phase: live rows that never change.
    if (cold_rows > 0) {
      for (int file = 0; file < cold_files; ++file) {
        const int64_t rows = cold_rows / cold_files + (file < cold_rows % cold_files ? 1 : 0);
        if (rows == 0) {
          continue;  // fewer cold rows than files: an empty memtable flushes nothing
        }
        for (int64_t i = 0; i < rows; ++i) {
          RETURN_NOT_OK(InsertRow(next_id));
          if (layout.live_ids.size() < sample) {
            layout.live_ids.push_back(next_id);
          }
          ++next_id;
        }
        layout.cold_files.push_back(VERIFY_RESULT(FlushToNewFile()));
      }
      layout.cold_rows = cold_rows;
    }

    if (churn_kind_ == ChurnKind::kUpdate) {
      // Update churn: every wave rewrites the same hot rows several times. Within a file the older
      // versions are shadowed; across files the previous wave's versions are the anchors.
      const int64_t updates = std::max(1, FLAGS_reclamation_bench_updates_per_wave);
      const int64_t hot_begin = next_id;
      for (int64_t i = 0; i < wave_rows && layout.live_ids.size() < sample; ++i) {
        layout.live_ids.push_back(hot_begin + i);
      }
      for (int wave = 0; wave < waves; ++wave) {
        for (int64_t version = 0; version < updates; ++version) {
          for (int64_t i = 0; i < wave_rows; ++i) {
            RETURN_NOT_OK(InsertRow(hot_begin + i, wave * updates + version));
          }
        }
        layout.churn_files.push_back(VERIFY_RESULT(FlushToNewFile()));
        layout.churn_dead_fraction.push_back(
            static_cast<double>(updates - 1) / static_cast<double>(updates));
        layout.churn_anchors.push_back(
            wave > 0 ? std::vector<size_t>{static_cast<size_t>(wave - 1)} : std::vector<size_t>{});
      }
      layout.churn_rows = wave_rows;
      layout.cutoff = NextHybridTime();
      return layout;
    }

    // Churn phase: each wave inserts wave_rows new ids and deletes a fraction of them after the
    // lag the mode prescribes.
    const int far_lag = std::max(1, waves / 2);
    std::vector<std::vector<int64_t>> pending_far_deletes(waves);
    std::vector<size_t> wave_insert_file(waves, 0);  // churn index holding a wave's inserts
    for (int wave = 0; wave < waves; ++wave) {
      std::vector<int64_t> wave_ids;
      wave_ids.reserve(wave_rows);
      for (int64_t i = 0; i < wave_rows; ++i) {
        RETURN_NOT_OK(InsertRow(next_id));
        wave_ids.push_back(next_id);
        ++next_id;
      }
      layout.churn_rows += wave_rows;
      const auto num_deleted = static_cast<int64_t>(wave_rows * delete_fraction);
      std::vector<int64_t> to_delete(wave_ids.begin(), wave_ids.begin() + num_deleted);
      for (int64_t i = num_deleted; i < wave_rows && layout.live_ids.size() < sample; ++i) {
        layout.live_ids.push_back(wave_ids[i]);
      }
      for (int64_t id : to_delete) {
        if (layout.deleted_ids.size() < sample) {
          layout.deleted_ids.push_back(id);
        }
      }
      layout.deleted_rows += num_deleted;

      auto flush_churn = [&](int64_t dead, int64_t total, std::vector<size_t> anchors) -> Status {
        layout.churn_files.push_back(VERIFY_RESULT(FlushToNewFile()));
        layout.churn_dead_fraction.push_back(
            total > 0 ? static_cast<double>(dead) / static_cast<double>(total) : 0.0);
        layout.churn_anchors.push_back(std::move(anchors));
        return Status::OK();
      };
      switch (lag_mode) {
        case LagMode::kSameFile:
          for (int64_t id : to_delete) {
            RETURN_NOT_OK(DeleteRow(id));
          }
          // Each deleted row leaves a tombstone and a shadowed value in this file.
          wave_insert_file[wave] = layout.churn_files.size();
          RETURN_NOT_OK(flush_churn(2 * num_deleted, wave_rows + num_deleted, {}));
          break;
        case LagMode::kNextFile:
          wave_insert_file[wave] = layout.churn_files.size();
          RETURN_NOT_OK(flush_churn(0, wave_rows, {}));
          if (num_deleted > 0) {
            for (int64_t id : to_delete) {
              RETURN_NOT_OK(DeleteRow(id));
            }
            RETURN_NOT_OK(flush_churn(num_deleted, num_deleted, {wave_insert_file[wave]}));
          }
          break;
        case LagMode::kFar: {
          // This wave's file carries the deletes of the wave `far_lag` waves back.
          int64_t deletes_here = 0;
          std::vector<size_t> anchors;
          if (wave >= far_lag) {
            for (int64_t id : pending_far_deletes[wave - far_lag]) {
              RETURN_NOT_OK(DeleteRow(id));
              ++deletes_here;
            }
            pending_far_deletes[wave - far_lag].clear();
            if (deletes_here > 0) {
              anchors.push_back(wave_insert_file[wave - far_lag]);
            }
          }
          pending_far_deletes[wave] = std::move(to_delete);
          wave_insert_file[wave] = layout.churn_files.size();
          RETURN_NOT_OK(flush_churn(deletes_here, wave_rows + deletes_here, std::move(anchors)));
          break;
        }
      }
    }
    if (lag_mode == LagMode::kFar) {
      int64_t remaining = 0;
      std::vector<size_t> anchors;
      for (int wave = 0; wave < waves; ++wave) {
        if (pending_far_deletes[wave].empty()) {
          continue;
        }
        for (int64_t id : pending_far_deletes[wave]) {
          RETURN_NOT_OK(DeleteRow(id));
          ++remaining;
        }
        anchors.push_back(wave_insert_file[wave]);
      }
      if (remaining > 0) {
        auto flush_status = [&]() -> Status {
          layout.churn_files.push_back(VERIFY_RESULT(FlushToNewFile()));
          layout.churn_dead_fraction.push_back(1.0);
          layout.churn_anchors.push_back(std::move(anchors));
          return Status::OK();
        }();
        RETURN_NOT_OK(flush_status);
      }
    }

    layout.cutoff = NextHybridTime();
    return layout;
  }

  // Fresh database for the next arm. Options (compaction style, context factory) persist.
  Status ResetDb() {
    RETURN_NOT_OK(DestroyRocksDB());
    RETURN_NOT_OK(InitRocksDBDir());
    RETURN_NOT_OK(OpenRocksDB());
    next_ht_micros_ = 1000;
    cold_rows_ = 0;
    InitOpId();
    return Status::OK();
  }

  // ---- measurement ----------------------------------------------------------------------------

  Tickers ReadTickers() {
    auto* stats = regular_db_options().statistics.get();
    return Tickers {
      .compact_read_bytes = stats->getTickerCount(rocksdb::COMPACT_READ_BYTES),
      .compact_write_bytes = stats->getTickerCount(rocksdb::COMPACT_WRITE_BYTES),
      .keys_dropped_user = stats->getTickerCount(rocksdb::COMPACTION_KEY_DROP_USER),
      .keys_dropped_newer = stats->getTickerCount(rocksdb::COMPACTION_KEY_DROP_NEWER_ENTRY),
      .keys_dropped_obsolete = stats->getTickerCount(rocksdb::COMPACTION_KEY_DROP_OBSOLETE),
      .db_next = stats->getTickerCount(rocksdb::NUMBER_DB_NEXT),
      .db_seek = stats->getTickerCount(rocksdb::NUMBER_DB_SEEK),
      .bloom_checked = stats->getTickerCount(rocksdb::BLOOM_FILTER_CHECKED),
      .bloom_useful = stats->getTickerCount(rocksdb::BLOOM_FILTER_USEFUL),
      .block_cache_miss = stats->getTickerCount(rocksdb::BLOCK_CACHE_MISS),
      .block_cache_filter_miss = stats->getTickerCount(rocksdb::BLOCK_CACHE_FILTER_MISS),
    };
  }

  Result<FileState> ReadFileState() {
    FileState state;
    for (const auto& file : regular_db_->GetLiveFilesMetaData()) {
      ++state.num_files;
      state.total_bytes += file.total_size;
    }
    rocksdb::TablePropertiesCollection props;
    RETURN_NOT_OK(regular_db_->GetPropertiesOfAllTables(&props));
    for (const auto& [_, table_props] : props) {
      state.num_entries += table_props->num_entries;
    }
    return state;
  }

  // File names (as CompactFiles wants them) for the given file numbers.
  std::vector<std::string> FileNames(const std::vector<uint64_t>& numbers) {
    rocksdb::ColumnFamilyMetaData cf_meta;
    regular_db_->GetColumnFamilyMetaData(&cf_meta);
    std::vector<std::string> names;
    for (const auto& file : cf_meta.levels[0].files) {
      if (std::find(numbers.begin(), numbers.end(), file.name_id) != numbers.end()) {
        names.push_back(file.Name());
      }
    }
    return names;
  }

  // Compacts the given files on the calling thread. Returns false when nothing was compacted: in
  // the automatic-compaction mode a selected file may have been merged away by the picker between
  // selection and here, or be an input of a compaction in flight; both are skipped.
  Result<bool> CompactFileNumbers(const std::vector<uint64_t>& numbers) {
    for (int attempt = 0; ; ++attempt) {
      RETURN_NOT_OK(SettleAutoCompactions());
      auto names = FileNames(numbers);
      if (names.size() != numbers.size()) {
        SCHECK(FLAGS_reclamation_bench_auto_compactions, IllegalState,
               "Some files to compact are gone");
        return false;
      }
      auto status = regular_db_->CompactFiles(
          rocksdb::CompactionOptions(), names, /* output_level= */ 0);
      if (status.ok()) {
        return true;
      }
      if (!FLAGS_reclamation_bench_auto_compactions || attempt >= 3) {
        return status;
      }
    }
  }

  // Follows the compactions that consumed a file: the file that now holds its surviving entries,
  // or 0 when it compacted to nothing.
  uint64_t CurrentFileOf(uint64_t number, const std::map<uint64_t, uint64_t>& moved) const {
    while (true) {
      auto it = moved.find(number);
      if (it == moved.end()) {
        return number;
      }
      number = it->second;
      if (number == 0) {
        return 0;
      }
    }
  }

  // Compacts a group and records where every consumed file went (single output under universal
  // compaction, or nowhere). Level-0 CompactFiles pulls in every file that sits between the
  // selected ones in the level's order (rocksdb::CompactionPicker's input sanitizer), so the
  // consumed set can be larger than the selection: an anchored group is really the whole run of
  // files from the anchor to the newest selected file.
  Result<bool> CompactGroupTracked(
      const std::vector<uint64_t>& numbers, std::map<uint64_t, uint64_t>* moved) {
    const auto before = LiveFileNumbers();
    const bool compacted = VERIFY_RESULT(CompactFileNumbers(numbers));
    if (!compacted) {
      return false;
    }
    const auto after = LiveFileNumbers();
    std::vector<uint64_t> added;
    std::set_difference(
        after.begin(), after.end(), before.begin(), before.end(), std::back_inserter(added));
    std::vector<uint64_t> consumed;
    std::set_difference(
        before.begin(), before.end(), after.begin(), after.end(), std::back_inserter(consumed));
    const uint64_t output = added.empty() ? 0 : added.back();
    for (uint64_t number : consumed) {
      (*moved)[number] = output;
    }
    return true;
  }

  // The consumer read: for every queue bucket, seek to the bucket and walk forward until the first
  // live row, or the end of the bucket when every row in it is dead. Returns the entries visited
  // over all buckets. A row is dead when the first (newest) entry under its doc key is a
  // tombstone; the row-level entry sorts before any column entry of the same row.
  Result<ScanResult> ScanQueueBuckets() {
    ScanResult result;
    rocksdb::ReadOptions read_opts;
    read_opts.query_id = rocksdb::kDefaultQueryId;
    std::unique_ptr<rocksdb::Iterator> iter(regular_db_->NewIterator(read_opts));
    if (key_shape_ == KeyShape::kUnique) {
      // No bucket order to exploit: the consumer read is a full walk, so the cost is every entry.
      for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
        ++result.entries_visited;
      }
      RETURN_NOT_OK(iter->status());
      return result;
    }
    for (int32_t bucket = 0; bucket < kQueueBuckets; ++bucket) {
      const auto prefix = VERIFY_RESULT(BucketPrefix(bucket));
      std::string current_doc_key;
      for (iter->Seek(prefix); iter->Valid() && iter->key().starts_with(prefix); iter->Next()) {
        ++result.entries_visited;
        const auto key = iter->key();
        const auto doc_key_size = VERIFY_RESULT(
            dockv::DocKey::EncodedSize(key, dockv::DocKeyPart::kWholeDocKey));
        const Slice doc_key(key.data(), doc_key_size);
        if (doc_key == Slice(current_doc_key)) {
          continue;  // an older entry of a row already classified as dead
        }
        current_doc_key.assign(doc_key.cdata(), doc_key.size());
        if (!VERIFY_RESULT(dockv::Value::IsTombstoned(iter->value()))) {
          ++result.buckets_with_live_row;
          break;
        }
      }
      RETURN_NOT_OK(iter->status());
    }
    return result;
  }

  // A point read the way the DocDB read path issues one: with the bloom-aware iterator filter,
  // so files whose filter excludes the key are skipped and the bloom tickers move.
  Status PointRead(int64_t id) {
    static const rocksdb::BloomFilterAwareFileFilter bloom_filter_aware_file_filter;
    const auto key = EncodedKey(id);
    rocksdb::ReadOptions read_opts;
    read_opts.query_id = rocksdb::kDefaultQueryId;
    read_opts.iterator_filter = &bloom_filter_aware_file_filter;
    read_opts.user_key_for_filter = key.AsSlice();
    std::unique_ptr<rocksdb::Iterator> iter(regular_db_->NewIterator(read_opts));
    iter->Seek(key.AsSlice());
    return iter->status();
  }

  // Reads every sampled id back: a deleted id must resolve to a tombstone or to nothing, a live
  // id to a value. A wrongly dropped tombstone shows up as a deleted id reading back as a value,
  // which the byte and entry counts alone cannot reveal.
  Status VerifyIds(const Layout& layout) {
    rocksdb::ReadOptions read_opts;
    read_opts.query_id = rocksdb::kDefaultQueryId;
    std::unique_ptr<rocksdb::Iterator> iter(regular_db_->NewIterator(read_opts));
    auto newest_is_tombstone = [&](int64_t id) -> Result<std::optional<bool>> {
      const auto key = EncodedKey(id);
      iter->Seek(key.AsSlice());
      // The debug iterator insists that status() is read after Valid() says false.
      const bool valid = iter->Valid();
      RETURN_NOT_OK(iter->status());
      if (!valid || !iter->key().starts_with(key.AsSlice())) {
        return std::nullopt;
      }
      return VERIFY_RESULT(dockv::Value::IsTombstoned(iter->value()));
    };
    for (int64_t id : layout.deleted_ids) {
      const auto tombstoned = VERIFY_RESULT(newest_is_tombstone(id));
      SCHECK(!tombstoned.has_value() || *tombstoned, IllegalState,
             Format("Deleted row $0 reads back as a value", id));
    }
    for (int64_t id : layout.live_ids) {
      const auto tombstoned = VERIFY_RESULT(newest_is_tombstone(id));
      SCHECK(tombstoned.has_value() && !*tombstoned, IllegalState,
             Format("Live row $0 is missing or tombstoned", id));
    }
    return Status::OK();
  }

  // Oracle selection: contiguous groups of up to group_files churn files in creation order, each
  // group containing at least one candidate. With `anchored`, the files holding the rows a group's
  // tombstones delete join the group (followed through earlier compactions).
  Result<size_t> RunOracleGroups(const Layout& layout, bool anchored) {
    const size_t group = std::max(1, FLAGS_reclamation_bench_group_files);
    std::map<uint64_t, uint64_t> moved;
    size_t compactions = 0;
    for (size_t i = 0; i < layout.churn_files.size(); i += group) {
      const size_t end = std::min(i + group, layout.churn_files.size());
      bool has_candidate = false;
      for (size_t j = i; j < end; ++j) {
        has_candidate = has_candidate ||
            layout.churn_dead_fraction[j] >= FLAGS_reclamation_bench_candidate_ratio;
      }
      if (!has_candidate) {
        continue;  // the trigger would not fire on any file of this group
      }
      std::vector<uint64_t> numbers;
      for (size_t j = i; j < end; ++j) {
        const auto current = CurrentFileOf(layout.churn_files[j], moved);
        if (current != 0 && std::find(numbers.begin(), numbers.end(), current) == numbers.end()) {
          numbers.push_back(current);
        }
      }
      if (anchored) {
        for (size_t j = i; j < end; ++j) {
          for (size_t anchor : layout.churn_anchors[j]) {
            const auto current = CurrentFileOf(layout.churn_files[anchor], moved);
            if (current != 0 &&
                std::find(numbers.begin(), numbers.end(), current) == numbers.end()) {
              numbers.push_back(current);
            }
          }
        }
      }
      if (numbers.empty()) {
        continue;
      }
      if (VERIFY_RESULT(CompactGroupTracked(numbers, &moved))) {
        ++compactions;
      }
    }
    return compactions;
  }

  // One compaction over every candidate group and its anchors; the level-0 sanitizer fills the
  // run in between.
  Result<size_t> RunOracleRange(const Layout& layout) {
    const size_t group = std::max(1, FLAGS_reclamation_bench_group_files);
    std::vector<uint64_t> numbers;
    for (size_t i = 0; i < layout.churn_files.size(); i += group) {
      const size_t end = std::min(i + group, layout.churn_files.size());
      bool has_candidate = false;
      for (size_t j = i; j < end; ++j) {
        has_candidate = has_candidate ||
            layout.churn_dead_fraction[j] >= FLAGS_reclamation_bench_candidate_ratio;
      }
      if (!has_candidate) {
        continue;
      }
      for (size_t j = i; j < end; ++j) {
        numbers.push_back(layout.churn_files[j]);
        for (size_t anchor : layout.churn_anchors[j]) {
          numbers.push_back(layout.churn_files[anchor]);
        }
      }
    }
    std::sort(numbers.begin(), numbers.end());
    numbers.erase(std::unique(numbers.begin(), numbers.end()), numbers.end());
    if (numbers.empty()) {
      return 0;
    }
    return VERIFY_RESULT(CompactFileNumbers(numbers)) ? 1 : 0;
  }

  Result<size_t> RunPropertyGroups() {
    size_t compactions = 0;
    for (const auto& numbers : VERIFY_RESULT(SelectGroupsByProperties())) {
      if (VERIFY_RESULT(CompactFileNumbers(numbers))) {
        ++compactions;
      }
    }
    return compactions;
  }

  Result<ArmResult> RunArm(Arm arm) {
    RETURN_NOT_OK(ResetDb());
    const auto auto_at_start = auto_compaction_counter_->snapshot();
    const auto layout = VERIFY_RESULT(BuildLayout());
    RETURN_NOT_OK(SettleAutoCompactions());
    ArmResult result;
    result.arm = arm;
    result.before = VERIFY_RESULT(ReadFileState());
    const auto auto_after_build = auto_compaction_counter_->snapshot();
    result.auto_during_build = auto_after_build - auto_at_start;

    ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_tombstone_drop) =
        arm == Arm::kReclaimDrop || arm == Arm::kReclaimAnchored || arm == Arm::kReclaimRange;
    SetHistoryCutoffHybridTime(layout.cutoff);
    const auto probe_before = ReadProbeCounters();
    const auto tickers_before = ReadTickers();
    const auto cpu_before = ThreadCpuMicros();
    const auto wall_before = MonoTime::Now();
    switch (arm) {
      case Arm::kNone:
        break;
      case Arm::kFull:
        if (VERIFY_RESULT(CompactFileNumbers(LiveFileNumbers()))) {
          result.compactions = 1;
        }
        break;
      case Arm::kReclaim:
      case Arm::kReclaimDrop:
      case Arm::kReclaimAnchored:
      case Arm::kReclaimRange: {
        if (selection_ == Selection::kProperties) {
          result.compactions = VERIFY_RESULT(RunPropertyGroups());
        } else if (arm == Arm::kReclaimRange) {
          result.compactions = VERIFY_RESULT(RunOracleRange(layout));
        } else {
          result.compactions =
              VERIFY_RESULT(RunOracleGroups(layout, arm == Arm::kReclaimAnchored));
        }
        // Reclamation must never touch a file it did not select. With the picker running, cold
        // files may legitimately have been merged among themselves, so only count them then.
        const auto live = LiveFileNumbers();
        for (uint64_t cold : layout.cold_files) {
          const bool present = std::binary_search(live.begin(), live.end(), cold);
          result.cold_files_left += present ? 1 : 0;
          SCHECK(present || FLAGS_reclamation_bench_auto_compactions, IllegalState,
                 Format("Cold file $0 was rewritten by a reclamation compaction", cold));
        }
        break;
      }
    }
    RETURN_NOT_OK(SettleAutoCompactions());
    result.wall_micros = (MonoTime::Now() - wall_before).ToMicroseconds();
    result.cpu_micros = ThreadCpuMicros() - cpu_before;
    result.compaction = ReadTickers() - tickers_before;
    result.probe = ReadProbeCounters() - probe_before;
    result.auto_during_policy = auto_compaction_counter_->snapshot() - auto_after_build;
    SetHistoryCutoffHybridTime(HybridTime::kMin);
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_tombstone_drop) = false;
    RETURN_NOT_OK(VerifyIds(layout));
    result.after = VERIFY_RESULT(ReadFileState());

    {
      const auto before = ReadTickers();
      result.scan_result = VERIFY_RESULT(ScanQueueBuckets());
      result.scan = ReadTickers() - before;
    }
    {
      const auto before = ReadTickers();
      for (int64_t id : layout.live_ids) {
        RETURN_NOT_OK(PointRead(id));
      }
      result.point_reads_live = ReadTickers() - before;
    }
    {
      const auto before = ReadTickers();
      for (int64_t id : layout.deleted_ids) {
        RETURN_NOT_OK(PointRead(id));
      }
      result.point_reads_deleted = ReadTickers() - before;
    }

    LOG(INFO) << "arm=" << ArmName(arm) << " value_format=" << FLAGS_reclamation_bench_value_format
              << " cold_rows=" << layout.cold_rows << " churn_rows=" << layout.churn_rows
              << " deleted_rows=" << layout.deleted_rows
              << " files_before=" << result.before.num_files
              << " bytes_before=" << result.before.total_bytes
              << " entries_before=" << result.before.num_entries
              << " compactions=" << result.compactions
              << " wall_ms=" << result.wall_micros / 1000
              << " cpu_ms=" << result.cpu_micros / 1000
              << " compact_read_bytes=" << result.compaction.compact_read_bytes
              << " compact_write_bytes=" << result.compaction.compact_write_bytes
              << " probes=" << result.probe.probes
              << " tombstones_dropped=" << result.probe.tombstones_dropped
              << " kept_probe_hit=" << result.probe.kept_probe_hit
              << " kept_fail_closed=" << result.probe.kept_fail_closed
              << " probes_disabled=" << result.probe.probes_disabled
              << " auto_compactions_build=" << result.auto_during_build.compactions
              << " auto_write_bytes_build=" << result.auto_during_build.output_bytes
              << " auto_compactions=" << result.auto_during_policy.compactions
              << " auto_write_bytes=" << result.auto_during_policy.output_bytes
              << " cold_files_left=" << result.cold_files_left
              << " files_after=" << result.after.num_files
              << " bytes_after=" << result.after.total_bytes
              << " entries_after=" << result.after.num_entries
              << " scan_entries_visited=" << result.scan_result.entries_visited
              << " scan_buckets_with_live=" << result.scan_result.buckets_with_live_row
              << " scan_db_next=" << result.scan.db_next
              << " point_live_seeks=" << result.point_reads_live.db_seek
              << " point_live_bloom_useful=" << result.point_reads_live.bloom_useful
              << " point_deleted_seeks=" << result.point_reads_deleted.db_seek
              << " point_deleted_bloom_useful=" << result.point_reads_deleted.bloom_useful;
    return result;
  }

  // ---- steady state ---------------------------------------------------------------------------

  // Compaction policy applied after every wave in the steady-state case.
  enum class Policy { kNone, kFullPeriodic, kReclaim, kReclaimDrop };

  static const char* PolicyName(Policy policy) {
    switch (policy) {
      case Policy::kNone: return "NONE";
      case Policy::kFullPeriodic: return "FULL_PERIODIC";
      case Policy::kReclaim: return "RECLAIM";
      case Policy::kReclaimDrop: return "RECLAIM_DROP";
    }
    return "?";
  }

  struct WaveSample {
    int wave = 0;
    FileState state;
    uint64_t cumulative_compact_write_bytes = 0;
    size_t cumulative_compactions = 0;
    int64_t cumulative_wall_micros = 0;
    // The picker's own compactions since the churn started (automatic-compaction mode only).
    AutoCompactionCounter::Snapshot cumulative_auto;
  };

  struct SteadyResult {
    Policy policy = Policy::kNone;
    std::vector<WaveSample> samples;
    ScanResult final_scan;
  };

  // Cold rows once, then `waves` churn waves (insert, delete a fraction in the same file, flush).
  // After each wave the history cutoff moves past everything written so far, as if the files had
  // aged past retention, and the policy runs: FULL_PERIODIC compacts every live file every
  // `full_every` waves; RECLAIM and RECLAIM_DROP compact the files of the last `group_files`
  // waves once that many have accumulated. One sample per wave.
  Result<SteadyResult> RunSteadyState(Policy policy) {
    RETURN_NOT_OK(ResetDb());
    SteadyResult result;
    result.policy = policy;

    const int64_t cold_rows = FLAGS_reclamation_bench_cold_rows >= 0
        ? FLAGS_reclamation_bench_cold_rows : kDefaultColdRows;
    const int64_t wave_rows = FLAGS_reclamation_bench_wave_rows >= 0
        ? FLAGS_reclamation_bench_wave_rows : kDefaultWaveRows;
    const int cold_files = std::max(1, FLAGS_reclamation_bench_cold_files);
    const int waves = FLAGS_reclamation_bench_steady_waves >= 0
        ? FLAGS_reclamation_bench_steady_waves : std::max(1, FLAGS_reclamation_bench_waves);
    const int full_every = std::max(1, FLAGS_reclamation_bench_full_every);
    const size_t group = std::max(1, FLAGS_reclamation_bench_group_files);
    const double delete_fraction = std::clamp(FLAGS_reclamation_bench_delete_fraction, 0.0, 1.0);

    cold_rows_ = cold_rows;
    int64_t next_id = 0;
    for (int file = 0; file < cold_files && cold_rows > 0; ++file) {
      const int64_t rows = cold_rows / cold_files + (file < cold_rows % cold_files ? 1 : 0);
      if (rows == 0) {
        continue;
      }
      for (int64_t i = 0; i < rows; ++i) {
        RETURN_NOT_OK(InsertRow(next_id++));
      }
      RETURN_NOT_OK(FlushToNewFile());
    }

    RETURN_NOT_OK(SettleAutoCompactions());
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_tombstone_drop) =
        policy == Policy::kReclaimDrop;
    const auto tickers_start = ReadTickers();
    const auto auto_start = auto_compaction_counter_->snapshot();
    std::vector<uint64_t> pending_files;
    for (int wave = 0; wave < waves; ++wave) {
      const int64_t first_id = next_id;
      for (int64_t i = 0; i < wave_rows; ++i) {
        RETURN_NOT_OK(InsertRow(next_id++));
      }
      const auto num_deleted = static_cast<int64_t>(wave_rows * delete_fraction);
      for (int64_t id = first_id; id < first_id + num_deleted; ++id) {
        RETURN_NOT_OK(DeleteRow(id));
      }
      const auto flushed = VERIFY_RESULT(FlushToNewFile());
      if (flushed != 0) {
        pending_files.push_back(flushed);
      }

      SetHistoryCutoffHybridTime(NextHybridTime());
      const auto wall_before = MonoTime::Now();
      size_t compactions = 0;
      switch (policy) {
        case Policy::kNone:
          break;
        case Policy::kFullPeriodic:
          if ((wave + 1) % full_every == 0) {
            if (VERIFY_RESULT(CompactFileNumbers(LiveFileNumbers()))) {
              ++compactions;
            }
            pending_files.clear();
          }
          break;
        case Policy::kReclaim:
        case Policy::kReclaimDrop:
          if (selection_ == Selection::kProperties) {
            compactions += VERIFY_RESULT(RunPropertyGroups());
            pending_files.clear();
          } else if (pending_files.size() >= group) {
            // Same-file layout: every pending file has the same reclaimable fraction.
            const double dead_fraction = wave_rows + num_deleted > 0
                ? static_cast<double>(2 * num_deleted) /
                      static_cast<double>(wave_rows + num_deleted)
                : 0.0;
            if (dead_fraction >= FLAGS_reclamation_bench_candidate_ratio) {
              if (VERIFY_RESULT(CompactFileNumbers(pending_files))) {
                ++compactions;
              }
            }
            pending_files.clear();
          }
          break;
      }
      RETURN_NOT_OK(SettleAutoCompactions());
      const auto wall = (MonoTime::Now() - wall_before).ToMicroseconds();
      SetHistoryCutoffHybridTime(HybridTime::kMin);

      WaveSample sample;
      sample.wave = wave + 1;
      sample.state = VERIFY_RESULT(ReadFileState());
      sample.cumulative_compact_write_bytes =
          (ReadTickers() - tickers_start).compact_write_bytes;
      sample.cumulative_auto = auto_compaction_counter_->snapshot() - auto_start;
      sample.cumulative_compactions =
          (result.samples.empty() ? 0 : result.samples.back().cumulative_compactions) +
          compactions;
      sample.cumulative_wall_micros =
          (result.samples.empty() ? 0 : result.samples.back().cumulative_wall_micros) + wall;
      result.samples.push_back(sample);
    }
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_tombstone_drop) = false;
    result.final_scan = VERIFY_RESULT(ScanQueueBuckets());

    const auto& last = result.samples.back();
    LOG(INFO) << "steady policy=" << PolicyName(policy)
              << " value_format=" << FLAGS_reclamation_bench_value_format << " waves=" << waves
              << " cold_rows=" << cold_rows << " wave_rows=" << wave_rows
              << " cumulative_compactions=" << last.cumulative_compactions
              << " cumulative_compact_write_bytes=" << last.cumulative_compact_write_bytes
              << " cumulative_wall_ms=" << last.cumulative_wall_micros / 1000
              << " auto_compactions=" << last.cumulative_auto.compactions
              << " auto_write_bytes=" << last.cumulative_auto.output_bytes
              << " final_files=" << last.state.num_files
              << " final_bytes=" << last.state.total_bytes
              << " final_entries=" << last.state.num_entries
              << " final_scan_entries_visited=" << result.final_scan.entries_visited;
    return result;
  }

  void AppendSteadyCsv(const std::vector<SteadyResult>& results) {
    if (FLAGS_reclamation_bench_steady_csv.empty()) {
      return;
    }
    std::ifstream probe(FLAGS_reclamation_bench_steady_csv);
    const bool need_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    probe.close();
    std::ofstream out(FLAGS_reclamation_bench_steady_csv, std::ios::app);
    ASSERT_TRUE(out.good()) << "Cannot open " << FLAGS_reclamation_bench_steady_csv;
    if (need_header) {
      out << "policy,wave,cold_rows,cold_files,wave_rows,delete_fraction,group_files,full_every,"
             "files,bytes,entries,cumulative_compactions,cumulative_compact_write_bytes,"
             "cumulative_wall_micros,final_scan_entries_visited,key_shape,value_format,"
             "selection,auto_compactions,cumulative_auto_compactions,"
             "cumulative_auto_write_bytes\n";
    }
    for (const auto& r : results) {
      for (const auto& s : r.samples) {
        out << Format(
            "$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,",
            PolicyName(r.policy), s.wave,
            FLAGS_reclamation_bench_cold_rows >= 0 ? FLAGS_reclamation_bench_cold_rows
                                                   : kDefaultColdRows,
            FLAGS_reclamation_bench_cold_files,
            FLAGS_reclamation_bench_wave_rows >= 0 ? FLAGS_reclamation_bench_wave_rows
                                                   : kDefaultWaveRows,
            FLAGS_reclamation_bench_delete_fraction, FLAGS_reclamation_bench_group_files,
            FLAGS_reclamation_bench_full_every, s.state.num_files, s.state.total_bytes,
            s.state.num_entries, s.cumulative_compactions, s.cumulative_compact_write_bytes,
            s.cumulative_wall_micros, r.final_scan.entries_visited,
            FLAGS_reclamation_bench_key_shape, FLAGS_reclamation_bench_value_format)
            << Format(
                   "$0,$1,$2,$3\n",
                   FLAGS_reclamation_bench_selection, FLAGS_reclamation_bench_auto_compactions,
                   s.cumulative_auto.compactions, s.cumulative_auto.output_bytes);
      }
    }
  }

  static std::string CsvHeader() {
    return "arm,cold_rows,cold_files,wave_rows,waves,delete_fraction,lag_mode,payload_bytes,"
           "group_files,files_before,bytes_before,entries_before,compactions,wall_micros,"
           "cpu_micros,compact_read_bytes,compact_write_bytes,keys_dropped_user,"
           "keys_dropped_newer,files_after,bytes_after,entries_after,scan_entries_visited,"
           "scan_buckets_with_live,scan_db_next,scan_block_cache_miss,point_live_seeks,"
           "point_live_bloom_checked,"
           "point_live_bloom_useful,point_deleted_seeks,point_deleted_bloom_checked,"
           "point_deleted_bloom_useful,key_shape,candidate_ratio,probes,tombstones_dropped,"
           "kept_probe_hit,kept_fail_closed,value_format,selection,auto_compactions,churn_kind,"
           "updates_per_wave,auto_compactions_build,auto_write_bytes_build,"
           "auto_compactions_policy,auto_write_bytes_policy,cold_files_left,probes_disabled\n";
  }

  static std::string CsvRow(const ArmResult& r) {
    return Format(
        "$0,$1,$2,$3,$4,$5,$6,$7,$8,",
        ArmName(r.arm),
        FLAGS_reclamation_bench_cold_rows >= 0 ? FLAGS_reclamation_bench_cold_rows
                                               : kDefaultColdRows,
        FLAGS_reclamation_bench_cold_files,
        FLAGS_reclamation_bench_wave_rows >= 0 ? FLAGS_reclamation_bench_wave_rows
                                               : kDefaultWaveRows,
        FLAGS_reclamation_bench_waves, FLAGS_reclamation_bench_delete_fraction,
        FLAGS_reclamation_bench_lag_mode, FLAGS_reclamation_bench_payload_bytes,
        FLAGS_reclamation_bench_group_files) +
        Format(
            "$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,",
            r.before.num_files, r.before.total_bytes, r.before.num_entries, r.compactions,
            r.wall_micros, r.cpu_micros, r.compaction.compact_read_bytes,
            r.compaction.compact_write_bytes, r.compaction.keys_dropped_user,
            r.compaction.keys_dropped_newer) +
        Format(
            "$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,",
            r.after.num_files, r.after.total_bytes, r.after.num_entries,
            r.scan_result.entries_visited, r.scan_result.buckets_with_live_row,
            r.scan.db_next, r.scan.block_cache_miss, r.point_reads_live.db_seek,
            r.point_reads_live.bloom_checked, r.point_reads_live.bloom_useful,
            r.point_reads_deleted.db_seek, r.point_reads_deleted.bloom_checked,
            r.point_reads_deleted.bloom_useful) +
        Format(
            "$0,$1,$2,$3,$4,$5,$6,",
            FLAGS_reclamation_bench_key_shape, FLAGS_reclamation_bench_candidate_ratio,
            r.probe.probes, r.probe.tombstones_dropped, r.probe.kept_probe_hit,
            r.probe.kept_fail_closed, FLAGS_reclamation_bench_value_format) +
        Format(
            "$0,$1,$2,$3,$4,$5,$6,$7,$8,$9\n",
            FLAGS_reclamation_bench_selection, FLAGS_reclamation_bench_auto_compactions,
            FLAGS_reclamation_bench_churn_kind, FLAGS_reclamation_bench_updates_per_wave,
            r.auto_during_build.compactions, r.auto_during_build.output_bytes,
            r.auto_during_policy.compactions, r.auto_during_policy.output_bytes,
            r.cold_files_left, r.probe.probes_disabled);
  }

  void AppendCsv(const std::vector<ArmResult>& results) {
    if (FLAGS_reclamation_bench_csv.empty()) {
      return;
    }
    std::ifstream probe(FLAGS_reclamation_bench_csv);
    const bool need_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    probe.close();
    std::ofstream out(FLAGS_reclamation_bench_csv, std::ios::app);
    ASSERT_TRUE(out.good()) << "Cannot open " << FLAGS_reclamation_bench_csv;
    if (need_header) {
      out << CsvHeader();
    }
    for (const auto& r : results) {
      out << CsvRow(r);
    }
  }

  void InitPayloads() {
    payload_bytes_ = std::max(1, FLAGS_reclamation_bench_payload_bytes);
    key_shape_ = CHECK_RESULT(ParseKeyShape(FLAGS_reclamation_bench_key_shape));
    value_format_ = CHECK_RESULT(ParseValueFormat(FLAGS_reclamation_bench_value_format));
    selection_ = CHECK_RESULT(ParseSelection(FLAGS_reclamation_bench_selection));
    churn_kind_ = CHECK_RESULT(ParseChurnKind(FLAGS_reclamation_bench_churn_kind));
    CHECK(!FLAGS_reclamation_bench_auto_compactions || selection_ == Selection::kProperties)
        << "--reclamation_bench_auto_compactions needs --reclamation_bench_selection=properties";
  }

  void CheckProductionGateGovernsPartialCompaction();

  size_t payload_bytes_ = 1;
  KeyShape key_shape_ = KeyShape::kBucket;
  ValueFormat value_format_ = ValueFormat::kColumn;
  Selection selection_ = Selection::kOracle;
  ChurnKind churn_kind_ = ChurnKind::kDelete;
  int64_t next_ht_micros_ = 1000;
  int64_t cold_rows_ = 0;
  MetricRegistry metric_registry_;
  scoped_refptr<MetricEntity> metric_entity_;
  CompactionMetrics compaction_metrics_;
  std::shared_ptr<AutoCompactionCounter> auto_compaction_counter_;
};

// Proves the production gate is in effect for partial compactions, which is what makes the
// RECLAIM arm meaningful: with the fixture's stub every partial compaction would keep every
// tombstone. Layout A (pair in the older file, live row in the newer file): compacting the older
// file alone drops the pair, because nothing outside the compaction is older than the tombstone.
// Layout B (live row in the older file, pair in the newer file): compacting the newer file alone
// keeps the tombstone, because the older file could hold an earlier version of the same key.
void ReclamationCompactionPerfTest::CheckProductionGateGovernsPartialCompaction() {
  InitPayloads();

  // Layout A.
  ASSERT_OK(InsertRow(1));
  ASSERT_OK(DeleteRow(1));
  const auto pair_file = ASSERT_RESULT(FlushToNewFile());
  ASSERT_OK(InsertRow(2));
  const auto newer_live_file = ASSERT_RESULT(FlushToNewFile());
  ASSERT_NE(pair_file, newer_live_file);
  SetHistoryCutoffHybridTime(NextHybridTime());
  ASSERT_TRUE(ASSERT_RESULT(CompactFileNumbers({pair_file})));
  auto dump = DocDBDebugDumpToStr();
  ASSERT_EQ(dump.find("DEL"), std::string::npos) << dump;
  ASSERT_EQ(ASSERT_RESULT(ReadFileState()).num_entries, 1);

  // Layout B.
  ASSERT_OK(ResetDb());
  ASSERT_OK(InsertRow(2));
  const auto older_live_file = ASSERT_RESULT(FlushToNewFile());
  ASSERT_OK(InsertRow(1));
  ASSERT_OK(DeleteRow(1));
  const auto newer_pair_file = ASSERT_RESULT(FlushToNewFile());
  ASSERT_NE(older_live_file, newer_pair_file);
  SetHistoryCutoffHybridTime(NextHybridTime());
  ASSERT_TRUE(ASSERT_RESULT(CompactFileNumbers({newer_pair_file})));
  dump = DocDBDebugDumpToStr();
  ASSERT_NE(dump.find("DEL"), std::string::npos) << dump;
  // The tombstone survives, the value it shadows does not.
  ASSERT_EQ(ASSERT_RESULT(ReadFileState()).num_entries, 2);
}

TEST_F(ReclamationCompactionPerfTest, ProductionGateGovernsPartialCompaction) {
  CheckProductionGateGovernsPartialCompaction();
}

// The same two layouts with packed rows: the row-level tombstone shadows the packed row inside
// the compaction exactly as it shadows a column entry.
TEST_F(ReclamationCompactionPerfTest, ProductionGateGovernsPartialCompactionPackedRows) {
  FLAGS_reclamation_bench_value_format = "packed";
  CheckProductionGateGovernsPartialCompaction();
}

// The comparison itself: NONE, FULL, RECLAIM and RECLAIM_DROP on the flag-defined layout, plus
// RECLAIM_ANCHORED when the oracle knows the anchors.
TEST_F(ReclamationCompactionPerfTest, YB_DISABLE_TEST_IN_TSAN(QueuePattern)) {
  InitPayloads();
  std::vector<Arm> arms = {Arm::kNone, Arm::kFull, Arm::kReclaim, Arm::kReclaimDrop};
  if (selection_ == Selection::kOracle) {
    arms.push_back(Arm::kReclaimAnchored);
    arms.push_back(Arm::kReclaimRange);
  }
  std::vector<ArmResult> results;
  for (Arm arm : arms) {
    results.push_back(ASSERT_RESULT(RunArm(arm)));
  }

  const auto& none = results[0];
  const auto& full = results[1];
  const auto& reclaim = results[2];
  const auto& drop = results[3];
  LOG(INFO) << "summary: bytes_after none/full/reclaim/drop = " << none.after.total_bytes << "/"
            << full.after.total_bytes << "/" << reclaim.after.total_bytes << "/"
            << drop.after.total_bytes
            << ", entries_after = " << none.after.num_entries << "/" << full.after.num_entries
            << "/" << reclaim.after.num_entries << "/" << drop.after.num_entries
            << ", compact_write_bytes full/reclaim/drop = " << full.compaction.compact_write_bytes
            << "/" << reclaim.compaction.compact_write_bytes << "/"
            << drop.compaction.compact_write_bytes
            << ", scan_entries_visited none/full/reclaim/drop = "
            << none.scan_result.entries_visited << "/" << full.scan_result.entries_visited << "/"
            << reclaim.scan_result.entries_visited << "/" << drop.scan_result.entries_visited;
  for (size_t i = 4; i < results.size(); ++i) {
    const auto& extra = results[i];
    LOG(INFO) << "summary " << ArmName(extra.arm)
              << ": compact_write_bytes=" << extra.compaction.compact_write_bytes
              << " compact_read_bytes=" << extra.compaction.compact_read_bytes
              << " entries_after=" << extra.after.num_entries
              << " scan_entries_visited=" << extra.scan_result.entries_visited;
  }

  // Sanity, not performance: full compaction removes at least as much as either reclamation arm,
  // dropping tombstones removes at least as much as keeping them, and a reclamation arm never adds
  // entries. Only the same-file layout guarantees that reclamation reclaims something: with a lag
  // the pair may never share a compaction, and then keeping everything is the correct outcome.
  // With the picker running the arms start from layouts that already differ, so only the FULL
  // bound is checked there.
  ASSERT_LE(full.after.num_entries, drop.after.num_entries);
  if (!FLAGS_reclamation_bench_auto_compactions) {
    ASSERT_LE(drop.after.num_entries, reclaim.after.num_entries);
    ASSERT_LE(reclaim.after.num_entries, none.after.num_entries);
    if (FLAGS_reclamation_bench_delete_fraction > 0 &&
        FLAGS_reclamation_bench_lag_mode == "same_file" &&
        FLAGS_reclamation_bench_churn_kind == "delete") {
      ASSERT_LT(reclaim.after.num_entries, none.after.num_entries);
    }
  }

  AppendCsv(results);
}

// Steady state: the same churn under each policy, wave after wave. Cumulative bytes written and
// the live size over time are the outputs; the sanity checks only pin the ordering that must hold.
TEST_F(ReclamationCompactionPerfTest, YB_DISABLE_TEST_IN_TSAN(SteadyState)) {
  InitPayloads();
  std::vector<SteadyResult> results;
  for (Policy policy :
       {Policy::kNone, Policy::kFullPeriodic, Policy::kReclaim, Policy::kReclaimDrop}) {
    results.push_back(ASSERT_RESULT(RunSteadyState(policy)));
  }

  const auto& none = results[0].samples.back();
  const auto& full = results[1].samples.back();
  const auto& reclaim = results[2].samples.back();
  const auto& drop = results[3].samples.back();
  LOG(INFO) << "steady summary: cumulative_compact_write_bytes none/full/reclaim/drop = "
            << none.cumulative_compact_write_bytes << "/" << full.cumulative_compact_write_bytes
            << "/" << reclaim.cumulative_compact_write_bytes << "/"
            << drop.cumulative_compact_write_bytes
            << ", final_bytes = " << none.state.total_bytes << "/" << full.state.total_bytes
            << "/" << reclaim.state.total_bytes << "/" << drop.state.total_bytes
            << ", final_entries = " << none.state.num_entries << "/" << full.state.num_entries
            << "/" << reclaim.state.num_entries << "/" << drop.state.num_entries;

  if (!FLAGS_reclamation_bench_auto_compactions) {
    ASSERT_EQ(none.cumulative_compact_write_bytes, 0);
    ASSERT_LE(drop.state.num_entries, reclaim.state.num_entries);
    if (FLAGS_reclamation_bench_delete_fraction > 0) {
      ASSERT_LT(reclaim.state.num_entries, none.state.num_entries);
    }
  }

  AppendSteadyCsv(results);
}

}  // namespace yb::docdb

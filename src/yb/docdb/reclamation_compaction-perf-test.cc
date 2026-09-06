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
#include <fstream>
#include <iterator>
#include <random>

#include <gtest/gtest.h>

#include "yb/common/ql_value.h"

#include "yb/docdb/docdb_compaction_context.h"
#include "yb/docdb/docdb_test_base.h"
#include "yb/docdb/key_bounds.h"

#include "yb/dockv/doc_key.h"
#include "yb/dockv/doc_path.h"
#include "yb/dockv/value.h"

#include "yb/rocksdb/db.h"
#include "yb/rocksdb/statistics.h"
#include "yb/rocksdb/table_properties.h"

#include "yb/util/flags.h"
#include "yb/util/format.h"
#include "yb/util/logging.h"
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

namespace yb::docdb {

namespace {

constexpr int64_t kDefaultColdRows = RegularBuildVsDebugVsSanitizers(20000, 2000, 500);
constexpr int64_t kDefaultWaveRows = RegularBuildVsDebugVsSanitizers(5000, 500, 100);
constexpr uint64_t kIdMixer = 0x9E3779B97F4A7C15ULL;
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

// NONE: no compaction, the baseline. FULL: one CompactFiles over every live file, the incumbent.
// RECLAIM: CompactFiles over groups of contiguous churn files only, tombstones governed by the
// coarse gate. RECLAIM_DROP: the same with docdb_reclamation_tombstone_drop on, so tombstones
// whose key is proven absent from the files outside the compaction are dropped too.
enum class Arm { kNone, kFull, kReclaim, kReclaimDrop };

const char* ArmName(Arm arm) {
  switch (arm) {
    case Arm::kNone: return "NONE";
    case Arm::kFull: return "FULL";
    case Arm::kReclaim: return "RECLAIM";
    case Arm::kReclaimDrop: return "RECLAIM_DROP";
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
    // The generator decides the file layout; no automatic compaction may rearrange it.
    ASSERT_OK(DisableCompactions());
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

  Schema CreateSchema() override { return Schema(); }

  Status InitRocksDBOptions() override {
    RETURN_NOT_OK(DocDBRocksDBFixture::InitRocksDBOptions());
    UseProductionCompactionConstraints();
    return Status::OK();
  }

  // The fixture's default provider returns a constant for other_min, a test stub that forbids
  // tombstone removal in every partial compaction. Install the production computation instead, so
  // partial compactions are governed by the memtable and live-file frontiers as in a tablet.
  // Must be re-applied after any ReinitDBOptions() call, which rebuilds the factory.
  void UseProductionCompactionConstraints() {
    UseProductionCompactionHybridTimeConstraints();
  }

  // ---- workload -------------------------------------------------------------------------------

  struct Layout {
    std::vector<uint64_t> cold_files;   // file numbers, creation order
    std::vector<uint64_t> churn_files;  // file numbers, creation order
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

  // A distinct pseudo-random payload per row, so block compression sees no cross-row repeats and
  // byte counts reflect the payload size rather than a small pool of reused strings.
  std::string Payload(int64_t id) const {
    std::mt19937_64 rng(FLAGS_reclamation_bench_seed ^ (static_cast<uint64_t>(id) * kIdMixer));
    return RandomHumanReadableString(payload_bytes_, &rng);
  }

  Status InsertRow(int64_t id) {
    return SetPrimitive(
        dockv::DocPath(EncodedKey(id), dockv::KeyEntryValue::MakeColumnId(kValueColumn)),
        QLValue::Primitive(Payload(id)), NextHybridTime());
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

  // Flushes the memtable and returns the number of the file it produced.
  Result<uint64_t> FlushToNewFile() {
    auto before = LiveFileNumbers();
    RETURN_NOT_OK(FlushRocksDbAndWait(rocksdb::FlushReason::kTestOnly));
    auto after = LiveFileNumbers();
    std::vector<uint64_t> added;
    std::set_difference(
        after.begin(), after.end(), before.begin(), before.end(), std::back_inserter(added));
    SCHECK_EQ(added.size(), 1, IllegalState, Format("Flush produced $0 files", added.size()));
    return added.front();
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

    // Churn phase: each wave inserts wave_rows new ids and deletes a fraction of them after the
    // lag the mode prescribes.
    const int far_lag = std::max(1, waves / 2);
    std::vector<std::vector<int64_t>> pending_far_deletes(waves);
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

      switch (lag_mode) {
        case LagMode::kSameFile:
          for (int64_t id : to_delete) {
            RETURN_NOT_OK(DeleteRow(id));
          }
          layout.churn_files.push_back(VERIFY_RESULT(FlushToNewFile()));
          break;
        case LagMode::kNextFile:
          layout.churn_files.push_back(VERIFY_RESULT(FlushToNewFile()));
          for (int64_t id : to_delete) {
            RETURN_NOT_OK(DeleteRow(id));
          }
          layout.churn_files.push_back(VERIFY_RESULT(FlushToNewFile()));
          break;
        case LagMode::kFar:
          // This wave's file carries the deletes of the wave `far_lag` waves back.
          if (wave >= far_lag) {
            for (int64_t id : pending_far_deletes[wave - far_lag]) {
              RETURN_NOT_OK(DeleteRow(id));
            }
            pending_far_deletes[wave - far_lag].clear();
          }
          pending_far_deletes[wave] = std::move(to_delete);
          layout.churn_files.push_back(VERIFY_RESULT(FlushToNewFile()));
          break;
      }
    }
    if (lag_mode == LagMode::kFar) {
      bool any = false;
      for (auto& ids : pending_far_deletes) {
        for (int64_t id : ids) {
          RETURN_NOT_OK(DeleteRow(id));
          any = true;
        }
      }
      if (any) {
        layout.churn_files.push_back(VERIFY_RESULT(FlushToNewFile()));
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

  Status CompactFileNumbers(const std::vector<uint64_t>& numbers) {
    auto names = FileNames(numbers);
    SCHECK_EQ(names.size(), numbers.size(), IllegalState, "Some files to compact are gone");
    return regular_db_->CompactFiles(rocksdb::CompactionOptions(), names, /* output_level= */ 0);
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

  Status PointRead(int64_t id) {
    rocksdb::ReadOptions read_opts;
    read_opts.query_id = rocksdb::kDefaultQueryId;
    std::unique_ptr<rocksdb::Iterator> iter(regular_db_->NewIterator(read_opts));
    iter->Seek(EncodedKey(id).AsSlice());
    return iter->status();
  }

  Result<ArmResult> RunArm(Arm arm) {
    RETURN_NOT_OK(ResetDb());
    const auto layout = VERIFY_RESULT(BuildLayout());
    ArmResult result;
    result.arm = arm;
    result.before = VERIFY_RESULT(ReadFileState());

    ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_tombstone_drop) =
        arm == Arm::kReclaimDrop;
    SetHistoryCutoffHybridTime(layout.cutoff);
    const auto tickers_before = ReadTickers();
    const auto cpu_before = ThreadCpuMicros();
    const auto wall_before = MonoTime::Now();
    switch (arm) {
      case Arm::kNone:
        break;
      case Arm::kFull:
        RETURN_NOT_OK(CompactFileNumbers(LiveFileNumbers()));
        result.compactions = 1;
        break;
      case Arm::kReclaim:
      case Arm::kReclaimDrop: {
        // Oracle selection: the generator knows which files carry the churn. Contiguous groups of
        // up to group_files, in creation order.
        const size_t group = std::max(1, FLAGS_reclamation_bench_group_files);
        for (size_t i = 0; i < layout.churn_files.size(); i += group) {
          std::vector<uint64_t> numbers(
              layout.churn_files.begin() + i,
              layout.churn_files.begin() + std::min(i + group, layout.churn_files.size()));
          RETURN_NOT_OK(CompactFileNumbers(numbers));
          ++result.compactions;
        }
        // Reclamation must never touch a file it did not select.
        const auto live = LiveFileNumbers();
        for (uint64_t cold : layout.cold_files) {
          SCHECK(std::binary_search(live.begin(), live.end(), cold), IllegalState,
                 Format("Cold file $0 was rewritten by a reclamation compaction", cold));
        }
        break;
      }
    }
    result.wall_micros = (MonoTime::Now() - wall_before).ToMicroseconds();
    result.cpu_micros = ThreadCpuMicros() - cpu_before;
    result.compaction = ReadTickers() - tickers_before;
    SetHistoryCutoffHybridTime(HybridTime::kMin);
    ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_tombstone_drop) = false;
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

    LOG(INFO) << "arm=" << ArmName(arm)
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
              << " keys_dropped_user=" << result.compaction.keys_dropped_user
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

    ANNOTATE_UNPROTECTED_WRITE(FLAGS_docdb_reclamation_tombstone_drop) =
        policy == Policy::kReclaimDrop;
    const auto tickers_start = ReadTickers();
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
      pending_files.push_back(VERIFY_RESULT(FlushToNewFile()));

      SetHistoryCutoffHybridTime(NextHybridTime());
      const auto wall_before = MonoTime::Now();
      size_t compactions = 0;
      switch (policy) {
        case Policy::kNone:
          break;
        case Policy::kFullPeriodic:
          if ((wave + 1) % full_every == 0) {
            RETURN_NOT_OK(CompactFileNumbers(LiveFileNumbers()));
            ++compactions;
            pending_files.clear();
          }
          break;
        case Policy::kReclaim:
        case Policy::kReclaimDrop:
          if (pending_files.size() >= group) {
            RETURN_NOT_OK(CompactFileNumbers(pending_files));
            ++compactions;
            pending_files.clear();
          }
          break;
      }
      const auto wall = (MonoTime::Now() - wall_before).ToMicroseconds();
      SetHistoryCutoffHybridTime(HybridTime::kMin);

      WaveSample sample;
      sample.wave = wave + 1;
      sample.state = VERIFY_RESULT(ReadFileState());
      sample.cumulative_compact_write_bytes =
          (ReadTickers() - tickers_start).compact_write_bytes;
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
    LOG(INFO) << "steady policy=" << PolicyName(policy) << " waves=" << waves
              << " cold_rows=" << cold_rows << " wave_rows=" << wave_rows
              << " cumulative_compactions=" << last.cumulative_compactions
              << " cumulative_compact_write_bytes=" << last.cumulative_compact_write_bytes
              << " cumulative_wall_ms=" << last.cumulative_wall_micros / 1000
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
             "cumulative_wall_micros,final_scan_entries_visited\n";
    }
    for (const auto& r : results) {
      for (const auto& s : r.samples) {
        out << Format(
            "$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14\n",
            PolicyName(r.policy), s.wave,
            FLAGS_reclamation_bench_cold_rows >= 0 ? FLAGS_reclamation_bench_cold_rows
                                                   : kDefaultColdRows,
            FLAGS_reclamation_bench_cold_files,
            FLAGS_reclamation_bench_wave_rows >= 0 ? FLAGS_reclamation_bench_wave_rows
                                                   : kDefaultWaveRows,
            FLAGS_reclamation_bench_delete_fraction, FLAGS_reclamation_bench_group_files,
            FLAGS_reclamation_bench_full_every, s.state.num_files, s.state.total_bytes,
            s.state.num_entries, s.cumulative_compactions, s.cumulative_compact_write_bytes,
            s.cumulative_wall_micros, r.final_scan.entries_visited);
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
           "point_deleted_bloom_useful\n";
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
            "$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12\n",
            r.after.num_files, r.after.total_bytes, r.after.num_entries,
            r.scan_result.entries_visited, r.scan_result.buckets_with_live_row,
            r.scan.db_next, r.scan.block_cache_miss, r.point_reads_live.db_seek,
            r.point_reads_live.bloom_checked, r.point_reads_live.bloom_useful,
            r.point_reads_deleted.db_seek, r.point_reads_deleted.bloom_checked,
            r.point_reads_deleted.bloom_useful);
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
  }

  size_t payload_bytes_ = 1;
  int64_t next_ht_micros_ = 1000;
  int64_t cold_rows_ = 0;
};

// Proves the production gate is in effect for partial compactions, which is what makes the
// RECLAIM arm meaningful: with the fixture's stub every partial compaction would keep every
// tombstone. Layout A (pair in the older file, live row in the newer file): compacting the older
// file alone drops the pair, because nothing outside the compaction is older than the tombstone.
// Layout B (live row in the older file, pair in the newer file): compacting the newer file alone
// keeps the tombstone, because the older file could hold an earlier version of the same key.
TEST_F(ReclamationCompactionPerfTest, ProductionGateGovernsPartialCompaction) {
  InitPayloads();

  // Layout A.
  ASSERT_OK(InsertRow(1));
  ASSERT_OK(DeleteRow(1));
  const auto pair_file = ASSERT_RESULT(FlushToNewFile());
  ASSERT_OK(InsertRow(2));
  const auto newer_live_file = ASSERT_RESULT(FlushToNewFile());
  ASSERT_NE(pair_file, newer_live_file);
  SetHistoryCutoffHybridTime(NextHybridTime());
  ASSERT_OK(CompactFileNumbers({pair_file}));
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
  ASSERT_OK(CompactFileNumbers({newer_pair_file}));
  dump = DocDBDebugDumpToStr();
  ASSERT_NE(dump.find("DEL"), std::string::npos) << dump;
  // The tombstone survives, the value it shadows does not.
  ASSERT_EQ(ASSERT_RESULT(ReadFileState()).num_entries, 2);
}

// The comparison itself: NONE, FULL, RECLAIM and RECLAIM_DROP on the flag-defined layout.
TEST_F(ReclamationCompactionPerfTest, YB_DISABLE_TEST_IN_TSAN(QueuePattern)) {
  InitPayloads();
  std::vector<ArmResult> results;
  for (Arm arm : {Arm::kNone, Arm::kFull, Arm::kReclaim, Arm::kReclaimDrop}) {
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

  // Sanity, not performance: full compaction removes at least as much as either reclamation arm,
  // dropping tombstones removes at least as much as keeping them, and every compacting arm
  // reduces the baseline entry count whenever something was deleted.
  ASSERT_LE(full.after.num_entries, drop.after.num_entries);
  ASSERT_LE(drop.after.num_entries, reclaim.after.num_entries);
  if (FLAGS_reclamation_bench_delete_fraction > 0) {
    ASSERT_LT(reclaim.after.num_entries, none.after.num_entries);
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

  ASSERT_EQ(none.cumulative_compact_write_bytes, 0);
  ASSERT_LE(drop.state.num_entries, reclaim.state.num_entries);
  if (FLAGS_reclamation_bench_delete_fraction > 0) {
    ASSERT_LT(reclaim.state.num_entries, none.state.num_entries);
  }

  AppendSteadyCsv(results);
}

}  // namespace yb::docdb

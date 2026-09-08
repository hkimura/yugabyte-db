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

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <boost/functional/hash.hpp>

#include "yb/common/column_id.h"
#include "yb/common/common_types.pb.h"
#include "yb/common/hybrid_time.h"

#include "yb/docdb/docdb_fwd.h"
#include "yb/dockv/expiration.h"

#include "yb/dockv/value_type.h"
#include "yb/gutil/thread_annotations.h"

#include "yb/rocksdb/compaction_filter.h"
#include "yb/rocksdb/db/compaction_context.h"
#include "yb/rocksdb/metadata.h"

#include "yb/server/hybrid_clock.h"

#include "yb/util/metrics_fwd.h"
#include "yb/util/strongly_typed_bool.h"

namespace yb::docdb {

YB_STRONGLY_TYPED_BOOL(IsMajorCompaction);
YB_STRONGLY_TYPED_BOOL(ShouldRetainDeleteMarkersInMajorCompaction);

using ColumnIds = std::unordered_set<ColumnId, boost::hash<ColumnId>>;

std::optional<dockv::PackedRowVersion> PackedRowVersion(TableType table_type, bool is_colocated);

// A more detailed history cutoff for allowing a different policy for cotables
// on the master. In case of master both the below fields are set.
// cotables_cutoff_ht is used for cotables (aka the ysql system tables) and
// primary_cutoff_ht is used for the sys catalog table (aka the docdb metadata table).
// On tservers, only the primary_cutoff_ht is valid and used for all
// tables both colocated and non-colocated. The cotables_cutoff_ht is invalid.
struct HistoryCutoff {
  // Set only on the master and applies to cotables.
  HybridTime cotables_cutoff_ht;

  // Used everywhere else i.e. for the sys catalog table on the master,
  // colocated tables on tservers and non-colocated tables on tservers.
  HybridTime primary_cutoff_ht;

  void MakeAtLeast(const HistoryCutoff& rhs) {
    cotables_cutoff_ht.MakeAtLeast(rhs.cotables_cutoff_ht);
    primary_cutoff_ht.MakeAtLeast(rhs.primary_cutoff_ht);
  }

  void MakeAtMost(const HistoryCutoff& rhs) {
    cotables_cutoff_ht.MakeAtMost(rhs.cotables_cutoff_ht);
    primary_cutoff_ht.MakeAtMost(rhs.primary_cutoff_ht);
  }

  std::string ToString() const {
    return YB_STRUCT_TO_STRING(cotables_cutoff_ht, primary_cutoff_ht);
  }
};

bool operator==(HistoryCutoff a, HistoryCutoff b);

HistoryCutoff ConstructMinCutoff(HistoryCutoff a, HistoryCutoff b);

std::ostream& operator<<(std::ostream& out, HistoryCutoff cutoff);

struct EncodedHistoryCutoff {
  explicit EncodedHistoryCutoff(HistoryCutoff value)
      : cotables_cutoff_encoded(value.cotables_cutoff_ht, kMaxWriteId),
        primary_cutoff_encoded(value.primary_cutoff_ht, kMaxWriteId) {}

  std::string ToString() {
    return YB_STRUCT_TO_STRING(primary_cutoff_encoded, cotables_cutoff_encoded);
  }

  EncodedDocHybridTime cotables_cutoff_encoded;
  EncodedDocHybridTime primary_cutoff_encoded;
};

// A "directive" of how a particular compaction should retain old (overwritten or deleted) values.
struct HistoryRetentionDirective {
  // We will not keep history below this hybrid_time. The view of the database at this hybrid_time
  // is preserved, but after the compaction completes, we should not expect to be able to do
  // consistent scans at DocDB hybrid times lower than this. Those scans will result in missing
  // data. Therefore, it is really important to always set this to a value lower than or equal to
  // the lowest "read point" of any pending read operations.
  HistoryCutoff history_cutoff;

  MonoDelta table_ttl;

  ShouldRetainDeleteMarkersInMajorCompaction retain_delete_markers_in_major_compaction{false};
};

struct CompactionSchemaInfo {
  TableType table_type;
  uint32_t schema_version = std::numeric_limits<uint32_t>::max();
  std::shared_ptr<const dockv::SchemaPacking> schema_packing;
  Uuid cotable_id;
  ColumnIds deleted_cols;
  std::optional<dockv::PackedRowVersion> packed_row_version;
  bool table_owns_vector_reverse_mapping = false;

  size_t pack_limit() const; // As usual, when not specified size is in bytes.

  // Whether we should keep original write time when combining columns updates into packed row.
  bool keep_write_time() const;
};

// Used to query latest possible schema version.
constexpr SchemaVersion kLatestSchemaVersion = std::numeric_limits<SchemaVersion>::max();

class SchemaPackingProvider {
 public:
  // Returns schema packing for provided cotable_id and schema version.
  // Passing Uuid::Nil() for cotable_id indicates the primary table.
  // If schema_version is kLatestSchemaVersion, then latest possible schema packing is returned.
  // Thread safety may be required depending on the usage.
  virtual Result<CompactionSchemaInfo> CotablePacking(
      const Uuid& cotable_id, uint32_t schema_version, HybridTime history_cutoff) = 0;

  // Returns schema packing for provided colocation_id and schema version.
  // If schema_version is kLatestSchemaVersion, then latest possible schema packing is returned.
  // Thread safety may be required depending on the usage.
  virtual Result<CompactionSchemaInfo> ColocationPacking(
      ColocationId colocation_id, uint32_t schema_version, HybridTime history_cutoff) = 0;

  // Check if the schema packing for provided cotable_id and schema version is existing,
  // this check will be skipped if the table has already been dropped.
  virtual Status CheckCotablePacking(
      const Uuid& cotable_id, uint32_t schema_version, HybridTime history_cutoff) {
    return Status::OK();
  }

  // Check if the schema packing for provided colocation_id and schema version is existing,
  // this check will be skipped if the table has already been dropped.
  virtual Status CheckColocationPacking(
      ColocationId colocation_id, uint32_t schema_version, HybridTime history_cutoff) {
    return Status::OK();
  }

  // Called when a table-level tombstone is written to regular DB (truncate / drop path).
  // Default is a no-op; RaftGroupMetadata advances the colocated tombstone-time cache watermark.
  virtual void NotifyTableTombstoneWritten(ColocationId colocation_id, HybridTime write_ht) {}
  virtual void NotifyTableTombstoneWritten(const Uuid& cotable_id, HybridTime write_ht) {}

  virtual ~SchemaPackingProvider() = default;
};

// A strategy for deciding how the history of old database operations should be retained during
// compactions. We may implement this differently in production and in tests.
class HistoryRetentionPolicy {
 public:
  virtual ~HistoryRetentionPolicy() = default;
  virtual HistoryRetentionDirective GetRetentionDirective() = 0;
  virtual HybridTime ProposedHistoryCutoff() = 0;
};

class DocVectorMetadataIteratorProvider {
 public:
  virtual ~DocVectorMetadataIteratorProvider() = default;
  virtual Result<docdb::IntentAwareIteratorWithBounds> CreateVectorMetadataIterator(
      const ReadHybridTime& read_ht, docdb::DocDBStatistics* statistics) const = 0;
};

struct CompactionHybridTimeConstraints {
  // Min and max time of entries participating in compaction.
  HybridTime input_min = HybridTime::kMax;
  HybridTime input_max = HybridTime::kMin;
  // Min time of entry that does not participate in compaction.
  HybridTime other_min = HybridTime::kMax;
  // The same floor split by where the other data lives: live files, whose bloom filters can
  // prove a key absent, versus the memtable and running transactions, which cannot be probed.
  // Maintained by HandleOtherFileRange / HandleOtherNonFileRange; floors_split records that both
  // were populated that way, so a consumer may rely on the split.
  HybridTime other_min_files = HybridTime::kMax;
  HybridTime other_min_nonfiles = HybridTime::kMax;
  bool floors_split = false;
  // Whether other_min_nonfiles is a true lower bound on the hybrid time of every entry that is
  // not yet in a file. The memtable frontier carries the batch hybrid time, and a write applied
  // with an external hybrid time (xCluster, index backfill) lands below its frontier, so the
  // floor cannot be trusted while such a write may still be in memory; the running-transaction
  // hybrid time is unknown until transactions are loaded. False keeps per-key tombstone drops
  // off; the coarse gate (other_min) is unaffected.
  bool nonfile_floor_trusted = false;
  // Latest batch hybrid time at which a write with an external hybrid time was applied (kMin:
  // never; kMax: at any time). A file whose smallest frontier is above this holds only entries
  // at or above that frontier (InitFrontiers puts min(batch, commit) there for every other
  // path), so such a file can be excluded from a probe by its frontier; any other file cannot.
  HybridTime external_writes_upto_ht = HybridTime::kMax;
  // The live files outside the compaction as seen when these constraints were computed, so a
  // consumer holding a file list from an earlier moment can tell whether a file appeared since.
  std::vector<uint64_t> other_file_numbers = {};
  // Min and max time of entries that could be repacked during this compaction.
  // I.e. we don't have entries that does not participate in compaction within this time interval.
  // Those constraints are exclusive.
  HybridTime repack_range_min = HybridTime::kMin;
  HybridTime repack_range_max = HybridTime::kMax;

  // input range should be already initialized before calling this function.
  void HandleOtherRange(HybridTime min, HybridTime max);
  void HandleOtherRange(
      const storage::UserFrontier& smallest, const storage::UserFrontier& largest);
  // Same as HandleOtherRange, also recording which floor the range belongs to.
  void HandleOtherFileRange(HybridTime min, HybridTime max);
  void HandleOtherFileRange(
      const storage::UserFrontier& smallest, const storage::UserFrontier& largest);
  void HandleOtherNonFileRange(HybridTime min, HybridTime max);
  void HandleOtherNonFileRange(
      const storage::UserFrontier& smallest, const storage::UserFrontier& largest);

  std::string ToString() const;
};

using CompactionHybridTimeLimitsProvider = std::function<CompactionHybridTimeConstraints(
    const std::vector<rocksdb::FileMetaData*>&)>;

// Fills the input range of `result` (input_min / input_max) from the frontiers of `inputs`. An
// input without frontiers (a file imported by bulk load) widens the range to everything.
void SetCompactionInputHybridTimeRange(
    const std::vector<rocksdb::FileMetaData*>& inputs, CompactionHybridTimeConstraints* result);

// Computes the constraints for compacting `inputs` of `db`: the input range from the inputs, then
// the "other" ranges in the order a write travels through a tablet: the running-transactions
// floor (`min_running_txn_ht`, present when the tablet has a transaction participant), the
// memtable frontiers, and every live file that is not an input. `external_writes_upto_ht` is the
// latest batch hybrid time at which a write with an external hybrid time was applied (kMin when
// none ever was, kMax when such writes may arrive at any time, e.g. under xCluster replication):
// the non-file floor is trusted only when every in-memory batch is newer than that, and when
// the running-transaction hybrid time, if given, is valid. `log_prefix` is prepended to
// diagnostics.
CompactionHybridTimeConstraints ComputeCompactionHybridTimeConstraints(
    rocksdb::DB& db, const std::vector<rocksdb::FileMetaData*>& inputs,
    std::optional<HybridTime> min_running_txn_ht, HybridTime external_writes_upto_ht,
    const std::string& log_prefix);

// Counters for column tombstones that compaction could not merge into the packed row they
// shadow. See DocDBCompactionFeed::Feed for why such a tombstone is kept rather than
// garbage-collected. Either pointer may be null, in which case the event is not recorded.
// Both count decisions rather than distinct tombstones.
struct CompactionMetrics {
  // Includes tombstones over columns schema packing never packs at all -- a YCQL collection,
  // which ProcessColumn refuses -- which are therefore kept permanently rather than transiently.
  CounterPtr column_tombstones_kept_unmerged;

  // Only non-zero while docdb_keep_unmerged_column_tombstones_over_packed_row is off, in which
  // case deleted column values are being resurrected and replicas that compact on opposite sides
  // of the flag diverge.
  CounterPtr column_tombstones_dropped_unmerged;

  // Per-key tombstone drop in partial compactions (docdb_reclamation_tombstone_drop): decisions
  // taken, and the bloom probes they issued. `dropped` counts tombstones actually removed from
  // the output; `kept_fail_closed` counts per-tombstone refusals to probe; `probes_disabled`
  // counts compactions that ran with the flag on but could not probe at all, in which case no
  // per-tombstone counter moves and the coarse gate decides alone.
  CounterPtr reclamation_tombstones_dropped;
  CounterPtr reclamation_tombstones_kept_probe_hit;
  CounterPtr reclamation_tombstones_kept_fail_closed;
  CounterPtr reclamation_probes_disabled;
  CounterPtr reclamation_probes;
};

CompactionMetrics CreateCompactionMetrics(const MetricEntityPtr& tablet_metric_entity);

std::shared_ptr<rocksdb::CompactionContextFactory> CreateCompactionContextFactory(
    std::shared_ptr<HistoryRetentionPolicy> retention_policy,
    const KeyBounds* key_bounds,
    const CompactionHybridTimeLimitsProvider& compaction_hybrid_time_limit_provider,
    SchemaPackingProvider* schema_packing_provider,
    DocVectorMetadataIteratorProvider* vector_metadata_iterator_provider,
    const CompactionMetrics& metrics);

// A history retention policy that can be configured manually. Useful in tests. This class is
// useful for testing and is thread-safe.
class ManualHistoryRetentionPolicy : public HistoryRetentionPolicy {
 public:
  HistoryRetentionDirective GetRetentionDirective() EXCLUDES(history_cutoff_mutex_) override;

  HybridTime ProposedHistoryCutoff() EXCLUDES(history_cutoff_mutex_) override;

  void SetHistoryCutoff(HistoryCutoff history_cutoff) EXCLUDES(history_cutoff_mutex_);

  void SetHistoryCutoff(HybridTime history_cutoff) EXCLUDES(history_cutoff_mutex_);

  void SetTableTTLForTests(MonoDelta ttl);

  // Emulates an index backfill in progress, which asks compactions to retain delete markers.
  void SetRetainDeleteMarkersForTests(bool retain);

 private:
  std::mutex history_cutoff_mutex_;
  HistoryCutoff history_cutoff_ GUARDED_BY(history_cutoff_mutex_)
      = { HybridTime::kInvalid, HybridTime::kMin };
  std::atomic<MonoDelta> table_ttl_{MonoDelta::kMax};
  std::atomic<bool> retain_delete_markers_{false};
};

HybridTime GetHistoryCutoffForKey(Slice coprefix, HistoryCutoff cutoff_info);

}  // namespace yb::docdb

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <folly/Executor.h>
#include <folly/container/F14Map.h>
#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/IndexReader.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::connector::hive {

class HiveConfig;

/// Provides index lookup for Hive-metastore-backed tables.
///
/// Owns format-agnostic orchestration: remaining filter evaluation, output
/// projection, and partition routing. Delegates format-specific reads to
/// either the built-in FileIndexReader (for Nimble) or external
/// SplitIndexReader implementations registered via IndexReaderFactoryRegistry.
/// Handles partitioned tables by routing probe rows to the correct file based
/// on partition column values from split metadata.
class HiveIndexSource : public IndexSource,
                        public std::enable_shared_from_this<HiveIndexSource> {
 public:
  HiveIndexSource(
      const RowTypePtr& requestType,
      const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions,
      const RowTypePtr& outputType,
      HiveTableHandlePtr tableHandle,
      const ColumnHandleMap& columnHandles,
      FileHandleFactory* fileHandleFactory,
      ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<HiveConfig>& hiveConfig,
      folly::Executor* executor);

  ~HiveIndexSource() override = default;

  void addSplits(std::vector<std::shared_ptr<ConnectorSplit>> splits) override;

  std::shared_ptr<ResultIterator> lookup(const Request& request) override;

  std::unordered_map<std::string, RuntimeMetric> runtimeStats() override;

 private:
  friend class HiveLookupIterator;

  memory::MemoryPool* pool() const {
    return pool_;
  }

  const RowTypePtr& outputType() const {
    return outputType_;
  }

  const RowVectorPtr& getEmptyOutput() {
    if (!emptyOutput_) {
      emptyOutput_ = RowVector::createEmpty(outputType_, pool_);
    }
    return emptyOutput_;
  }

  // Evaluates remainingFilter on the specified vector. Returns number of rows
  // passed. Populates filterEvalCtx_.selectedIndices and selectedBits if only
  // some rows passed the filter. If none or all rows passed
  // filterEvalCtx_.selectedIndices and selectedBits are not updated.
  vector_size_t evaluateRemainingFilter(RowVectorPtr& rowVector);

  // Applies non-index equi-join conditions as post-read equality filters.
  // Compares reader output column values against request (probe) values
  // for each non-index join condition. Returns the number of rows that pass
  // all conditions. Populates 'passingIndices' with the indices of rows that
  // pass when not all rows pass.
  vector_size_t applyNonIndexConditions(
      const RowVectorPtr& output,
      const RowVectorPtr& request,
      const BufferPtr& inputHits,
      BufferPtr& passingIndices);

  // Projects reader output to the final output type. 'remainingIndices'
  // (optional) selects rows that passed the remaining filter. Partition
  // columns are emitted by the reader directly via setConstantValue() on
  // the scan spec, so they are projected like any other column here.
  RowVectorPtr projectOutput(
      vector_size_t numRows,
      const BufferPtr& remainingIndices,
      const RowVectorPtr& rowVector);

  // Builds a scan spec from the configured reader output type, projected
  // subfields, filters, and data columns. Optionally applies per-split
  // partition constants by calling setConstantValue() on the matching
  // partition-column children. The returned spec is independent of any
  // existing scan spec — call this once per partition group, since each
  // group's partition values overwrite the spec in place.
  std::shared_ptr<common::ScanSpec> buildScanSpec(
      const std::unordered_map<std::string, std::optional<std::string>>*
          partitionValues = nullptr);

  void init(
      const ColumnHandleMap& assignments,
      const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions);

  // Categorizes join conditions into three buckets, populating one member
  // per bucket:
  // 1. Index conditions → indexLookupConditions_, pushed to the reader.
  //    Filters on index columns are converted to index lookup conditions.
  // 2. Partition routing conditions → partitionRoutingConditions_, used to
  //    dispatch probe rows to the correct partition's files.
  // 3. Non-index conditions → nonIndexConditionNames_ as (table column,
  //    probe column) pairs; later resolved to NonIndexCondition entries
  //    once readerOutputType_ is built. Used for post-read equality filters
  //    (e.g., bucket columns in colocated joins).
  // Validates that at least one index condition exists and that every
  // partition column has a corresponding routing condition.
  void categorizeJoinConditions(
      const std::vector<core::IndexLookupConditionPtr>& joinConditions,
      const ColumnHandleMap& assignments);

  using IndexLookupConditionMap =
      folly::F14FastMap<std::string, core::IndexLookupConditionPtr>;

  // Builds a map from table column name to the matching join condition.
  // Each condition's key is rewritten from its probe-side input name to the
  // table column name via 'assignments'. Throws on duplicate keys.
  IndexLookupConditionMap buildIndexLookupConditionMap(
      const std::vector<core::IndexLookupConditionPtr>& joinConditions,
      const ColumnHandleMap& assignments) const;

  // Walks indexColumns() in order. For each column with either a pushdown
  // filter or an index lookup condition, pushes an entry to
  // indexLookupConditions_ (converting filters via
  // tryConvertFilterToIndexLookupCondition). Stops at the first prefix gap
  // or range condition. Records every column it consumed in
  // 'processedConditions'.
  void processIndexColumnConditions(
      const IndexLookupConditionMap& conditionMap,
      folly::F14FastSet<std::string>& processedConditions);

  // Tries to convert the pushdown filter on 'indexColumn' into an index
  // lookup condition (point or range). Returns true if more index columns
  // can be processed after this one (i.e., a point lookup was emitted),
  // false if the walk should stop (range emitted, or filter could not be
  // converted). On success, removes the consumed entry from filters_.
  bool tryConvertFilterToIndexLookupCondition(
      const std::string& indexColumn,
      const common::SubfieldFilters::iterator& filterIt);

  // Categorizes join conditions on non-index columns into either
  // partitionRoutingConditions_ (when the column is a partition key) or
  // nonIndexConditionNames_ (otherwise). Skips entries already consumed by
  // processIndexColumnConditions (tracked in 'processedConditions') and
  // rejects unprocessed entries on index columns as a prefix-gap violation.
  void processNonIndexColumnConditions(
      const IndexLookupConditionMap& conditionMap,
      const folly::F14FastSet<std::string>& processedConditions);

  // Final consistency checks: every input join condition lands in exactly
  // one bucket, at least one index condition exists, and every partition
  // column has a corresponding routing condition.
  // 'numIndexConditionsConsumed' is the count of join conditions consumed
  // as index conditions (excludes index conditions converted from pushdown
  // filters, since those did not come from joinConditions).
  void validateCategorization(
      size_t numInputConditions,
      size_t numIndexConditionsConsumed) const;

  // Initializes the remaining filter:
  // - Compiles the remaining filter expression.
  // - Identifies columns to eagerly materialize.
  // - Adds columns referenced by remaining filter but not projected.
  // - Extracts and processes subfields from the remaining filter.
  void initRemainingFilter(
      std::vector<std::string>& readColumnNames,
      std::vector<TypePtr>& readColumnTypes);

  // Creates a reader for a single split using an external IndexReaderFactory.
  void createCustomIndexReader(
      const IndexReaderFactory& factory,
      std::shared_ptr<const HiveConnectorSplit> split,
      std::vector<std::unique_ptr<SplitIndexReader>>& readers);

  // Creates a FileIndexReader for a single split. The reader internally
  // dispatches to the format-specific IndexReader for the split's file
  // format.
  void createFileIndexReader(
      std::shared_ptr<const HiveConnectorSplit> split,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      std::vector<std::unique_ptr<SplitIndexReader>>& readers);

  // Creates readers from splits, preferring an IndexReaderFactory registered
  // for the split's file format and falling back to FileIndexReader for
  // formats it supports natively. All readers share the given scanSpec.
  void createReadersFromSplits(
      std::vector<std::shared_ptr<const HiveConnectorSplit>> splits,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      std::vector<std::unique_ptr<SplitIndexReader>>& readers);

  // Groups splits by their routing-column partition values, then builds
  // one PartitionReaderGroup per distinct partition with its own scan
  // spec (partition columns set as constants) and routing constants for
  // runtime probe matching. Populates partitionGroups_. Called from
  // addSplits() when partitionRoutingConditions_ is non-empty.
  void buildPartitionGroups(
      std::vector<std::shared_ptr<const HiveConnectorSplit>> hiveSplits);

  // Buckets 'hiveSplits' by the tuple of values for the columns in
  // partitionRoutingConditions_. Map keys are opaque encodings of those
  // value tuples — used only for grouping, not for inspection.
  folly::F14FastMap<
      std::string,
      std::vector<std::shared_ptr<const HiveConnectorSplit>>>
  groupSplitsByPartitionValues(
      std::vector<std::shared_ptr<const HiveConnectorSplit>> hiveSplits) const;

  // Per-iteration timing breakdown for index lookups.
  struct IterationStats {
    uint64_t setupWallNs{0};
    uint64_t setupCpuNs{0};
    uint64_t readWallNs{0};
    uint64_t readCpuNs{0};
    uint64_t outputWallNs{0};
    uint64_t outputCpuNs{0};
    uint64_t filterWallNs{0};
    uint64_t filterCpuNs{0};

    static constexpr std::string_view kConnectorLookupWallNanos{
        "connectorLookupWallNanos"};
    static constexpr std::string_view kConnectorResultPrepareCpuNanos{
        "connectorResultPrepareCpuNanos"};
    static constexpr std::string_view kIndexSetupWallNanos{
        "connectorIndexSetupWallNanos"};
    static constexpr std::string_view kIndexSetupCpuNanos{
        "connectorIndexSetupCpuNanos"};
    static constexpr std::string_view kIndexReadWallNanos{
        "connectorIndexReadWallNanos"};
    static constexpr std::string_view kIndexReadCpuNanos{
        "connectorIndexReadCpuNanos"};
    static constexpr std::string_view kPostFilterWallNanos{
        "connectorPostFilterWallNanos"};
    static constexpr std::string_view kPostFilterCpuNanos{
        "connectorPostFilterCpuNanos"};
  };

  // Records per-iteration timing breakdown using addOperatorRuntimeStats to
  // preserve per-call count/min/max granularity.
  void recordIterationStats(const IterationStats& iterationStats);

  FileHandleFactory* const fileHandleFactory_;
  ConnectorQueryCtx* const connectorQueryCtx_;
  const std::shared_ptr<HiveConfig> hiveConfig_;
  memory::MemoryPool* const pool_;
  core::ExpressionEvaluator* const expressionEvaluator_;
  const uint32_t maxRowsPerIndexRequest_;

  const HiveTableHandlePtr tableHandle_;
  const RowTypePtr requestType_;
  const RowTypePtr outputType_;
  folly::Executor* const executor_;

  // All index lookup conditions including equal lookup keys converted to
  // EqualIndexLookupConditions and original non-filter index lookup conditions.
  // This is passed to FileIndexReader.
  std::vector<core::IndexLookupConditionPtr> indexLookupConditions_;

  // Non-index equi-join conditions: join keys that are not index columns
  // (e.g., bucket columns used for colocated joins). Applied as post-read
  // equality filters during lookup.
  struct NonIndexCondition {
    column_index_t readerOutputIndex;
    column_index_t requestIndex;
  };
  std::vector<NonIndexCondition> nonIndexConditions_;

  // Scratch state populated by categorizeJoinConditions(). Holds (table
  // column name, probe column name) pairs that need later resolution to
  // NonIndexCondition entries — deferred because they require
  // readerOutputType_, which init() builds after categorization.
  std::vector<std::pair<std::string, std::string>> nonIndexConditionNames_;

  // Partition routing condition: join condition on a partition column.
  // Used to route probe rows to the correct file based on partition values.
  struct PartitionRoutingCondition {
    // Table-side column name (handle->name()), used to look up the partition
    // value from HiveConnectorSplit::partitionKeys.
    std::string partitionColumnName;
    // Index of the corresponding probe column in requestType_.
    column_index_t requestIndex;
    // Logical type used to parse the partition value into a typed constant.
    TypePtr type;
    // True when the partition value string is encoded as days-since-epoch
    // (alternative to a YYYY-MM-DD date string). Mirrors the same flag on
    // HiveColumnHandle and is consumed by newConstantFromString.
    bool isPartitionDateValueDaysSinceEpoch{false};
  };
  std::vector<PartitionRoutingCondition> partitionRoutingConditions_;

  // Partition column handles, keyed by table column name (handle->name()).
  // Populated from kPartitionKey assignments in init(). Used to:
  // - classify a join condition as a routing condition
  // - feed partition handles into makeScanSpec()
  // - look up dataType() and isPartitionDateValueDaysSinceEpoch() when
  //   parsing partition value strings into typed constants
  std::unordered_map<std::string, FileColumnHandlePtr> partitionKeyHandles_;

  // Splits sharing the same partition values, grouped during addSplits().
  // Under standard grouped execution a lifespan can receive splits from
  // multiple partitions, in which case there is one group per distinct
  // partition. Under partition-aware grouped execution a lifespan
  // receives splits for exactly one (bucket, partition) pair, so this
  // vector typically holds a single group. Each group owns a fresh scan
  // spec with setConstantValue() applied to its partition-column
  // children, so the underlying reader emits the
  // partition value as a constant for every row.
  struct PartitionReaderGroup {
    // Index readers for splits in this partition group.
    std::vector<std::unique_ptr<SplitIndexReader>> readers;
    // Typed constants aligned with partitionRoutingConditions_, used by
    // findPartitionGroup() to match against probe row values.
    std::vector<VectorPtr> routingConstants;
  };
  std::vector<PartitionReaderGroup> partitionGroups_;

  // Creates a lookup iterator over the given readers: a single
  // HiveLookupIterator if there's only one reader, otherwise a
  // UnionResultIterator that k-way-merges per-reader results.
  std::shared_ptr<ResultIterator> createLookupIterator(
      const Request& request,
      const SplitIndexReader::Options& options,
      std::vector<std::unique_ptr<SplitIndexReader>>& readers);

  // Finds the partition reader group matching the given probe input's
  // partition column values. Returns nullptr if no group matches; the
  // caller (lookup()) translates that to an empty iterator. Inspects the
  // first probe row only — every row in a batch is expected to share the
  // same partition values under grouped execution.
  PartitionReaderGroup* FOLLY_NULLABLE
  findPartitionGroup(const RowVectorPtr& probeInput);

  // Filters for pushdown. Includes subfield filters from table handle and
  // filters converted from constant index lookup conditions.
  common::SubfieldFilters filters_;

  // Remaining filter expression set after filter pushdown.
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;
  // Subfields referenced by the remaining filter.
  std::vector<common::Subfield> remainingFilterSubfields_;
  // Total time spent on evaluating remaining filter in nanoseconds.
  uint64_t remainingFilterTimeNs_{0};
  // Field indices referenced in both remaining filter and output type. These
  // columns need to be materialized eagerly to avoid missing values in output.
  std::vector<column_index_t> remainingEagerlyLoadFields_;

  // Filter evaluation state for remaining filter.
  SelectivityVector remainingFilterRows_;
  VectorPtr remainingFilterResult_;
  DecodedVector remainingFilterLazyDecoded_;
  SelectivityVector remainingFilterLazyBaseRows_;
  exec::FilterEvalCtx remainingFilterEvalCtx_;

  // Points to subfields from both output columns' required subfields and
  // remainingFilterSubfields_. Used to tell the reader which nested fields to
  // project out.
  folly::F14FastMap<std::string, std::vector<const common::Subfield*>>
      projectedSubfields_;

  // Output type for the index reader. Includes partition columns when they
  // appear in outputType_; the per-split reader emits their values as
  // constants via the scan spec's setConstantValue().
  RowTypePtr readerOutputType_;
  // All index readers (both built-in and external). FileIndexReader
  // (Nimble) and external readers both implement SplitIndexReader.
  // Created by addSplits().
  std::vector<std::unique_ptr<SplitIndexReader>> readers_;

  // Cached empty output vector.
  RowVectorPtr emptyOutput_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<IoStats> ioStats_;

  // Per-call timing stats accumulated via addOperatorRuntimeStats().
  std::unordered_map<std::string, RuntimeMetric> runtimeStats_;
};

} // namespace facebook::velox::connector::hive

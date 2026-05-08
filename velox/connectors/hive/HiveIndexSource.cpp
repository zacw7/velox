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
#include "velox/connectors/hive/HiveIndexSource.h"

#include <folly/ScopeGuard.h>
#include <folly/container/F14Set.h>
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/FileDataSource.h"
#include "velox/connectors/hive/FileIndexReader.h"
#include "velox/connectors/hive/FileSplitReader.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/FieldReference.h"
#include "velox/vector/LazyVector.h"

namespace facebook::velox::connector::hive {
namespace {

// Extracts a constant value from a point lookup filter (single value equality).
// Returns nullopt if the filter is not a point lookup or cannot be converted.
std::optional<variant> extractPointLookupValue(const common::Filter* filter) {
  VELOX_CHECK_NOT_NULL(filter);

  switch (filter->kind()) {
    case common::FilterKind::kBigintRange: {
      const auto* range = filter->as<common::BigintRange>();
      if (range->isSingleValue()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kDoubleRange: {
      const auto* range = filter->as<common::DoubleRange>();
      if (!range->lowerUnbounded() && !range->upperUnbounded() &&
          range->lower() == range->upper() && !range->lowerExclusive() &&
          !range->upperExclusive()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kFloatRange: {
      const auto* range = filter->as<common::FloatRange>();
      if (!range->lowerUnbounded() && !range->upperUnbounded() &&
          range->lower() == range->upper() && !range->lowerExclusive() &&
          !range->upperExclusive()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kBytesRange: {
      const auto* range = filter->as<common::BytesRange>();
      if (range->isSingleValue()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kBytesValues: {
      const auto* values = filter->as<common::BytesValues>();
      if (values->values().size() == 1) {
        return variant(*values->values().begin());
      }
      return std::nullopt;
    }
    case common::FilterKind::kBoolValue: {
      const auto* boolFilter = filter->as<common::BoolValue>();
      // BoolValue doesn't expose value() getter - use testBool to determine the
      // value
      return variant(boolFilter->testBool(true));
    }
    default:
      return std::nullopt;
  }
}

// Extracts range bounds from a range filter.
// Returns a pair of (lower, upper) variants. If a bound is unbounded, the
// corresponding variant will be null.
// Returns nullopt if the filter is not a range filter or cannot be converted.
std::optional<std::pair<variant, variant>> extractRangeBounds(
    const common::Filter* filter) {
  VELOX_CHECK_NOT_NULL(filter);

  switch (filter->kind()) {
    case common::FilterKind::kBigintRange: {
      const auto* range = filter->as<common::BigintRange>();
      return std::make_pair(variant(range->lower()), variant(range->upper()));
    }
    case common::FilterKind::kDoubleRange: {
      const auto* range = filter->as<common::DoubleRange>();
      if (range->lowerUnbounded() || range->upperUnbounded()) {
        // Cannot convert unbounded ranges to BetweenCondition.
        return std::nullopt;
      }
      return std::make_pair(variant(range->lower()), variant(range->upper()));
    }
    case common::FilterKind::kFloatRange: {
      const auto* range = filter->as<common::FloatRange>();
      if (range->lowerUnbounded() || range->upperUnbounded()) {
        return std::nullopt;
      }
      return std::make_pair(variant(range->lower()), variant(range->upper()));
    }
    case common::FilterKind::kBytesRange: {
      const auto* range = filter->as<common::BytesRange>();
      if (!range->isSingleValue()) {
        // Only convert bounded range filters.
        return std::make_pair(variant(range->lower()), variant(range->upper()));
      }
      return std::nullopt;
    }
    default:
      return std::nullopt;
  }
}

// Creates an EqualIndexLookupCondition with a constant value.
core::IndexLookupConditionPtr createEqualConditionWithConstant(
    const std::string& columnName,
    const TypePtr& type,
    const variant& value) {
  auto keyExpr = std::make_shared<core::FieldAccessTypedExpr>(type, columnName);
  auto constantExpr = std::make_shared<core::ConstantTypedExpr>(type, value);
  return std::make_shared<core::EqualIndexLookupCondition>(
      std::move(keyExpr), std::move(constantExpr));
}

// Creates a BetweenIndexLookupCondition with constant bounds.
core::IndexLookupConditionPtr createBetweenConditionWithConstants(
    const std::string& columnName,
    const TypePtr& type,
    const variant& lowerValue,
    const variant& upperValue) {
  auto keyExpr = std::make_shared<core::FieldAccessTypedExpr>(type, columnName);
  auto lowerExpr = std::make_shared<core::ConstantTypedExpr>(type, lowerValue);
  auto upperExpr = std::make_shared<core::ConstantTypedExpr>(type, upperValue);
  return std::make_shared<core::BetweenIndexLookupCondition>(
      std::move(keyExpr), std::move(lowerExpr), std::move(upperExpr));
}

// Checks that a HiveColumnHandle is a regular column type.
void checkColumnHandleIsRegular(const FileColumnHandle& handle) {
  VELOX_CHECK_EQ(
      handle.columnType(),
      FileColumnHandle::ColumnType::kRegular,
      "Expected regular column, got {} for column {}",
      FileColumnHandle::columnTypeName(handle.columnType()),
      handle.name());
}

// Gets the table column name from assignments for a given input column name.
// Throws if not found in assignments.
std::string getTableColumnName(
    const std::string& inputColumnName,
    const connector::ColumnHandleMap& assignments) {
  auto it = assignments.find(inputColumnName);
  VELOX_USER_CHECK(
      it != assignments.end(),
      "Column not found in assignments: {}",
      inputColumnName);
  const auto* handle =
      checkedPointerCast<const HiveColumnHandle>(it->second.get());
  return handle->name();
}

// Creates a new FieldAccessTypedExpr with the given column name but same type.
core::FieldAccessTypedExprPtr renameFieldAccess(
    const core::FieldAccessTypedExprPtr& field,
    const std::string& newName) {
  return std::make_shared<core::FieldAccessTypedExpr>(field->type(), newName);
}

// Converts an index lookup condition's key name from input column name to table
// column name. Returns a new condition with the converted key name.
core::IndexLookupConditionPtr convertConditionKeyName(
    const core::IndexLookupConditionPtr& condition,
    const connector::ColumnHandleMap& assignments) {
  const auto tableColumnName =
      getTableColumnName(condition->key->name(), assignments);
  auto newKey = renameFieldAccess(condition->key, tableColumnName);

  if (auto equalCondition =
          std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(
              condition)) {
    return std::make_shared<core::EqualIndexLookupCondition>(
        std::move(newKey), equalCondition->value);
  }
  if (auto betweenCondition =
          std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(
              condition)) {
    return std::make_shared<core::BetweenIndexLookupCondition>(
        std::move(newKey), betweenCondition->lower, betweenCondition->upper);
  }

  VELOX_UNREACHABLE(
      "Unsupported IndexLookupCondition type: {}", condition->toString());
}

// Filters input indices based on selected indices from filter evaluation.
BufferPtr filterIndices(
    vector_size_t numRows,
    const BufferPtr& selectedIndices,
    const BufferPtr& inputIndices,
    memory::MemoryPool* pool) {
  auto resultIndices = allocateIndices(numRows, pool);
  auto* rawResultIndices = resultIndices->asMutable<vector_size_t>();
  const auto* rawSelected = selectedIndices->as<vector_size_t>();
  const auto* rawInputIndices = inputIndices->as<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawResultIndices[i] = rawInputIndices[rawSelected[i]];
  }
  return resultIndices;
}
} // namespace

namespace {

// Merges results from multiple split-level ResultIterators in inputHit order.
// Each split independently produces results with sorted inputHits. This
// iterator interleaves rows across splits to maintain the global non-decreasing
// inputHit ordering required by IndexLookupJoin (for left join missed-row
// detection and the check in prepareLookupResult).
//
// Uses a k-way merge: buffers one Result per split, then repeatedly picks
// the split with the smallest current request index and copies all its rows
// for that index before moving on.
class UnionResultIterator : public IndexSource::ResultIterator {
 public:
  UnionResultIterator(
      std::vector<std::shared_ptr<IndexSource::ResultIterator>> splitIters,
      const RowTypePtr& outputType,
      memory::MemoryPool* pool)
      : outputType_(outputType), pool_(pool) {
    VELOX_CHECK_GT(
        splitIters.size(),
        1,
        "UnionResultIterator requires at least two iterators");
    splits_.reserve(splitIters.size());
    for (auto& iter : splitIters) {
      splits_.emplace_back(std::move(iter));
    }
  }

  bool hasNext() override {
    for (const auto& split : splits_) {
      if (!split.hasExhausted()) {
        return true;
      }
    }
    return false;
  }

  std::optional<std::unique_ptr<IndexSource::Result>> next(
      vector_size_t size,
      ContinueFuture& future) override {
    // Fetch results for all non-exhausted splits that need data.
    for (auto& split : splits_) {
      if (split.needFetchResult()) {
        if (!split.fetchResult(size, future)) {
          VELOX_CHECK(future.valid(), "Async return requires a valid future");
          return std::nullopt;
        }
      }
    }

    // Merge rows from all active splits in inputHit order by repeatedly
    // picking the split with the smallest current request index and copying
    // all its rows for that index.
    auto mergedInputHits = allocateIndices(size, pool_);
    auto* rawMergedHits = mergedInputHits->asMutable<vector_size_t>();
    auto mergedOutput = BaseVector::create<RowVector>(outputType_, size, pool_);

    vector_size_t numOutput = 0;
    while (numOutput < size) {
      int minSplitIndex = -1;
      auto minRequestIndex = std::numeric_limits<vector_size_t>::max();
      for (size_t i = 0; i < splits_.size(); ++i) {
        if (splits_[i].hasResult() &&
            splits_[i].currentRequestIndex() < minRequestIndex) {
          minSplitIndex = static_cast<int>(i);
          minRequestIndex = splits_[i].currentRequestIndex();
        }
      }
      if (minSplitIndex < 0) {
        // All splits are exhausted with no buffered data.
        break;
      }

      auto& split = splits_[minSplitIndex];
      VELOX_CHECK_LE(numOutput, size);
      numOutput += split.fillResult(
          minRequestIndex,
          mergedOutput,
          numOutput,
          rawMergedHits,
          size - numOutput);

      // Stop if this split's buffer is consumed but not exhausted. We must
      // refill it before continuing to avoid emitting larger inputHits from
      // other splits that would violate non-decreasing order across next()
      // calls.
      if (split.needFetchResult()) {
        break;
      }
    }

    if (numOutput == 0) {
      return nullptr;
    }

    mergedInputHits->setSize(numOutput * sizeof(vector_size_t));
    mergedOutput->resize(numOutput);
    return std::make_unique<IndexSource::Result>(
        std::move(mergedInputHits), std::move(mergedOutput));
  }

 private:
  // Tracks iteration state for a single split's ResultIterator. Buffers
  // one Result at a time and tracks the current read position within it.
  struct SplitState {
    explicit SplitState(std::shared_ptr<IndexSource::ResultIterator> splitIter)
        : iter(std::move(splitIter)) {}

    const std::shared_ptr<IndexSource::ResultIterator> iter;
    // Current buffered result from this split, or nullptr if not yet fetched.
    std::unique_ptr<IndexSource::Result> result;
    // Next row to read within 'result'.
    vector_size_t resultOffset{0};
    // True when the underlying iterator has no more results.
    bool exhausted{false};

    // Returns true if there are unconsumed rows in the current buffered result.
    bool hasResult() const {
      return result != nullptr && resultOffset < result->size();
    }

    // Returns true if the underlying iterator has no more results.
    bool hasExhausted() const {
      return exhausted;
    }

    // Returns true if the split needs to fetch the next result batch.
    bool needFetchResult() const {
      return !hasResult() && !hasExhausted();
    }

    // Returns the request index (inputHit) of the current row in the buffer.
    vector_size_t currentRequestIndex() const {
      VELOX_CHECK(hasResult());
      return result->inputHits->as<const vector_size_t>()[resultOffset];
    }

    // Copies buffered rows matching the given request index to the output,
    // up to maxRows. Returns the number of rows copied.
    vector_size_t fillResult(
        vector_size_t requestIndex,
        const RowVectorPtr& output,
        vector_size_t outputOffset,
        vector_size_t* rawHits,
        vector_size_t maxRows) {
      VELOX_CHECK(hasResult());
      // Count contiguous rows with the same request index.
      const auto* hits = result->inputHits->as<const vector_size_t>();
      vector_size_t count = 0;
      while (count < maxRows && resultOffset + count < result->size() &&
             hits[resultOffset + count] == requestIndex) {
        ++count;
      }
      if (count > 0) {
        output->copy(result->output.get(), outputOffset, resultOffset, count);
        std::fill(
            rawHits + outputOffset,
            rawHits + outputOffset + count,
            requestIndex);
        resultOffset += count;
      }
      return count;
    }

    // Fetches the next non-empty result from the underlying iterator.
    // Returns true when data is ready (or split is exhausted), false if async.
    bool fetchResult(vector_size_t size, ContinueFuture& future) {
      VELOX_CHECK(
          !hasResult(), "Must consume current result before fetching next");
      while (!exhausted) {
        if (!iter->hasNext()) {
          exhausted = true;
          return true;
        }
        auto resultOpt = iter->next(size, future);
        if (!resultOpt.has_value()) {
          return false;
        }
        auto fetchedResult = std::move(resultOpt).value();
        if (fetchedResult == nullptr) {
          exhausted = true;
          return true;
        }
        // Skip empty results (e.g., when all rows are filtered out by
        // remaining filter) and continue fetching.
        if (fetchedResult->size() == 0) {
          continue;
        }
        result = std::move(fetchedResult);
        resultOffset = 0;
        return true;
      }
      VELOX_UNREACHABLE();
    }
  };

  // Output schema used to allocate merged result vectors.
  const RowTypePtr outputType_;
  memory::MemoryPool* const pool_;
  // Per-split state for buffering and tracking iteration progress. Not const
  // because elements are mutated during iteration (result, resultOffset,
  // exhausted), and const vector makes elements const via const T& access.
  std::vector<SplitState> splits_;
};

// Iterator that produces no results. Returned by lookup() when no
// partition group matches the probe input; IndexLookupJoin requires a
// non-null iterator (it deref-checks at IndexLookupJoin.cpp's
// getLookupResults), so we hand it an iterator that immediately reports
// hasNext() == false.
class EmptyResultIterator : public IndexSource::ResultIterator {
 public:
  bool hasNext() override {
    return false;
  }

  std::optional<std::unique_ptr<IndexSource::Result>> next(
      vector_size_t /*size*/,
      ContinueFuture& /*future*/) override {
    return std::nullopt;
  }
};

} // namespace

// Scope-attached timer that accumulates wall and CPU time into the
// corresponding iterationStats fields when the attached block exits.
// Expects a local variable named 'iterationStats' of type
// HiveIndexSource::IterationStats.
//
// Usage:
//   RECORD_CPU_WALL(setup) {
//     // timed work
//   }
#define RECORD_CPU_WALL(name)                                        \
  if (DeltaCpuWallTimer _timer_##name([&](const CpuWallTiming& _t) { \
        iterationStats.name##WallNs += _t.wallNanos;                 \
        iterationStats.name##CpuNs += _t.cpuNanos;                   \
      });                                                            \
      true)

/// Iterates over results from a SplitIndexReader and applies HiveIndexSource's
/// format-agnostic orchestration: remaining filter evaluation and output
/// projection.
class HiveLookupIterator : public IndexSource::ResultIterator {
 public:
  HiveLookupIterator(
      std::shared_ptr<HiveIndexSource> indexSource,
      SplitIndexReader* indexReader,
      IndexSource::Request request,
      SplitIndexReader::Options options)
      : indexSource_(std::move(indexSource)),
        indexReader_(indexReader),
        request_(std::move(request)),
        options_(options) {}

  bool hasNext() override {
    return state_ != State::kEnd;
  }

  std::optional<std::unique_ptr<IndexSource::Result>> next(
      vector_size_t size,
      ContinueFuture& /*unused*/) override {
    if (state_ == State::kEnd) {
      return nullptr;
    }

    HiveIndexSource::IterationStats iterationStats;
    SCOPE_EXIT {
      indexSource_->recordIterationStats(iterationStats);
    };

    // Initialize lookup on first call.
    if (state_ == State::kInit) {
      RECORD_CPU_WALL(setup) {
        indexReader_->startLookup(request_, options_);
      }
      setState(State::kRead);
    }

    if (!indexReader_->hasNext()) {
      setState(State::kEnd);
      return nullptr;
    }
    return getOutput(size, iterationStats);
  }

 private:
  // State of the iterator.
  enum class State {
    // Initial state after creation.
    kInit,
    // After lookup request has been set in index reader.
    kRead,
    // After all data has been read.
    kEnd,
  };

  // Sets the state with validation of allowed transitions.
  // Allowed transitions:
  //   kInit -> kRead (when request is set)
  //   kInit -> kEnd (when no matches on first call)
  //   kRead -> kEnd (when all data has been read)
  void setState(State newState) {
    switch (state_) {
      case State::kInit:
        VELOX_CHECK(
            newState == State::kRead || newState == State::kEnd,
            "Invalid state transition from {} to {}",
            stateName(state_),
            stateName(newState));
        break;
      case State::kRead:
        VELOX_CHECK(
            newState == State::kEnd,
            "Invalid state transition from {} to {}",
            stateName(state_),
            stateName(newState));
        break;
      case State::kEnd:
        VELOX_FAIL(
            "Invalid state transition from {} to {}",
            stateName(state_),
            stateName(newState));
    }
    state_ = newState;
  }

  static std::string stateName(State state) {
    switch (state) {
      case State::kInit:
        return "kInit";
      case State::kRead:
        return "kRead";
      case State::kEnd:
        return "kEnd";
      default:
        VELOX_UNREACHABLE("Unknown state {}", static_cast<int>(state));
    }
  }

  std::unique_ptr<IndexSource::Result> getOutput(
      vector_size_t size,
      HiveIndexSource::IterationStats& iterationStats) {
    std::unique_ptr<IndexSource::Result> result;
    RECORD_CPU_WALL(read) {
      result = indexReader_->next(size);
    }
    if (result == nullptr) {
      VELOX_CHECK(!indexReader_->hasNext());
      setState(State::kEnd);
      return nullptr;
    }

    // Apply non-index equi-join conditions as post-read equality filters.
    if (!indexSource_->nonIndexConditions_.empty()) {
      BufferPtr passingIndices{nullptr};
      const auto numPassing = indexSource_->applyNonIndexConditions(
          result->output, request_.input, result->inputHits, passingIndices);
      if (numPassing == 0) {
        return getEmptyResult();
      }
      if (passingIndices) {
        result->inputHits = filterIndices(
            numPassing, passingIndices, result->inputHits, indexSource_->pool_);
        result->output = exec::wrap(numPassing, passingIndices, result->output);
      }
    }

    // Evaluate the remaining filter (if any) into a row-selection buffer,
    // then run a single projection that applies the selection AND interleaves
    // partition constants into the final outputType_ shape.
    BufferPtr remainingIndices{nullptr};
    auto numRows = result->output->size();
    if (indexSource_->remainingFilterExprSet_ != nullptr) {
      RECORD_CPU_WALL(filter) {
        numRows = indexSource_->evaluateRemainingFilter(result->output);
      }
      if (numRows == 0) {
        return getEmptyResult();
      }
      if (numRows != result->output->size()) {
        remainingIndices =
            indexSource_->remainingFilterEvalCtx_.selectedIndices;
        result->inputHits = filterIndices(
            numRows, remainingIndices, result->inputHits, indexSource_->pool_);
      }
    }

    RECORD_CPU_WALL(output) {
      result->output = indexSource_->projectOutput(
          numRows, remainingIndices, result->output);
    }
    return result;
  }

  std::unique_ptr<IndexSource::Result> getEmptyResult() {
    if (emptyResult_ == nullptr) {
      emptyResult_ = std::make_unique<IndexSource::Result>(
          allocateIndices(0, indexSource_->pool_),
          indexSource_->getEmptyOutput());
    }
    return std::make_unique<IndexSource::Result>(
        emptyResult_->inputHits, emptyResult_->output);
  }

  const std::shared_ptr<HiveIndexSource> indexSource_;
  // Raw pointer to index reader for lookup operations.
  SplitIndexReader* const indexReader_;
  const IndexSource::Request request_;
  const SplitIndexReader::Options options_;

  State state_{State::kInit};
  // Cached empty result for reuse when no rows pass the remaining filter.
  std::unique_ptr<IndexSource::Result> emptyResult_;
};

HiveIndexSource::HiveIndexSource(
    const RowTypePtr& requestType,
    const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions,
    const RowTypePtr& outputType,
    HiveTableHandlePtr tableHandle,
    const ColumnHandleMap& columnHandles,
    FileHandleFactory* fileHandleFactory,
    ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<HiveConfig>& hiveConfig,
    folly::Executor* executor)
    : fileHandleFactory_(fileHandleFactory),
      connectorQueryCtx_(connectorQueryCtx),
      hiveConfig_(hiveConfig),
      pool_(connectorQueryCtx->memoryPool()),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()),
      maxRowsPerIndexRequest_(hiveConfig_->maxRowsPerIndexRequest(
          connectorQueryCtx_->sessionProperties())),
      tableHandle_(std::move(tableHandle)),
      requestType_(requestType),
      outputType_(outputType),
      executor_(executor),
      ioStatistics_(std::make_shared<io::IoStatistics>()),
      ioStats_(std::make_shared<IoStats>()) {
  init(columnHandles, indexLookupConditions);
}

void HiveIndexSource::categorizeJoinConditions(
    const std::vector<core::IndexLookupConditionPtr>& joinConditions,
    const ColumnHandleMap& assignments) {
  const auto conditionMap =
      buildIndexLookupConditionMap(joinConditions, assignments);
  folly::F14FastSet<std::string> processedConditions;
  processIndexColumnConditions(conditionMap, processedConditions);
  processNonIndexColumnConditions(conditionMap, processedConditions);
  validateCategorization(joinConditions.size(), processedConditions.size());
}

HiveIndexSource::IndexLookupConditionMap
HiveIndexSource::buildIndexLookupConditionMap(
    const std::vector<core::IndexLookupConditionPtr>& joinConditions,
    const ColumnHandleMap& assignments) const {
  // The key name in each IndexLookupCondition references the probe input
  // column name; rewrite it to the table column name via 'assignments' so
  // downstream lookups can match against indexColumns()/partitionKeyHandles_.
  IndexLookupConditionMap conditionMap;
  for (const auto& condition : joinConditions) {
    auto convertedCondition = convertConditionKeyName(condition, assignments);
    const auto& columnName = convertedCondition->key->name();
    VELOX_USER_CHECK(
        conditionMap.emplace(columnName, std::move(convertedCondition)).second,
        "Duplicate lookup key found in indexLookupConditions: {}",
        columnName);
  }
  return conditionMap;
}

void HiveIndexSource::processIndexColumnConditions(
    const IndexLookupConditionMap& conditionMap,
    folly::F14FastSet<std::string>& processedConditions) {
  const auto& indexColumns = tableHandle_->indexColumns();
  indexLookupConditions_.reserve(indexColumns.size());

  // Process index columns in order, converting filters to index lookup
  // conditions where possible. A range filter/condition stops further
  // processing.
  for (const auto& indexColumn : indexColumns) {
    const common::Subfield subfield(indexColumn);
    const auto filterIt = filters_.find(subfield);
    const bool hasFilter = filterIt != filters_.end();
    const auto conditionIt = conditionMap.find(indexColumn);
    const bool hasIndexLookupCondition = conditionIt != conditionMap.end();

    // Cannot have both a filter and an index lookup condition on the same
    // column.
    VELOX_CHECK(
        !(hasFilter && hasIndexLookupCondition),
        "Cannot have both filter and index lookup condition on index column {}",
        indexColumn);

    if (!hasFilter && !hasIndexLookupCondition) {
      // No filter or index lookup condition on this column - stop processing.
      break;
    }

    if (hasIndexLookupCondition) {
      // Use the existing index lookup condition as-is.
      const auto& condition = conditionIt->second;
      VELOX_CHECK(!condition->isFilter());
      indexLookupConditions_.push_back(condition);
      processedConditions.insert(indexColumn);
      // Check if this is a range condition (Between) - stops further
      // processing.
      if (std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(
              condition)) {
        break;
      }
      continue;
    }

    if (!tryConvertFilterToIndexLookupCondition(indexColumn, filterIt)) {
      break;
    }
  }
}

bool HiveIndexSource::tryConvertFilterToIndexLookupCondition(
    const std::string& indexColumn,
    const common::SubfieldFilters::iterator& filterIt) {
  // Resolving the column type requires dataColumns; the condition-only path
  // in the caller doesn't need it, so this null check is scoped to here.
  const auto& dataColumns = tableHandle_->dataColumns();
  VELOX_CHECK_NOT_NULL(
      dataColumns,
      "dataColumns is required to convert filter to index lookup condition "
      "for index column {}",
      indexColumn);
  const auto typeIdx = dataColumns->getChildIdxIfExists(indexColumn);
  VELOX_CHECK(
      typeIdx.has_value(),
      "Index column {} not found in data columns",
      indexColumn);
  const auto& columnType = dataColumns->childAt(*typeIdx);
  const auto* filter = filterIt->second.get();

  // Try point lookup conversion first.
  if (auto pointValue = extractPointLookupValue(filter)) {
    indexLookupConditions_.push_back(createEqualConditionWithConstant(
        indexColumn, columnType, pointValue.value()));
    // Remove converted filter from filters_ map.
    filters_.erase(filterIt);
    return true;
  }

  // Try range conversion.
  if (auto rangeBounds = extractRangeBounds(filter)) {
    indexLookupConditions_.push_back(createBetweenConditionWithConstants(
        indexColumn, columnType, rangeBounds->first, rangeBounds->second));
    // Remove converted filter from filters_ map.
    filters_.erase(filterIt);
    // Range condition stops further processing.
    return false;
  }

  // Filter cannot be converted — leave it in filters_ for post-read
  // evaluation and stop walking.
  return false;
}

void HiveIndexSource::processNonIndexColumnConditions(
    const IndexLookupConditionMap& conditionMap,
    const folly::F14FastSet<std::string>& processedConditions) {
  // Categorize remaining conditions not consumed as index conditions:
  // - Conditions on index columns must have been processed in the prefix
  //   walk above; if any remain it's a prefix-gap violation.
  // - Conditions on partition columns become partition routing conditions.
  // - All other equi-conditions become post-read equality filters.
  const auto& indexColumns = tableHandle_->indexColumns();
  folly::F14FastSet<std::string_view> indexColumnSet(
      indexColumns.begin(), indexColumns.end());
  for (const auto& [columnName, condition] : conditionMap) {
    if (processedConditions.count(columnName) > 0) {
      continue;
    }
    VELOX_CHECK_EQ(
        indexColumnSet.count(columnName),
        0,
        "Unprocessed join condition on index column "
        "(conditions must follow index column order as a prefix): {}",
        columnName);

    auto equalCondition =
        std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(condition);
    VELOX_CHECK_NOT_NULL(
        equalCondition,
        "Non-index join condition must be an equal condition: {}",
        columnName);
    VELOX_CHECK(
        !equalCondition->isFilter(),
        "Non-index join condition cannot be a constant filter: {}",
        columnName);
    auto probeFieldAccess =
        checkedPointerCast<const core::FieldAccessTypedExpr>(
            equalCondition->value);

    auto partitionIt = partitionKeyHandles_.find(columnName);
    if (partitionIt != partitionKeyHandles_.end()) {
      const auto requestIdx =
          requestType_->getChildIdx(probeFieldAccess->name());
      partitionRoutingConditions_.push_back(
          {columnName,
           static_cast<column_index_t>(requestIdx),
           condition->key->type(),
           partitionIt->second->isPartitionDateValueDaysSinceEpoch()});
    } else {
      // Non-index conditions resolve to (readerOutput, request) indices
      // later in init() once readerOutputType_ is built.
      nonIndexConditionNames_.emplace_back(
          columnName, probeFieldAccess->name());
    }
  }
}

void HiveIndexSource::validateCategorization(
    size_t numInputConditions,
    size_t numIndexConditionsConsumed) const {
  // Every input condition must end up in exactly one bucket.
  // numIndexConditionsConsumed counts only join conditions consumed as
  // index lookup conditions; index lookup conditions converted from
  // pushdown filters are not counted, since those filters did not come
  // from joinConditions.
  VELOX_CHECK_EQ(
      numIndexConditionsConsumed + nonIndexConditionNames_.size() +
          partitionRoutingConditions_.size(),
      numInputConditions,
      "Not all join conditions were categorized");

  VELOX_CHECK(
      !indexLookupConditions_.empty(),
      "No index column join conditions found. At least one must be an "
      "index column");

  // Every partition column must have a corresponding routing condition.
  // Otherwise addSplits() takes the non-partitioned path, which builds a
  // scan spec without partition constants set; the reader then tries to
  // read the partition column from the file, where it does not exist.
  for (const auto& [partitionName, _] : partitionKeyHandles_) {
    auto found = std::any_of(
        partitionRoutingConditions_.begin(),
        partitionRoutingConditions_.end(),
        [&](const auto& cond) {
          return cond.partitionColumnName == partitionName;
        });
    VELOX_CHECK(
        found,
        "Partition column has no routing join condition: {}",
        partitionName);
  }
}

void HiveIndexSource::initRemainingFilter(
    std::vector<std::string>& readColumnNames,
    std::vector<TypePtr>& readColumnTypes) {
  VELOX_CHECK_NULL(remainingFilterExprSet_);
  double sampleRate = tableHandle_->sampleRate();
  auto remainingFilter = extractFiltersFromRemainingFilter(
      tableHandle_->remainingFilter(),
      expressionEvaluator_,
      filters_,
      sampleRate);
  // TODO: support sample rate later.
  VELOX_CHECK_EQ(
      sampleRate, 1, "Sample rate is not supported for index source");

  if (remainingFilter == nullptr) {
    return;
  }

  remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);

  const auto& remainingFilterExpr = remainingFilterExprSet_->expr(0);
  folly::F14FastMap<std::string, column_index_t> columnNames;
  for (int i = 0; i < readColumnNames.size(); ++i) {
    columnNames[readColumnNames[i]] = i;
  }
  for (const auto* input : remainingFilterExpr->distinctFields()) {
    auto it = columnNames.find(input->field());
    if (it != columnNames.end()) {
      if (shouldEagerlyMaterialize(*remainingFilterExpr, *input)) {
        remainingEagerlyLoadFields_.push_back(it->second);
      }
      continue;
    }
    // Remaining filter may reference columns that are not used otherwise,
    // e.g. are not being projected out and are not used in range filters.
    // Make sure to add these columns to readerOutputType_.
    readColumnNames.push_back(input->field());
    readColumnTypes.push_back(input->type());
  }

  remainingFilterSubfields_ = remainingFilterExpr->extractSubfields();
  for (auto& subfield : remainingFilterSubfields_) {
    const auto& name = getColumnName(subfield);
    auto it = projectedSubfields_.find(name);
    if (it != projectedSubfields_.end()) {
      // Some subfields of the column are already projected out, we append
      // the remainingFilter subfield
      it->second.push_back(&subfield);
    } else if (columnNames.count(name) == 0) {
      // remainingFilter subfield's column is not projected out, we add the
      // column and append the subfield
      projectedSubfields_[name].push_back(&subfield);
    }
  }
}

void HiveIndexSource::init(
    const ColumnHandleMap& assignments,
    const std::vector<core::IndexLookupConditionPtr>& indexLookupConditions) {
  VELOX_CHECK_NOT_NULL(tableHandle_);

  // Keyed by handle->name() (the actual table column name, not the query
  // alias) so that two assignments with different aliases pointing to the
  // same table column are detected as duplicates. The string_view is borrowed
  // from the handle, which the assignments map keeps alive for the duration
  // of init().
  folly::F14FastMap<std::string_view, const HiveColumnHandle*> columnHandles;
  for (const auto& [_, columnHandle] : assignments) {
    auto handle = checkedPointerCast<const HiveColumnHandle>(columnHandle);
    const auto [it, unique] =
        columnHandles.emplace(handle->name(), handle.get());
    VELOX_CHECK(unique, "Duplicate column handle for {}", handle->name());
    // Allow regular and partition key columns. Partition keys are not read
    // from files — their values are synthesized from split metadata.
    VELOX_CHECK(
        handle->columnType() == HiveColumnHandle::ColumnType::kRegular ||
            handle->columnType() == HiveColumnHandle::ColumnType::kPartitionKey,
        "Unsupported column type {} for column {}",
        HiveColumnHandle::columnTypeName(handle->columnType()),
        handle->name());
    if (handle->columnType() == HiveColumnHandle::ColumnType::kPartitionKey) {
      partitionKeyHandles_.emplace(handle->name(), handle);
    }
  }

  for (const auto& handle : tableHandle_->filterColumnHandles()) {
    auto it = columnHandles.find(handle->name());
    if (it != columnHandles.end()) {
      checkColumnHandleConsistent(*handle, *it->second);
      continue;
    }
    checkColumnHandleIsRegular(*handle);
  }

  std::vector<std::string> readColumnNames;
  auto readColumnTypes = outputType_->children();
  for (const auto& outputName : outputType_->names()) {
    auto it = assignments.find(outputName);
    VELOX_CHECK(
        it != assignments.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);

    auto handle = checkedPointerCast<const HiveColumnHandle>(it->second);
    readColumnNames.push_back(handle->name());
    if (handle->columnType() == HiveColumnHandle::ColumnType::kPartitionKey) {
      // Subfield projection / postProcessor checks below don't apply to
      // partition columns; their values come from split metadata.
      continue;
    }

    for (auto& subfield : handle->requiredSubfields()) {
      VELOX_USER_CHECK_EQ(
          getColumnName(subfield),
          handle->name(),
          "Required subfield does not match column name");
      projectedSubfields_[handle->name()].push_back(&subfield);
    }
    VELOX_CHECK_NULL(handle->postProcessor(), "Post processor not supported");
  }

  if (hiveConfig_->isFileColumnNamesReadAsLowerCase(
          connectorQueryCtx_->sessionProperties())) {
    checkColumnNameLowerCase(outputType_);
    checkColumnNameLowerCase(tableHandle_->subfieldFilters(), {});
    checkColumnNameLowerCase(tableHandle_->remainingFilter());
  }

  for (const auto& [subfield, filter] : tableHandle_->subfieldFilters()) {
    filters_.emplace(subfield.clone(), filter);
  }

  initRemainingFilter(readColumnNames, readColumnTypes);

  categorizeJoinConditions(indexLookupConditions, assignments);

  // Non-index condition columns may not be in the output projection, but we
  // still need the reader to materialize them so applyNonIndexConditions()
  // can compare their values against probe-side values at runtime. Use the
  // table-side type from the column handle (not the probe-side type from
  // requestType_) so that schema evolution or implicit casts on the probe
  // side don't cause silent type mis-reads.
  folly::F14FastSet<std::string> readColumnNameSet(
      readColumnNames.begin(), readColumnNames.end());
  for (const auto& [tableColumnName, _] : nonIndexConditionNames_) {
    if (readColumnNameSet.count(tableColumnName) == 0) {
      auto handleIt = columnHandles.find(tableColumnName);
      VELOX_CHECK(
          handleIt != columnHandles.end(),
          "Non-index condition column missing from assignments: {}",
          tableColumnName);
      readColumnNames.push_back(tableColumnName);
      readColumnTypes.push_back(handleIt->second->dataType());
      readColumnNameSet.insert(tableColumnName);
    }
  }

  readerOutputType_ =
      ROW(std::move(readColumnNames), std::move(readColumnTypes));

  // Resolve non-index condition column names to (readerOutput, request)
  // index pairs now that readerOutputType_ exists.
  nonIndexConditions_.reserve(nonIndexConditionNames_.size());
  for (const auto& [tableColumnName, probeColumnName] :
       nonIndexConditionNames_) {
    auto readerOutputIdx = readerOutputType_->getChildIdx(tableColumnName);
    auto requestIdx = requestType_->getChildIdx(probeColumnName);
    nonIndexConditions_.push_back(
        {static_cast<column_index_t>(readerOutputIdx),
         static_cast<column_index_t>(requestIdx)});
  }
  nonIndexConditionNames_.clear();
}

std::shared_ptr<common::ScanSpec> HiveIndexSource::buildScanSpec(
    const std::unordered_map<std::string, std::optional<std::string>>*
        partitionValues) {
  auto spec = makeScanSpec(
      readerOutputType_,
      projectedSubfields_,
      filters_,
      /*indexColumns=*/{},
      tableHandle_->dataColumns(),
      partitionKeyHandles_,
      /*infoColumns=*/{},
      /*specialColumns=*/{},
      hiveConfig_->readStatsBasedFilterReorderDisabled(
          connectorQueryCtx_->sessionProperties()),
      pool_);
  if (partitionValues == nullptr) {
    return spec;
  }
  // Set partition values as constants on the matching scan spec children.
  // The selective reader emits these constants for every row instead of
  // reading from the file (the partition column doesn't exist there).
  const bool readTimestampAsLocal =
      hiveConfig_->readTimestampPartitionValueAsLocalTime(
          connectorQueryCtx_->sessionProperties());
  // Static storage so the "absent" branch yields a reference with stable
  // address, avoiding a copy of the inner string when value is present.
  static const std::optional<std::string> kNull;
  for (const auto& [name, handle] : partitionKeyHandles_) {
    auto* childSpec = spec->childByName(name);
    if (childSpec == nullptr) {
      // Partition column not in output (only used for routing). Nothing
      // to project.
      continue;
    }
    auto it = partitionValues->find(name);
    const auto& value = it != partitionValues->end() ? it->second : kNull;
    childSpec->setConstantValue(newConstantFromString(
        handle->dataType(),
        value,
        pool_,
        readTimestampAsLocal,
        handle->isPartitionDateValueDaysSinceEpoch()));
  }
  return spec;
}

void HiveIndexSource::addSplits(
    std::vector<std::shared_ptr<ConnectorSplit>> splits) {
  VELOX_CHECK(
      readers_.empty() && partitionGroups_.empty(),
      "addSplits can only be called once for HiveIndexSource");

  std::vector<std::shared_ptr<const HiveConnectorSplit>> hiveSplits;
  hiveSplits.reserve(splits.size());
  for (auto& split : splits) {
    hiveSplits.push_back(checkedPointerCast<const HiveConnectorSplit>(split));
  }

  if (partitionRoutingConditions_.empty()) {
    // Non-partitioned path: a single shared scan spec for all readers. There
    // are no partition columns in scope (the completeness check in
    // categorizeJoinConditions guarantees this), so no per-split constants.
    createReadersFromSplits(std::move(hiveSplits), buildScanSpec(), readers_);
    return;
  }

  // Partitioned path: group splits by partition values, build a per-group
  // scan spec with partition constants set, and create readers per group.
  buildPartitionGroups(std::move(hiveSplits));
}

void HiveIndexSource::buildPartitionGroups(
    std::vector<std::shared_ptr<const HiveConnectorSplit>> hiveSplits) {
  auto splitGroups = groupSplitsByPartitionValues(std::move(hiveSplits));

  const bool readTimestampAsLocal =
      hiveConfig_->readTimestampPartitionValueAsLocalTime(
          connectorQueryCtx_->sessionProperties());

  // For each distinct partition, build typed routing constants (used by
  // findPartitionGroup() at lookup time) and a per-group scan spec where
  // partition columns are emitted as constants by the underlying reader.
  partitionGroups_.reserve(splitGroups.size());
  for (auto& [_, groupSplits] : splitGroups) {
    PartitionReaderGroup group;
    const auto& split = *groupSplits.front();

    group.routingConstants.reserve(partitionRoutingConditions_.size());
    static const std::optional<std::string> kNull;
    for (const auto& cond : partitionRoutingConditions_) {
      auto it = split.partitionKeys.find(cond.partitionColumnName);
      const std::optional<std::string>& value =
          it != split.partitionKeys.end() ? it->second : kNull;
      group.routingConstants.push_back(newConstantFromString(
          cond.type,
          value,
          pool_,
          readTimestampAsLocal,
          cond.isPartitionDateValueDaysSinceEpoch));
    }

    auto groupScanSpec = buildScanSpec(&split.partitionKeys);
    createReadersFromSplits(
        std::move(groupSplits), groupScanSpec, group.readers);
    partitionGroups_.push_back(std::move(group));
  }
}

folly::F14FastMap<
    std::string,
    std::vector<std::shared_ptr<const HiveConnectorSplit>>>
HiveIndexSource::groupSplitsByPartitionValues(
    std::vector<std::shared_ptr<const HiveConnectorSplit>> hiveSplits) const {
  // Look up a partition value in the split, returning a reference to the
  // entry when present and a reference to a shared empty optional
  // otherwise. Static storage gives the empty case a stable address so we
  // can return by reference and avoid copying the inner string.
  auto extractValue =
      [](const HiveConnectorSplit& split,
         const std::string& name) -> const std::optional<std::string>& {
    static const std::optional<std::string> kNull;
    auto it = split.partitionKeys.find(name);
    return it != split.partitionKeys.end() ? it->second : kNull;
  };

  // Encode a split's tuple of routing partition values into an opaque
  // grouping key. Uses '\0' as the field separator and '\1' as a presence
  // sentinel that distinguishes a null partition value from "":
  //   - '\0' is safe as a separator: Hive partition values come from
  //     filesystem directory names, which cannot contain '\0'.
  //   - '\1' is safe as a presence sentinel: a partition value beginning
  //     with '\1' would collide with an empty-string value, but Hive
  //     partition values are alphanumeric in practice (dates, ids,
  //     hashes), so this collision is theoretical. The DCHECK below
  //     catches accidental misuse in debug builds.
  auto makeGroupKey = [this, &extractValue](const HiveConnectorSplit& split) {
    std::string key;
    for (const auto& cond : partitionRoutingConditions_) {
      const auto& value = extractValue(split, cond.partitionColumnName);
      if (value.has_value()) {
        VELOX_DCHECK(
            value->find_first_of(std::string_view("\0\1", 2)) ==
                std::string::npos,
            "Hive partition value contains reserved byte ('\\0' or '\\1'): "
            "column={}, value={}",
            cond.partitionColumnName,
            *value);
        key += '\1';
        key += *value;
      }
      key += '\0';
    }
    return key;
  };

  folly::F14FastMap<
      std::string,
      std::vector<std::shared_ptr<const HiveConnectorSplit>>>
      splitGroups;
  for (auto& split : hiveSplits) {
    auto key = makeGroupKey(*split);
    splitGroups[std::move(key)].push_back(std::move(split));
  }
  return splitGroups;
}

void HiveIndexSource::createReadersFromSplits(
    std::vector<std::shared_ptr<const HiveConnectorSplit>> hiveSplits,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::vector<std::unique_ptr<SplitIndexReader>>& readers) {
  auto* registry = IndexReaderFactoryRegistry::getInstance();
  for (auto& split : hiveSplits) {
    const auto* factory = registry->getFactory(split->fileFormat);
    if (factory != nullptr) {
      createCustomIndexReader(*factory, std::move(split), readers);
    } else {
      VELOX_CHECK(
          split->fileFormat == dwio::common::FileFormat::NIMBLE ||
              split->fileFormat == dwio::common::FileFormat::SST,
          "No IndexReaderFactory registered for format: {}",
          dwio::common::toString(split->fileFormat));
      createFileIndexReader(std::move(split), scanSpec, readers);
    }
  }

  VELOX_CHECK(!readers.empty(), "No index readers created from splits");
}

HiveIndexSource::PartitionReaderGroup* FOLLY_NULLABLE
HiveIndexSource::findPartitionGroup(const RowVectorPtr& probeInput) {
  VELOX_CHECK_GT(probeInput->size(), 0);
  // Upstream pre-partitions the probe input so every row in a batch shares
  // the same partition column values. Routing only inspects row 0; in
  // debug builds, spot-check the last row to catch contract violations.
#ifndef NDEBUG
  if (probeInput->size() > 1) {
    const vector_size_t lastRow = probeInput->size() - 1;
    for (const auto& cond : partitionRoutingConditions_) {
      const auto& probeCol = probeInput->childAt(cond.requestIndex);
      VELOX_DCHECK(
          probeCol->equalValueAt(probeCol.get(), lastRow, 0),
          "Probe partition column is not constant across batch at column index {}",
          cond.requestIndex);
    }
  }
#endif
  // Linear scan: O(P × K) per probe batch where P is the number of
  // partition groups and K is the number of routing columns. P is
  // typically small (1 under partition-aware grouped execution; a handful
  // under standard grouped execution), so the linear scan with
  // branch-friendly equalValueAt comparisons beats a hash lookup that
  // would require re-encoding probe values into a key on every call.
  // Switch to an indexed lookup if P grows large.
  //
  // Null handling: equalValueAt treats null == null as true. This is
  // intentional here — a null partition value is a real partition
  // (Hive's __HIVE_DEFAULT_PARTITION__), not the SQL "unknown" used by
  // applyNonIndexConditions, which treats either-side null as not equal.
  for (auto& group : partitionGroups_) {
    bool match = true;
    for (size_t i = 0; i < partitionRoutingConditions_.size(); ++i) {
      const auto& cond = partitionRoutingConditions_[i];
      const auto& probeCol = probeInput->childAt(cond.requestIndex);
      if (!probeCol->equalValueAt(group.routingConstants[i].get(), 0, 0)) {
        match = false;
        break;
      }
    }
    if (match) {
      return &group;
    }
  }
  return nullptr;
}

std::shared_ptr<IndexSource::ResultIterator> HiveIndexSource::lookup(
    const Request& request) {
  auto options = SplitIndexReader::Options{
      .maxRowsPerRequest = static_cast<vector_size_t>(maxRowsPerIndexRequest_)};

  // Partitioned path: route probe rows to the matching partition group.
  if (!partitionGroups_.empty()) {
    auto* group = findPartitionGroup(request.input);
    if (group == nullptr) {
      // No group matched the probe's partition values. Return an empty
      // iterator (not nullptr) — IndexLookupJoin requires a non-null
      // iterator and handles the "no results" case as LEFT JOIN nulls.
      // Cached: the iterator has no state, so a single shared instance
      // serves every miss.
      static const std::shared_ptr<IndexSource::ResultIterator> kEmpty =
          std::make_shared<EmptyResultIterator>();
      return kEmpty;
    }
    return createLookupIterator(request, options, group->readers);
  }

  return createLookupIterator(request, options, readers_);
}

std::shared_ptr<IndexSource::ResultIterator>
HiveIndexSource::createLookupIterator(
    const Request& request,
    const SplitIndexReader::Options& options,
    std::vector<std::unique_ptr<SplitIndexReader>>& readers) {
  if (readers.size() == 1) {
    return std::make_shared<HiveLookupIterator>(
        shared_from_this(), readers[0].get(), request, options);
  }

  std::vector<std::shared_ptr<IndexSource::ResultIterator>> splitIters;
  splitIters.reserve(readers.size());
  for (auto& reader : readers) {
    splitIters.push_back(
        std::make_shared<HiveLookupIterator>(
            shared_from_this(), reader.get(), request, options));
  }
  return std::make_shared<UnionResultIterator>(
      std::move(splitIters), outputType_, pool_);
}

std::unordered_map<std::string, RuntimeMetric> HiveIndexSource::runtimeStats() {
  // Start with accumulated per-call timing stats.
  auto stats = runtimeStats_;

  if (remainingFilterTimeNs_ != 0) {
    stats[std::string(Connector::kTotalRemainingFilterTime)] =
        RuntimeMetric(remainingFilterTimeNs_, RuntimeCounter::Unit::kNanos);
  }

  auto mergeReaderStats =
      [&stats](const std::vector<std::unique_ptr<SplitIndexReader>>& readers) {
        for (auto& reader : readers) {
          for (auto& [key, metric] : reader->runtimeStats()) {
            auto it = stats.find(key);
            if (it != stats.end()) {
              it->second.merge(metric);
            } else {
              stats.emplace(key, metric);
            }
          }
        }
      };

  mergeReaderStats(readers_);

  for (auto& group : partitionGroups_) {
    mergeReaderStats(group.readers);
  }
  // Add I/O stats from ioStatistics_ (storage read, ram cache, ssd cache).
  if (ioStatistics_) {
    const auto& read = ioStatistics_->read();
    if (read.count() > 0) {
      stats[std::string(FileDataSource::kStorageReadBytes)] = RuntimeMetric(
          static_cast<int64_t>(read.sum()),
          read.count(),
          static_cast<int64_t>(read.min()),
          static_cast<int64_t>(read.max()),
          RuntimeCounter::Unit::kBytes);
    }
    const auto& ramHit = ioStatistics_->ramHit();
    if (ramHit.count() > 0) {
      stats[std::string(FileDataSource::kNumRamRead)] =
          RuntimeMetric(static_cast<int64_t>(ramHit.count()));
      stats[std::string(FileDataSource::kRamReadBytes)] = RuntimeMetric(
          static_cast<int64_t>(ramHit.sum()),
          ramHit.count(),
          static_cast<int64_t>(ramHit.min()),
          static_cast<int64_t>(ramHit.max()),
          RuntimeCounter::Unit::kBytes);
    }
    const auto& ssdRead = ioStatistics_->ssdRead();
    if (ssdRead.count() > 0) {
      stats[std::string(FileDataSource::kNumLocalRead)] =
          RuntimeMetric(static_cast<int64_t>(ssdRead.count()));
      stats[std::string(FileDataSource::kLocalReadBytes)] = RuntimeMetric(
          static_cast<int64_t>(ssdRead.sum()),
          ssdRead.count(),
          static_cast<int64_t>(ssdRead.min()),
          static_cast<int64_t>(ssdRead.max()),
          RuntimeCounter::Unit::kBytes);
    }
  }

  return stats;
}

vector_size_t HiveIndexSource::evaluateRemainingFilter(
    RowVectorPtr& rowVector) {
  remainingFilterRows_.resize(rowVector->size());
  for (auto fieldIndex : remainingEagerlyLoadFields_) {
    LazyVector::ensureLoadedRows(
        rowVector->childAt(fieldIndex),
        remainingFilterRows_,
        remainingFilterLazyDecoded_,
        remainingFilterLazyBaseRows_);
  }

  uint64_t filterTimeNs{0};
  vector_size_t numRemainingRows{0};
  {
    NanosecondTimer timer(&filterTimeNs);
    expressionEvaluator_->evaluate(
        remainingFilterExprSet_.get(),
        remainingFilterRows_,
        *rowVector,
        remainingFilterResult_);
    numRemainingRows = exec::processFilterResults(
        remainingFilterResult_,
        remainingFilterRows_,
        remainingFilterEvalCtx_,
        pool_);
  }
  remainingFilterTimeNs_ += filterTimeNs;
  return numRemainingRows;
}

void HiveIndexSource::recordIterationStats(
    const IterationStats& iterationStats) {
  const auto totalWallNs = iterationStats.setupWallNs +
      iterationStats.readWallNs + iterationStats.outputWallNs +
      iterationStats.filterWallNs;
  if (totalWallNs != 0) {
    exec::addOperatorRuntimeStats(
        IterationStats::kConnectorLookupWallNanos,
        RuntimeCounter(
            static_cast<int64_t>(totalWallNs), RuntimeCounter::Unit::kNanos),
        runtimeStats_);
  }
  if (iterationStats.setupWallNs != 0) {
    exec::addOperatorRuntimeStats(
        IterationStats::kIndexSetupWallNanos,
        RuntimeCounter(
            static_cast<int64_t>(iterationStats.setupWallNs),
            RuntimeCounter::Unit::kNanos),
        runtimeStats_);
  }
  if (iterationStats.setupCpuNs != 0) {
    exec::addOperatorRuntimeStats(
        IterationStats::kIndexSetupCpuNanos,
        RuntimeCounter(
            static_cast<int64_t>(iterationStats.setupCpuNs),
            RuntimeCounter::Unit::kNanos),
        runtimeStats_);
  }
  if (iterationStats.readWallNs != 0) {
    exec::addOperatorRuntimeStats(
        IterationStats::kIndexReadWallNanos,
        RuntimeCounter(
            static_cast<int64_t>(iterationStats.readWallNs),
            RuntimeCounter::Unit::kNanos),
        runtimeStats_);
  }
  if (iterationStats.readCpuNs != 0) {
    exec::addOperatorRuntimeStats(
        IterationStats::kIndexReadCpuNanos,
        RuntimeCounter(
            static_cast<int64_t>(iterationStats.readCpuNs),
            RuntimeCounter::Unit::kNanos),
        runtimeStats_);
  }
  if (iterationStats.outputCpuNs != 0) {
    exec::addOperatorRuntimeStats(
        IterationStats::kConnectorResultPrepareCpuNanos,
        RuntimeCounter(
            static_cast<int64_t>(iterationStats.outputCpuNs),
            RuntimeCounter::Unit::kNanos),
        runtimeStats_);
  }
  if (iterationStats.filterWallNs != 0) {
    exec::addOperatorRuntimeStats(
        IterationStats::kPostFilterWallNanos,
        RuntimeCounter(
            static_cast<int64_t>(iterationStats.filterWallNs),
            RuntimeCounter::Unit::kNanos),
        runtimeStats_);
  }
  if (iterationStats.filterCpuNs != 0) {
    exec::addOperatorRuntimeStats(
        IterationStats::kPostFilterCpuNanos,
        RuntimeCounter(
            static_cast<int64_t>(iterationStats.filterCpuNs),
            RuntimeCounter::Unit::kNanos),
        runtimeStats_);
  }
}

vector_size_t HiveIndexSource::applyNonIndexConditions(
    const RowVectorPtr& output,
    const RowVectorPtr& request,
    const BufferPtr& inputHits,
    BufferPtr& passingIndices) {
  const auto numRows = output->size();
  const auto* hits = inputHits->as<vector_size_t>();

  SelectivityVector passing(numRows, true);

  for (const auto& condition : nonIndexConditions_) {
    const auto& outputCol = output->childAt(condition.readerOutputIndex);
    const auto& requestCol = request->childAt(condition.requestIndex);
    for (vector_size_t row = 0; row < numRows; ++row) {
      if (!passing.isValid(row)) {
        continue;
      }
      const auto requestRow = hits[row];
      // Either null → not equal (standard SQL join semantics).
      if (outputCol->isNullAt(row) || requestCol->isNullAt(requestRow) ||
          !outputCol->equalValueAt(requestCol.get(), row, requestRow)) {
        passing.setValid(row, false);
      }
    }
  }
  passing.updateBounds();

  const auto numPassing = passing.countSelected();
  if (numPassing == numRows) {
    return numRows;
  }

  passingIndices = allocateIndices(numPassing, pool_);
  auto* rawIndices = passingIndices->asMutable<vector_size_t>();
  vector_size_t idx = 0;
  passing.applyToSelected([&](vector_size_t row) { rawIndices[idx++] = row; });
  VELOX_CHECK_EQ(idx, numPassing);
  return numPassing;
}

RowVectorPtr HiveIndexSource::projectOutput(
    vector_size_t numRows,
    const BufferPtr& remainingIndices,
    const RowVectorPtr& rowVector) {
  if (outputType_->size() == 0) {
    return exec::wrap(numRows, remainingIndices, rowVector);
  }

  std::vector<VectorPtr> outputColumns;
  outputColumns.reserve(outputType_->size());
  for (int i = 0; i < outputType_->size(); ++i) {
    auto& child = rowVector->childAt(i);
    if (remainingIndices) {
      // Disable dictionary values caching in expression eval so that we
      // don't need to reallocate the result for every batch.
      child->disableMemo();
    }
    auto column = exec::wrapChild(numRows, remainingIndices, child);
    outputColumns.push_back(std::move(column));
  }

  return std::make_shared<RowVector>(
      pool_, outputType_, BufferPtr(nullptr), numRows, outputColumns);
}

void HiveIndexSource::createCustomIndexReader(
    const IndexReaderFactory& factory,
    std::shared_ptr<const HiveConnectorSplit> split,
    std::vector<std::unique_ptr<SplitIndexReader>>& readers) {
  VELOX_CHECK_NOT_NULL(split);
  auto reader = factory(split, tableHandle_, connectorQueryCtx_);
  VELOX_CHECK_NOT_NULL(
      reader,
      "IndexReaderFactory returned null for format: {}",
      dwio::common::toString(split->fileFormat));
  readers.push_back(std::move(reader));
}

void HiveIndexSource::createFileIndexReader(
    std::shared_ptr<const HiveConnectorSplit> split,
    const std::shared_ptr<common::ScanSpec>& scanSpec,
    std::vector<std::unique_ptr<SplitIndexReader>>& readers) {
  VELOX_CHECK_NOT_NULL(split);
  readers.push_back(
      std::make_unique<FileIndexReader>(
          std::move(split),
          tableHandle_,
          connectorQueryCtx_,
          hiveConfig_,
          scanSpec,
          indexLookupConditions_,
          requestType_,
          readerOutputType_,
          ioStatistics_,
          ioStats_,
          fileHandleFactory_,
          executor_,
          maxRowsPerIndexRequest_));
}

} // namespace facebook::velox::connector::hive

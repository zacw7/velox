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
#include "velox/functions/prestosql/aggregates/ArrayAggAggregate.h"
#include "folly/container/small_vector.h"
#include "velox/exec/ContainerRowSerde.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/aggregates/ValueList.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"

namespace facebook::velox::aggregate::prestosql {
namespace {

struct ArrayAccumulator {
  ValueList elements;
};

struct SourceRange {
  VectorPtr vector;
  vector_size_t offset;
  vector_size_t size;
};

// Accumulator for clustered input.  We keep the ranges from different source
// vectors that should be collected in this group.
struct ClusteredAccumulator {
  folly::small_vector<SourceRange, 2> sources;
};

class ArrayAggAggregate : public exec::Aggregate {
 public:
  explicit ArrayAggAggregate(TypePtr resultType, bool ignoreNulls)
      : Aggregate(resultType), ignoreNulls_(ignoreNulls) {}

  int32_t accumulatorFixedWidthSize() const override {
    return clusteredInput_ ? sizeof(ClusteredAccumulator)
                           : sizeof(ArrayAccumulator);
  }

  bool accumulatorUsesExternalMemory() const override {
    return true;
  }

  bool isFixedSize() const override {
    return false;
  }

  bool supportsToIntermediate() const override {
    return true;
  }

  void toIntermediate(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      VectorPtr& result) const override {
    const auto& elements = args[0];

    const auto numRows = rows.size();

    // Convert input to a single-entry array.

    // Set nulls for rows not present in 'rows'.
    auto* pool = allocator_->pool();
    BufferPtr nulls = allocateNulls(numRows, pool);
    auto mutableNulls = nulls->asMutable<uint64_t>();
    memcpy(
        nulls->asMutable<uint64_t>(),
        rows.asRange().bits(),
        bits::nbytes(numRows));

    auto loadedElements = BaseVector::loadedVectorShared(elements);

    if (ignoreNulls_ && loadedElements->mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t row) {
        if (loadedElements->isNullAt(row)) {
          bits::setNull(mutableNulls, row);
        }
      });
    }

    // Set offsets to 0, 1, 2, 3...
    BufferPtr offsets = allocateOffsets(numRows, pool);
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    std::iota(rawOffsets, rawOffsets + numRows, 0);

    // Set sizes to 1.
    BufferPtr sizes = allocateSizes(numRows, pool);
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    std::fill(rawSizes, rawSizes + numRows, 1);

    result = std::make_shared<ArrayVector>(
        pool,
        ARRAY(elements->type()),
        nulls,
        numRows,
        offsets,
        sizes,
        loadedElements);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto* vector = (*result)->as<ArrayVector>();
    VELOX_CHECK_NOT_NULL(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);
    auto* resultOffsets =
        vector->mutableOffsets(numGroups)->asMutable<vector_size_t>();
    auto* resultSizes =
        vector->mutableSizes(numGroups)->asMutable<vector_size_t>();
    auto& elements = vector->elements();
    const auto numElements = countElements(groups, numGroups);
    elements->resize(numElements);

    vector_size_t arrayOffset = 0;
    if (clusteredInput_) {
      bool singleSource{true};
      VectorPtr* currentSource{nullptr};
      std::vector<BaseVector::CopyRange> ranges;
      for (int32_t i = 0; i < numGroups; ++i) {
        auto* accumulator = value<ClusteredAccumulator>(groups[i]);
        resultOffsets[i] = arrayOffset;
        vector_size_t arraySize = 0;
        for (auto& source : accumulator->sources) {
          if (currentSource && currentSource->get() != source.vector.get()) {
            elements->copyRanges(currentSource->get(), ranges);
            ranges.clear();
          }
          if (!currentSource || currentSource->get() != source.vector.get()) {
            singleSource = currentSource == nullptr;
            currentSource = &source.vector;
            ranges.push_back({source.offset, arrayOffset, source.size});
          } else if (
              ranges.back().sourceIndex + ranges.back().count ==
              source.offset) {
            ranges.back().count += source.size;
          } else {
            VELOX_DCHECK_LT(
                ranges.back().sourceIndex + ranges.back().count, source.offset);
            ranges.push_back({source.offset, arrayOffset, source.size});
          }
          arrayOffset += source.size;
          arraySize += source.size;
        }
        resultSizes[i] = arraySize;
        if (arraySize == 0) {
          vector->setNull(i, true);
        } else {
          clearNull(rawNulls, i);
        }
      }
      if (currentSource != nullptr) {
        VELOX_DCHECK(!ranges.empty());
        if (singleSource && ranges.size() == 1) {
          VELOX_CHECK_GE(currentSource->get()->size(), numElements);
          VELOX_CHECK_EQ(ranges[0].count, numElements);
          elements = currentSource->get()->slice(
              ranges[0].sourceIndex, ranges[0].count);
        } else {
          elements->copyRanges(currentSource->get(), ranges);
        }
      }
    } else {
      for (int32_t i = 0; i < numGroups; ++i) {
        auto& values = value<ArrayAccumulator>(groups[i])->elements;
        auto arraySize = values.size();
        if (arraySize) {
          clearNull(rawNulls, i);
          ValueListReader reader(values);
          for (auto index = 0; index < arraySize; ++index) {
            reader.next(*elements, arrayOffset + index);
          }
          resultOffsets[i] = arrayOffset;
          resultSizes[i] = arraySize;
          arrayOffset += arraySize;
        } else {
          vector->setNull(i, true);
        }
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    VELOX_CHECK(!clusteredInput_);
    decodedElements_.decode(*args[0], rows);
    rows.applyToSelected([&](vector_size_t row) {
      if (ignoreNulls_ && decodedElements_.isNullAt(row)) {
        return;
      }
      auto group = groups[row];
      auto tracker = trackRowSize(group);
      value<ArrayAccumulator>(group)->elements.appendValue(
          decodedElements_, row, allocator_);
    });
  }

  bool supportsAddRawClusteredInput() const override {
    return clusteredInput_;
  }

  void addRawClusteredInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      const folly::Range<const vector_size_t*>& groupBoundaries) override {
    VELOX_CHECK(clusteredInput_);
    decodedElements_.decode(*args[0]);
    vector_size_t groupStart = 0;
    auto forEachAccumulator = [&](auto func) {
      for (const vector_size_t groupEnd : groupBoundaries) {
        auto* accumulator = value<ClusteredAccumulator>(groups[groupEnd - 1]);
        func(groupEnd, accumulator);
        groupStart = groupEnd;
      }
    };
    if (rows.isAllSelected() &&
        (!ignoreNulls_ || !decodedElements_.mayHaveNulls())) {
      forEachAccumulator(
          [&](vector_size_t groupEnd, ClusteredAccumulator* accumulator) {
            accumulator->sources.push_back(
                {args[0], groupStart, groupEnd - groupStart});
          });
    } else {
      forEachAccumulator(
          [&](vector_size_t groupEnd, ClusteredAccumulator* accumulator) {
            for (auto i = groupStart; i < groupEnd; ++i) {
              if (!rows.isValid(i) ||
                  (ignoreNulls_ && decodedElements_.isNullAt(i))) {
                if (i > groupStart) {
                  accumulator->sources.push_back(
                      {args[0], groupStart, i - groupStart});
                }
                groupStart = i + 1;
              }
            }
            if (groupEnd > groupStart) {
              accumulator->sources.push_back(
                  {args[0], groupStart, groupEnd - groupStart});
            }
          });
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    auto arrayVector = decodedIntermediate_.base()->as<ArrayVector>();
    auto& elements = arrayVector->elements();
    rows.applyToSelected([&](vector_size_t row) {
      auto group = groups[row];
      auto decodedRow = decodedIntermediate_.index(row);
      auto tracker = trackRowSize(group);
      if (!decodedIntermediate_.isNullAt(row)) {
        value<ArrayAccumulator>(group)->elements.appendRange(
            elements,
            arrayVector->offsetAt(decodedRow),
            arrayVector->sizeAt(decodedRow),
            allocator_);
      }
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    VELOX_CHECK(!clusteredInput_);
    auto& values = value<ArrayAccumulator>(group)->elements;

    decodedElements_.decode(*args[0], rows);
    auto tracker = trackRowSize(group);
    rows.applyToSelected([&](vector_size_t row) {
      if (ignoreNulls_ && decodedElements_.isNullAt(row)) {
        return;
      }
      values.appendValue(decodedElements_, row, allocator_);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    VELOX_CHECK(!clusteredInput_);
    decodedIntermediate_.decode(*args[0], rows);
    auto arrayVector = decodedIntermediate_.base()->as<ArrayVector>();

    auto& values = value<ArrayAccumulator>(group)->elements;
    auto elements = arrayVector->elements();
    rows.applyToSelected([&](vector_size_t row) {
      if (!decodedIntermediate_.isNullAt(row)) {
        auto decodedRow = decodedIntermediate_.index(row);
        values.appendRange(
            elements,
            arrayVector->offsetAt(decodedRow),
            arrayVector->sizeAt(decodedRow),
            allocator_);
      }
    });
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto index : indices) {
      if (clusteredInput_) {
        new (groups[index] + offset_) ClusteredAccumulator();
      } else {
        new (groups[index] + offset_) ArrayAccumulator();
      }
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto group : groups) {
      if (isInitialized(group)) {
        if (clusteredInput_) {
          auto* accumulator = value<ClusteredAccumulator>(group);
          std::destroy_at(accumulator);
        } else {
          value<ArrayAccumulator>(group)->elements.free(allocator_);
        }
      }
    }
  }

 private:
  vector_size_t countElements(char** groups, int32_t numGroups) const {
    vector_size_t size = 0;
    if (clusteredInput_) {
      for (int32_t i = 0; i < numGroups; ++i) {
        auto* accumulator = value<ClusteredAccumulator>(groups[i]);
        for (auto& source : accumulator->sources) {
          size += source.size;
        }
      }
    } else {
      for (int32_t i = 0; i < numGroups; ++i) {
        size += value<ArrayAccumulator>(groups[i])->elements.size();
      }
    }
    return size;
  }

  // A boolean representing whether to ignore nulls when aggregating inputs.
  const bool ignoreNulls_;
  // Reusable instance of DecodedVector for decoding input vectors.
  DecodedVector decodedElements_;
  DecodedVector decodedIntermediate_;
};

} // namespace

void registerArrayAggAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("E")
          .returnType("array(E)")
          .intermediateType("array(E)")
          .argumentType("E")
          .build()};

  auto name = prefix + kArrayAgg;
  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(
            argTypes.size(), 1, "{} takes at most one argument", name);
        return std::make_unique<ArrayAggAggregate>(
            resultType, config.prestoArrayAggIgnoreNulls());
      },
      withCompanionFunctions,
      overwrite);
}

void registerInternalArrayAggAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("E")
          .returnType("array(E)")
          .intermediateType("array(E)")
          .argumentType("E")
          .build()};

  auto name = prefix + "$internal$array_agg";
  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(
            argTypes.size(), 1, "{} takes at most one argument", name);
        return std::make_unique<ArrayAggAggregate>(
            resultType, /*ignoreNulls*/ false);
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql

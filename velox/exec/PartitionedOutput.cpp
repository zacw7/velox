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

#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {
namespace {
std::unique_ptr<VectorSerde::Options> getVectorSerdeOptions(
    const core::QueryConfig& queryConfig,
    VectorSerde::Kind kind) {
  std::unique_ptr<VectorSerde::Options> options =
      kind == VectorSerde::Kind::kPresto
      ? std::make_unique<serializer::presto::PrestoVectorSerde::PrestoOptions>()
      : std::make_unique<VectorSerde::Options>();
  options->compressionKind =
      common::stringToCompressionKind(queryConfig.shuffleCompressionKind());
  options->minCompressionRatio = PartitionedOutput::minCompressionRatio();
  return options;
}
} // namespace

namespace detail {
Destination::Destination(
    const std::string& taskId,
    int destination,
    VectorSerde* serde,
    VectorSerde::Options* serdeOptions,
    memory::MemoryPool* pool,
    bool eagerFlush,
    std::function<void(uint64_t bytes, uint64_t rows)> recordEnqueued)
    : taskId_(taskId),
      destination_(destination),
      serde_(serde),
      serdeOptions_(serdeOptions),
      pool_(pool),
      eagerFlush_(eagerFlush),
      recordEnqueued_(std::move(recordEnqueued)),
      rows_(raw_vector<vector_size_t>(pool)) {
  setTargetSizePct();
}

BlockingReason Destination::advance(
    uint64_t maxBytes,
    const std::vector<vector_size_t>& sizes,
    const RowVectorPtr& output,
    const row::CompactRow* outputCompactRow,
    const row::UnsafeRowFast* outputUnsafeRow,
    OutputBufferManager& bufferManager,
    const std::function<void()>& bufferReleaseFn,
    bool* atEnd,
    ContinueFuture* future,
    Scratch& scratch) {
  VELOX_CHECK_LE(!!outputCompactRow + !!outputUnsafeRow, 1);
  if (rowIdx_ >= rows_.size()) {
    *atEnd = true;
    return BlockingReason::kNotBlocked;
  }

  const auto firstRow = rowIdx_;
  const uint32_t adjustedMaxBytes = (maxBytes * targetSizePct_) / 100;
  if (bytesInCurrent_ >= adjustedMaxBytes) {
    return flush(bufferManager, bufferReleaseFn, future);
  }

  // Collect rows to serialize.
  bool shouldFlush = false;
  while (rowIdx_ < rows_.size() && !shouldFlush) {
    bytesInCurrent_ += sizes[rows_[rowIdx_]];
    ++rowIdx_;
    ++rowsInCurrent_;
    shouldFlush =
        bytesInCurrent_ >= adjustedMaxBytes || rowsInCurrent_ >= targetNumRows_;
  }

  // Serialize
  if (current_ == nullptr) {
    current_ = std::make_unique<VectorStreamGroup>(pool_, serde_);
    const auto rowType = asRowType(output->type());
    current_->createStreamTree(rowType, rowsInCurrent_, serdeOptions_);
  }

  const auto rows = folly::Range(&rows_[firstRow], rowIdx_ - firstRow);
  if (serde_->kind() == VectorSerde::Kind::kCompactRow) {
    VELOX_CHECK_NOT_NULL(outputCompactRow);
    current_->append(*outputCompactRow, rows, sizes);
  } else if (serde_->kind() == VectorSerde::Kind::kUnsafeRow) {
    VELOX_CHECK_NOT_NULL(outputUnsafeRow);
    current_->append(*outputUnsafeRow, rows, sizes);
  } else {
    VELOX_CHECK_EQ(serde_->kind(), VectorSerde::Kind::kPresto);
    current_->append(output, rows, scratch);
  }

  // Update output state variable.
  if (rowIdx_ == rows_.size()) {
    *atEnd = true;
  }
  if (shouldFlush || (eagerFlush_ && rowsInCurrent_ > 0)) {
    return flush(bufferManager, bufferReleaseFn, future);
  }
  return BlockingReason::kNotBlocked;
}

BlockingReason Destination::flush(
    OutputBufferManager& bufferManager,
    const std::function<void()>& bufferReleaseFn,
    ContinueFuture* future) {
  if (!current_ || rowsInCurrent_ == 0) {
    return BlockingReason::kNotBlocked;
  }

  // Upper limit of message size with no columns.
  constexpr int32_t kMinMessageSize = 128;
  auto listener = bufferManager.newListener();
  IOBufOutputStream stream(
      *current_->pool(),
      listener.get(),
      std::max<int64_t>(kMinMessageSize, current_->size()));
  const int64_t flushedRows = rowsInCurrent_;

  current_->flush(&stream);
  current_->clear();

  const int64_t flushedBytes = stream.tellp();

  bytesInCurrent_ = 0;
  rowsInCurrent_ = 0;
  setTargetSizePct();

  bool blocked = bufferManager.enqueue(
      taskId_,
      destination_,
      std::make_unique<SerializedPage>(
          stream.getIOBuf(bufferReleaseFn), nullptr, flushedRows),
      future);

  recordEnqueued_(flushedBytes, flushedRows);

  return blocked ? BlockingReason::kWaitForConsumer
                 : BlockingReason::kNotBlocked;
}

void Destination::updateStats(Operator* op) {
  VELOX_CHECK(finished_);
  if (current_) {
    const auto serializerStats = current_->runtimeStats();
    auto lockedStats = op->stats().wlock();
    for (auto& pair : serializerStats) {
      lockedStats->addRuntimeStat(pair.first, pair.second);
    }
  }
}

} // namespace detail

PartitionedOutput::PartitionedOutput(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "PartitionedOutput"),
      keyChannels_(toChannels(planNode->inputType(), planNode->keys())),
      numDestinations_(planNode->numPartitions()),
      replicateNullsAndAny_(planNode->isReplicateNullsAndAny()),
      partitionFunction_(
          numDestinations_ == 1 ? nullptr
                                : planNode->partitionFunctionSpec().create(
                                      numDestinations_,
                                      /*localExchange=*/false)),
      outputChannels_(calculateOutputChannels(
          planNode->inputType(),
          planNode->outputType(),
          planNode->outputType())),
      bufferManager_(OutputBufferManager::getInstanceRef()),
      // NOTE: 'bufferReleaseFn_' holds a reference on the associated task to
      // prevent it from deleting while there are output buffers being accessed
      // out of the partitioned output buffer manager such as in Prestissimo,
      // the http server holds the buffers while sending the data response.
      bufferReleaseFn_([task = operatorCtx_->task()]() {}),
      maxBufferedBytes_(ctx->task->queryCtx()
                            ->queryConfig()
                            .maxPartitionedOutputBufferSize()),
      eagerFlush_(eagerFlush),
      serde_(getNamedVectorSerde(planNode->serdeKind())),
      serdeOptions_(getVectorSerdeOptions(
          operatorCtx_->driverCtx()->queryConfig(),
          planNode->serdeKind())) {
  if (!planNode->isPartitioned()) {
    VELOX_USER_CHECK_EQ(numDestinations_, 1);
  }
  if (numDestinations_ == 1) {
    VELOX_USER_CHECK(keyChannels_.empty());
    VELOX_USER_CHECK_NULL(partitionFunction_);
  }
}

void PartitionedOutput::initializeInput(RowVectorPtr input) {
  input_ = std::move(input);
  if (outputType_->size() == 0) {
    output_ = std::make_shared<RowVector>(
        input_->pool(),
        outputType_,
        nullptr /*nulls*/,
        input_->size(),
        std::vector<VectorPtr>{});
  } else if (outputChannels_.empty()) {
    output_ = input_;
  } else {
    std::vector<VectorPtr> outputColumns;
    outputColumns.reserve(outputChannels_.size());
    for (auto i : outputChannels_) {
      outputColumns.push_back(input_->childAt(i));
    }

    output_ = std::make_shared<RowVector>(
        input_->pool(),
        outputType_,
        nullptr /*nulls*/,
        input_->size(),
        outputColumns);
  }

  // Lazy load all the input columns.
  for (auto i = 0; i < output_->childrenSize(); ++i) {
    output_->childAt(i)->loadedVector();
  }

  if (serde_->kind() == VectorSerde::Kind::kCompactRow) {
    outputCompactRow_ = std::make_unique<row::CompactRow>(output_);
  } else if (serde_->kind() == VectorSerde::Kind::kUnsafeRow) {
    outputUnsafeRow_ = std::make_unique<row::UnsafeRowFast>(output_);
  }
}

void PartitionedOutput::initializeDestinations() {
  if (destinations_.empty()) {
    auto taskId = operatorCtx_->taskId();
    for (int i = 0; i < numDestinations_; ++i) {
      destinations_.push_back(std::make_unique<detail::Destination>(
          taskId,
          i,
          serde_,
          serdeOptions_.get(),
          pool(),
          eagerFlush_,
          [&](uint64_t bytes, uint64_t rows) {
            auto lockedStats = stats_.wlock();
            lockedStats->addOutputVector(bytes, rows);
          }));
    }
  }
}

void PartitionedOutput::initializeSizeBuffers() {
  const auto numInput = input_->size();
  if (numInput > rowSize_.size()) {
    rowSize_.resize(numInput);
    sizePointers_.resize(numInput);
    // Set all the size pointers since 'rowSize_' may have been reallocated.
    for (vector_size_t i = 0; i < numInput; ++i) {
      sizePointers_[i] = &rowSize_[i];
    }
  }
}

void PartitionedOutput::estimateRowSizes() {
  const auto numInput = input_->size();
  std::fill(rowSize_.begin(), rowSize_.end(), 0);
  raw_vector<vector_size_t> storage(pool());
  const auto numbers = iota(numInput, storage);
  const auto rows = folly::Range(numbers, numInput);
  if (serde_->kind() == VectorSerde::Kind::kCompactRow) {
    VELOX_CHECK_NOT_NULL(outputCompactRow_);
    serde_->estimateSerializedSize(
        outputCompactRow_.get(), rows, sizePointers_.data());
  } else if (serde_->kind() == VectorSerde::Kind::kUnsafeRow) {
    VELOX_CHECK_NOT_NULL(outputUnsafeRow_);
    serde_->estimateSerializedSize(
        outputUnsafeRow_.get(), rows, sizePointers_.data());
  } else {
    VELOX_CHECK_EQ(serde_->kind(), VectorSerde::Kind::kPresto);
    serde_->estimateSerializedSize(
        output_.get(), rows, sizePointers_.data(), scratch_);
  }
}

void PartitionedOutput::addInput(RowVectorPtr input) {
  initializeInput(std::move(input));
  initializeDestinations();
  initializeSizeBuffers();
  estimateRowSizes();

  for (auto& destination : destinations_) {
    destination->beginBatch();
  }

  auto numInput = input_->size();
  if (numDestinations_ == 1) {
    destinations_[0]->addRows(IndexRange{0, numInput});
  } else {
    auto singlePartition = partitionFunction_->partition(*input_, partitions_);
    if (replicateNullsAndAny_) {
      collectNullRows();

      vector_size_t start = 0;
      if (!replicatedAny_) {
        for (auto& destination : destinations_) {
          destination->addRow(0);
        }
        replicatedAny_ = true;
        // Make sure not to replicate first row twice.
        start = 1;
      }
      for (auto i = start; i < numInput; ++i) {
        if (nullRows_.isValid(i)) {
          for (auto& destination : destinations_) {
            destination->addRow(i);
          }
        } else {
          if (singlePartition.has_value()) {
            destinations_[singlePartition.value()]->addRow(i);
          } else {
            destinations_[partitions_[i]]->addRow(i);
          }
        }
      }
    } else {
      if (singlePartition.has_value()) {
        destinations_[singlePartition.value()]->addRows(
            IndexRange{0, numInput});
      } else {
        for (vector_size_t i = 0; i < numInput; ++i) {
          destinations_[partitions_[i]]->addRow(i);
        }
      }
    }
  }
}

void PartitionedOutput::collectNullRows() {
  auto size = input_->size();
  rows_.resize(size);
  rows_.setAll();

  nullRows_.resize(size);
  nullRows_.clearAll();

  decodedVectors_.resize(keyChannels_.size());

  for (size_t keyChannelIndex = 0; keyChannelIndex < keyChannels_.size();
       ++keyChannelIndex) {
    column_index_t keyChannel = keyChannels_[keyChannelIndex];
    // Skip constant channel.
    if (keyChannel == kConstantChannel) {
      continue;
    }
    auto& keyVector = input_->childAt(keyChannel);
    if (keyVector->mayHaveNulls()) {
      DecodedVector& decodedVector = decodedVectors_[keyChannelIndex];
      decodedVector.decode(*keyVector, rows_);
      if (auto* rawNulls = decodedVector.nulls(&rows_)) {
        bits::orWithNegatedBits(
            nullRows_.asMutableRange().bits(), rawNulls, 0, size);
      }
    }
  }
  nullRows_.updateBounds();
}

RowVectorPtr PartitionedOutput::getOutput() {
  if (finished_) {
    return nullptr;
  }

  blockingReason_ = BlockingReason::kNotBlocked;
  detail::Destination* blockedDestination = nullptr;
  auto bufferManager = bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager, "OutputBufferManager was already destructed");

  // Limit serialized pages to 1MB.
  static const uint64_t kMaxPageSize = 1 << 20;
  const uint64_t maxPageSize = std::max<uint64_t>(
      kMinDestinationSize,
      std::min<uint64_t>(kMaxPageSize, maxBufferedBytes_ / numDestinations_));

  bool workLeft;
  do {
    workLeft = false;
    for (auto& destination : destinations_) {
      bool atEnd = false;
      blockingReason_ = destination->advance(
          maxPageSize,
          rowSize_,
          output_,
          outputCompactRow_.get(),
          outputUnsafeRow_.get(),
          *bufferManager,
          bufferReleaseFn_,
          &atEnd,
          &future_,
          scratch_);
      if (blockingReason_ != BlockingReason::kNotBlocked) {
        blockedDestination = destination.get();
        workLeft = false;
        // We stop on first blocked. Adding data to unflushed targets
        // would be possible but could allocate memory. We wait for
        // free space in the outgoing queue.
        break;
      }
      if (!atEnd) {
        workLeft = true;
      }
    }
  } while (workLeft);

  if (blockedDestination) {
    // If we are going off-thread, we may as well make the output in
    // progress for other destinations available, unless it is too
    // small to be worth transfer.
    for (auto& destination : destinations_) {
      if (destination.get() == blockedDestination ||
          destination->serializedBytes() < kMinDestinationSize) {
        continue;
      }
      destination->flush(*bufferManager, bufferReleaseFn_, nullptr);
    }
    return nullptr;
  }
  // All of 'output_' is written into the destinations. We are finishing, hence
  // move all the destinations to the output queue. This will not grow memory
  // and hence does not need blocking.
  if (noMoreInput_) {
    for (auto& destination : destinations_) {
      if (destination->isFinished()) {
        continue;
      }
      destination->flush(*bufferManager, bufferReleaseFn_, nullptr);
      destination->setFinished();
      destination->updateStats(this);
    }

    bufferManager->noMoreData(operatorCtx_->task()->taskId());
    finished_ = true;
  }
  // The input is fully processed, drop the reference to allow reuse.
  input_ = nullptr;
  output_ = nullptr;
  outputCompactRow_.reset();
  outputUnsafeRow_.reset();
  return nullptr;
}

bool PartitionedOutput::isFinished() {
  return finished_;
}

void PartitionedOutput::close() {
  Operator::close();
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        Operator::kShuffleSerdeKind,
        RuntimeCounter(static_cast<int64_t>(serde_->kind())));
    lockedStats->addRuntimeStat(
        Operator::kShuffleCompressionKind,
        RuntimeCounter(static_cast<int64_t>(serdeOptions_->compressionKind)));
  }
  destinations_.clear();
}

} // namespace facebook::velox::exec

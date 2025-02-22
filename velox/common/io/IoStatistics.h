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

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <folly/dynamic.h>
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::velox::io {

struct OperationCounters {
  uint64_t resourceThrottleCount{0};
  uint64_t localThrottleCount{0};
  uint64_t networkThrottleCount{0};
  uint64_t globalThrottleCount{0};
  uint64_t fullThrottleCount{0};
  uint64_t partialThrottleCount{0};
  uint64_t retryCount{0};
  uint64_t latencyInMs{0};
  uint64_t requestCount{0};
  uint64_t delayInjectedInSecs{0};

  void merge(const OperationCounters& other);
};

class IoCounter {
 public:
  uint64_t count() const {
    return count_;
  }

  uint64_t sum() const {
    return sum_;
  }

  uint64_t min() const {
    return min_;
  }

  uint64_t max() const {
    return max_;
  }

  void increment(uint64_t amount) {
    ++count_;
    sum_ += amount;
    casLoop(min_, amount, std::greater());
    casLoop(max_, amount, std::less());
  }

  void merge(const IoCounter& other) {
    sum_ += other.sum_;
    count_ += other.count_;
    casLoop(min_, other.min_, std::greater());
    casLoop(max_, other.max_, std::less());
  }

 private:
  template <typename Compare>
  static void
  casLoop(std::atomic<uint64_t>& value, uint64_t newValue, Compare compare) {
    uint64_t old = value;
    while (compare(old, newValue) &&
           !value.compare_exchange_weak(old, newValue)) {
    }
  }

  std::atomic<uint64_t> count_{0};
  std::atomic<uint64_t> sum_{0};
  std::atomic<uint64_t> min_{std::numeric_limits<uint64_t>::max()};
  std::atomic<uint64_t> max_{0};
};

class IoStatistics {
 public:
  uint64_t rawBytesRead() const;
  uint64_t rawOverreadBytes() const;
  uint64_t rawBytesWritten() const;
  uint64_t inputBatchSize() const;
  uint64_t outputBatchSize() const;
  uint64_t totalScanTime() const;
  uint64_t writeIOTimeUs() const;

  uint64_t incRawBytesRead(int64_t);
  uint64_t incRawOverreadBytes(int64_t);
  uint64_t incRawBytesWritten(int64_t);
  uint64_t incInputBatchSize(int64_t);
  uint64_t incOutputBatchSize(int64_t);
  uint64_t incTotalScanTime(int64_t);
  uint64_t incWriteIOTimeUs(int64_t);

  IoCounter& prefetch() {
    return prefetch_;
  }

  IoCounter& read() {
    return read_;
  }

  IoCounter& ssdRead() {
    return ssdRead_;
  }

  IoCounter& ramHit() {
    return ramHit_;
  }

  IoCounter& queryThreadIoLatency() {
    return queryThreadIoLatency_;
  }

  void incOperationCounters(
      const std::string& operation,
      const uint64_t resourceThrottleCount,
      const uint64_t localThrottleCount,
      const uint64_t networkThrottleCount,
      const uint64_t globalThrottleCount,
      const uint64_t retryCount,
      const uint64_t latencyInMs,
      const uint64_t delayInjectedInSecs,
      const uint64_t fullThrottleCount = 0,
      const uint64_t partialThrottleCount = 0);

  std::unordered_map<std::string, OperationCounters> operationStats() const;
  std::unordered_map<std::string, RuntimeMetric> storageStats() const;

  void addStorageStats(const std::string& name, const RuntimeCounter& counter);

  void merge(const IoStatistics& other);

  folly::dynamic getOperationStatsSnapshot() const;

 private:
  std::atomic<uint64_t> rawBytesRead_{0};
  std::atomic<uint64_t> rawBytesWritten_{0};
  std::atomic<uint64_t> inputBatchSize_{0};
  std::atomic<uint64_t> outputBatchSize_{0};
  std::atomic<uint64_t> rawOverreadBytes_{0};
  std::atomic<uint64_t> totalScanTime_{0};
  std::atomic<uint64_t> writeIOTimeUs_{0};

  // Planned read from storage or SSD.
  IoCounter prefetch_;

  // Read from storage, for sparsely accessed columns.
  IoCounter read_;

  // Hits from RAM cache. Does not include first use of prefetched data.
  IoCounter ramHit_;

  // Read from SSD cache instead of storage. Includes both random and planned
  // reads.
  IoCounter ssdRead_;

  // Time spent by a query processing thread waiting for synchronously issued IO
  // or for an in-progress read-ahead to finish.
  IoCounter queryThreadIoLatency_;

  std::unordered_map<std::string, OperationCounters> operationStats_;
  std::unordered_map<std::string, RuntimeMetric> storageStats_;
  mutable std::mutex operationStatsMutex_;
  mutable std::mutex storageStatsMutex_;
};

} // namespace facebook::velox::io

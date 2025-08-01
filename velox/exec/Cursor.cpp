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
#include "velox/exec/Cursor.h"
#include "velox/common/file/FileSystems.h"

#include <filesystem>

namespace facebook::velox::exec {

bool waitForTaskDriversToFinish(exec::Task* task, uint64_t maxWaitMicros) {
  VELOX_USER_CHECK(!task->isRunning());
  uint64_t waitMicros = 0;
  while ((task->numFinishedDrivers() != task->numTotalDrivers()) &&
         (waitMicros < maxWaitMicros)) {
    const uint64_t kWaitMicros = 1000;
    std::this_thread::sleep_for(std::chrono::microseconds(kWaitMicros));
    waitMicros += kWaitMicros;
  }

  if (task->numFinishedDrivers() != task->numTotalDrivers()) {
    LOG(ERROR) << "Timed out waiting for all drivers of task " << task->taskId()
               << " to finish. Finished drivers: " << task->numFinishedDrivers()
               << ". Total drivers: " << task->numTotalDrivers();
  }

  return task->numFinishedDrivers() == task->numTotalDrivers();
}

exec::BlockingReason TaskQueue::enqueue(
    RowVectorPtr vector,
    velox::ContinueFuture* future) {
  if (!vector) {
    std::lock_guard<std::mutex> l(mutex_);
    ++producersFinished_;
    if (consumerBlocked_) {
      consumerBlocked_ = false;
      consumerPromise_.setValue();
    }
    return exec::BlockingReason::kNotBlocked;
  }

  auto bytes = vector->retainedSize();
  TaskQueueEntry entry{std::move(vector), bytes};

  std::lock_guard<std::mutex> l(mutex_);
  // Check inside 'mutex_'
  if (closed_) {
    throw std::runtime_error("Consumer cursor is closed");
  }
  queue_.push_back(std::move(entry));
  totalBytes_ += bytes;
  if (consumerBlocked_) {
    consumerBlocked_ = false;
    consumerPromise_.setValue();
  }
  if (totalBytes_ > maxBytes_) {
    auto [unblockPromise, unblockFuture] = makeVeloxContinuePromiseContract();
    producerUnblockPromises_.emplace_back(std::move(unblockPromise));
    *future = std::move(unblockFuture);
    return exec::BlockingReason::kWaitForConsumer;
  }
  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr TaskQueue::dequeue() {
  for (;;) {
    RowVectorPtr vector;
    std::vector<ContinuePromise> mayContinue;
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (closed_) {
        return nullptr;
      }

      if (!queue_.empty()) {
        auto result = std::move(queue_.front());
        queue_.pop_front();
        totalBytes_ -= result.bytes;
        vector = std::move(result.vector);
        if (totalBytes_ < maxBytes_ / 2) {
          mayContinue = std::move(producerUnblockPromises_);
        }
      } else if (
          numProducers_.has_value() && producersFinished_ == numProducers_) {
        return nullptr;
      }
      if (!vector) {
        consumerBlocked_ = true;
        consumerPromise_ = ContinuePromise();
        consumerFuture_ = consumerPromise_.getFuture();
      }
    }
    // outside of 'mutex_'
    for (auto& promise : mayContinue) {
      promise.setValue();
    }
    if (vector) {
      return vector;
    }
    consumerFuture_.wait();
  }
}

void TaskQueue::close() {
  std::lock_guard<std::mutex> l(mutex_);
  closed_ = true;
  // Unblock producers.
  for (auto& promise : producerUnblockPromises_) {
    promise.setValue();
  }
  producerUnblockPromises_.clear();

  // Unblock consumers.
  if (consumerBlocked_) {
    consumerBlocked_ = false;
    consumerPromise_.setValue();
  }
}

bool TaskQueue::hasNext() {
  std::lock_guard<std::mutex> l(mutex_);
  return !queue_.empty();
}

class TaskCursorBase : public TaskCursor {
 public:
  TaskCursorBase(
      const CursorParameters& params,
      const std::shared_ptr<folly::Executor>& executor) {
    static std::atomic<int32_t> cursorId;
    taskId_ = fmt::format("test_cursor_{}", ++cursorId);

    if (params.queryCtx) {
      queryCtx_ = params.queryCtx;
    } else {
      // NOTE: the destructor of 'executor_' will wait for all the async task
      // activities to finish on TaskCursor destruction.
      executor_ = executor;
      static std::atomic<uint64_t> cursorQueryId{0};
      queryCtx_ = core::QueryCtx::create(
          executor_.get(),
          core::QueryConfig({}),
          std::
              unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
          cache::AsyncDataCache::getInstance(),
          nullptr,
          nullptr,
          fmt::format("TaskCursorQuery_{}", cursorQueryId++));
    }

    if (!params.queryConfigs.empty()) {
      auto configCopy = params.queryConfigs;
      queryCtx_->testingOverrideConfigUnsafe(std::move(configCopy));
    }

    planFragment_ = {
        params.planNode,
        params.executionStrategy,
        params.numSplitGroups,
        params.groupedExecutionLeafNodeIds};

    if (!params.spillDirectory.empty()) {
      taskSpillDirectory_ = params.spillDirectory + "/" + taskId_;
      auto fileSystem =
          velox::filesystems::getFileSystem(taskSpillDirectory_, nullptr);
      VELOX_CHECK_NOT_NULL(fileSystem, "File System is null!");
      try {
        fileSystem->mkdir(taskSpillDirectory_);
      } catch (...) {
        LOG(ERROR) << "Faield to create task spill directory "
                   << taskSpillDirectory_ << " base director "
                   << params.spillDirectory << " exists["
                   << std::filesystem::exists(taskSpillDirectory_) << "]";

        std::rethrow_exception(std::current_exception());
      }

      LOG(INFO) << "Task spill directory[" << taskSpillDirectory_
                << "] created";
    }
  }

 protected:
  std::string taskId_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  core::PlanFragment planFragment_;
  std::string taskSpillDirectory_;

 private:
  std::shared_ptr<folly::Executor> executor_;
};

class MultiThreadedTaskCursor : public TaskCursorBase {
 public:
  explicit MultiThreadedTaskCursor(const CursorParameters& params)
      : TaskCursorBase(
            params,
            std::make_shared<folly::CPUThreadPoolExecutor>(
                std::thread::hardware_concurrency())),
        maxDrivers_{params.maxDrivers},
        numConcurrentSplitGroups_{params.numConcurrentSplitGroups},
        numSplitGroups_{params.numSplitGroups} {
    VELOX_CHECK(!params.serialExecution);
    VELOX_CHECK(
        queryCtx_->isExecutorSupplied(),
        "Executor should be set in parallel task cursor");

    queue_ =
        std::make_shared<TaskQueue>(params.bufferedBytes, params.outputPool);

    // Captured as a shared_ptr by the consumer callback of task_.
    auto queue = queue_;
    task_ = Task::create(
        taskId_,
        std::move(planFragment_),
        params.destination,
        std::move(queryCtx_),
        Task::ExecutionMode::kParallel,
        // consumer
        [queue, copyResult = params.copyResult](
            const RowVectorPtr& vector,
            bool drained,
            velox::ContinueFuture* future) {
          VELOX_CHECK(
              !drained, "Unexpected drain in multithreaded task cursor");
          if (!vector || !copyResult) {
            return queue->enqueue(vector, future);
          }
          // Make sure to load lazy vector if not loaded already.
          for (auto& child : vector->children()) {
            child->loadedVector();
          }
          auto copy = BaseVector::create<RowVector>(
              vector->type(), vector->size(), queue->pool());
          copy->copy(vector.get(), 0, 0, vector->size());
          return queue->enqueue(std::move(copy), future);
        },
        0,
        [queue](std::exception_ptr) {
          // onError close the queue to unblock producers and consumers.
          // moveNext will handle rethrowing the error once it's
          // unblocked.
          queue->close();
        });

    if (!taskSpillDirectory_.empty()) {
      task_->setSpillDirectory(taskSpillDirectory_);
    }
  }

  ~MultiThreadedTaskCursor() override {
    queue_->close();
    if (task_ && !atEnd_) {
      task_->requestCancel();
    }
  }

  /// Starts the task if not started yet.
  void start() override {
    if (!started_) {
      started_ = true;
      try {
        task_->start(maxDrivers_, numConcurrentSplitGroups_);
        queue_->setNumProducers(numSplitGroups_ * task_->numOutputDrivers());
      } catch (const VeloxException& e) {
        // Could not find output pipeline, due to Task terminated before
        // start. Do not override the error.
        if (e.message().find("Output pipeline not found for task") ==
            std::string::npos) {
          throw;
        }
      }
    }
  }

  /// Fetches another batch from the task queue.
  /// Starts the task if not started yet.
  bool moveNext() override {
    start();
    if (error_) {
      std::rethrow_exception(error_);
    }

    // Task might be aborted before start.
    checkTaskError();
    current_ = queue_->dequeue();

    checkTaskError();
    if (!current_) {
      atEnd_ = true;
    }
    return current_ != nullptr;
  }

  void setNoMoreSplits() override {
    VELOX_CHECK(!noMoreSplits_);
    noMoreSplits_ = true;
  }

  bool noMoreSplits() const override {
    return noMoreSplits_;
  }

  bool hasNext() override {
    return queue_->hasNext();
  }

  RowVectorPtr& current() override {
    return current_;
  }

  void setError(std::exception_ptr error) override {
    error_ = error;
    if (task_) {
      task_->setError(error);
    }
  }

  const std::shared_ptr<Task>& task() override {
    return task_;
  }

 private:
  void checkTaskError() {
    if (!task_->error()) {
      return;
    }
    // Wait for the task to finish (there's' a small period of time between
    // when the error is set on the Task and terminate is called).
    task_->taskCompletionFuture()
        .within(std::chrono::microseconds(5'000'000))
        .wait();

    // Wait for all task drivers to finish to avoid destroying the executor_
    // before task_ finished using it and causing a crash.
    waitForTaskDriversToFinish(task_.get());
    std::rethrow_exception(task_->error());
  }

  const int32_t maxDrivers_;
  const int32_t numConcurrentSplitGroups_;
  const int32_t numSplitGroups_;

  bool started_{false};
  std::shared_ptr<TaskQueue> queue_;
  std::shared_ptr<exec::Task> task_;
  RowVectorPtr current_;
  bool atEnd_{false};
  tsan_atomic<bool> noMoreSplits_{false};
  std::exception_ptr error_;
};

class SingleThreadedTaskCursor : public TaskCursorBase {
 public:
  explicit SingleThreadedTaskCursor(const CursorParameters& params)
      : TaskCursorBase(params, nullptr) {
    VELOX_CHECK(params.serialExecution);
    VELOX_CHECK(
        !queryCtx_->isExecutorSupplied(),
        "Executor should not be set in serial task cursor");

    task_ = Task::create(
        taskId_,
        std::move(planFragment_),
        params.destination,
        std::move(queryCtx_),
        Task::ExecutionMode::kSerial);

    if (!taskSpillDirectory_.empty()) {
      task_->setSpillDirectory(taskSpillDirectory_);
    }

    VELOX_CHECK(
        task_->supportSerialExecutionMode(),
        "Plan doesn't support serial execution mode");
  }

  ~SingleThreadedTaskCursor() override {
    if (task_ && !SingleThreadedTaskCursor::hasNext()) {
      task_->requestCancel().wait();
    }
  }

  void start() override {
    // no-op
  }

  void setNoMoreSplits() override {
    VELOX_CHECK(!noMoreSplits_);
    noMoreSplits_ = true;
  }

  bool noMoreSplits() const override {
    return noMoreSplits_;
  }

  bool moveNext() override {
    if (!hasNext()) {
      return false;
    }
    current_ = next_;
    next_ = nullptr;
    return true;
  };

  bool hasNext() override {
    if (next_) {
      return true;
    }
    if (!task_->isRunning()) {
      return false;
    }
    while (true) {
      ContinueFuture future = ContinueFuture::makeEmpty();
      RowVectorPtr next = task_->next(&future);
      if (next != nullptr) {
        next_ = next;
        return true;
      }
      // When next is returned from task as a null pointer.
      if (!future.valid()) {
        VELOX_CHECK(!task_->isRunning() || !noMoreSplits_);
        return false;
      }
      // Task is blocked for some reason. Wait and try again.
      VELOX_CHECK_NULL(next);
      future.wait();
    }
  };

  RowVectorPtr& current() override {
    return current_;
  }

  void setError(std::exception_ptr error) override {
    error_ = error;
    if (task_) {
      task_->setError(error);
    }
  }

  const std::shared_ptr<Task>& task() override {
    return task_;
  }

 private:
  std::shared_ptr<exec::Task> task_;
  bool noMoreSplits_{false};
  RowVectorPtr current_;
  RowVectorPtr next_;
  std::exception_ptr error_;
};

std::unique_ptr<TaskCursor> TaskCursor::create(const CursorParameters& params) {
  if (params.serialExecution) {
    return std::make_unique<SingleThreadedTaskCursor>(params);
  }
  return std::make_unique<MultiThreadedTaskCursor>(params);
}

bool RowCursor::next() {
  if (++currentRow_ < numRows_) {
    return true;
  }
  if (!cursor_->moveNext()) {
    return false;
  }
  auto vector = cursor_->current();
  numRows_ = vector->size();
  if (!numRows_) {
    return next();
  }
  currentRow_ = 0;
  if (decoded_.empty()) {
    decoded_.resize(vector->childrenSize());
    for (int32_t i = 0; i < vector->childrenSize(); ++i) {
      decoded_[i] = std::make_unique<DecodedVector>();
    }
  }
  allRows_.resize(vector->size());
  allRows_.setAll();
  for (int32_t i = 0; i < decoded_.size(); ++i) {
    decoded_[i]->decode(*vector->childAt(i), allRows_);
  }
  return true;
}

bool RowCursor::hasNext() {
  return currentRow_ < numRows_ || cursor_->hasNext();
}

} // namespace facebook::velox::exec

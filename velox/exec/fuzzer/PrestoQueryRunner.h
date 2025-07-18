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

#include <folly/SocketAddress.h>
#include <folly/io/async/EventBaseThread.h>
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec::test {

template <typename T>
T extractSingleValue(const std::vector<RowVectorPtr>& data) {
  auto simpleVector = data[0]->childAt(0)->as<SimpleVector<T>>();
  VELOX_CHECK(!simpleVector->isNullAt(0));
  return simpleVector->valueAt(0);
}

class QueryRunnerContext {
 public:
  std::unordered_map<core::PlanNodeId, std::vector<std::string>> windowFrames_;
};

class PrestoQueryRunner : public velox::exec::test::ReferenceQueryRunner {
 public:
  /// @param coordinatorUri Presto REST API endpoint, e.g. http://127.0.0.1:8080
  /// @param user Username to use in X-Presto-User header.
  /// @param timeout Timeout in milliseconds of an HTTP request.
  PrestoQueryRunner(
      memory::MemoryPool* aggregatePool,
      std::string coordinatorUri,
      std::string user,
      std::chrono::milliseconds timeout);

  RunnerType runnerType() const override {
    return RunnerType::kPrestoQueryRunner;
  }

  const std::vector<TypePtr>& supportedScalarTypes() const override;

  static bool isSupportedDwrfType(const TypePtr& type);

  std::pair<std::vector<RowVectorPtr>, std::vector<core::ExprPtr>>
  inputProjections(const std::vector<RowVectorPtr>& input) const override;

  const std::unordered_map<std::string, DataSpec>&
  aggregationFunctionDataSpecs() const override;

  /// Converts Velox query plan to Presto SQL. Supports Values -> Aggregation or
  /// Window with an optional Project on top.
  ///
  /// Values node is converted into reading from 'tmp' table.
  ///
  /// @return std::nullopt if Values node uses types not supported by DWRF file
  /// format (DATE, INTERVAL, UNKNOWN).
  std::optional<std::string> toSql(
      const velox::core::PlanNodePtr& plan) override;

  bool isConstantExprSupported(const core::TypedExprPtr& expr) override;

  bool isSupported(const exec::FunctionSignature& signature) override;

  // Converts 'plan' into an SQL query and executes it. Result is returned as a
  // MaterializedRowMultiset with the ReferenceQueryErrorCode::kSuccess if
  // successful, or an std::nullopt with a ReferenceQueryErrorCode if the query
  // fails.
  std::pair<
      std::optional<std::multiset<std::vector<velox::variant>>>,
      ReferenceQueryErrorCode>
  execute(const core::PlanNodePtr& plan) override;

  /// Similar to 'execute' but returns results in RowVector format.
  /// Caller should ensure 'supportsVeloxVectorResults' returns true.
  std::pair<
      std::optional<std::vector<velox::RowVectorPtr>>,
      ReferenceQueryErrorCode>
  executeAndReturnVector(const core::PlanNodePtr& plan) override;

  /// Executes Presto SQL query and returns the results. Tables referenced by
  /// the query must already exist.
  std::vector<velox::RowVectorPtr> execute(const std::string& sql) override;

  /// Executes Presto SQL query with extra presto session property.
  std::vector<velox::RowVectorPtr> execute(
      const std::string& sql,
      const std::string& sessionProperty) override;

  bool supportsVeloxVectorResults() const override;

  std::shared_ptr<QueryRunnerContext> queryRunnerContext() {
    return queryRunnerContext_;
  }

 private:
  using ReferenceQueryRunner::toSql;

  memory::MemoryPool* pool() {
    return pool_.get();
  }

  std::string startQuery(
      const std::string& sql,
      const std::string& sessionProperty = "");

  std::string fetchNext(const std::string& nextUri);

  // Creates an empty table with given data type and table name. The function
  // returns the root directory of table files.
  std::string createTable(const std::string& name, const TypePtr& type);
  void cleanUp(const std::string& name);

  const std::string coordinatorUri_;
  const std::string user_;
  const std::chrono::milliseconds timeout_;
  folly::EventBaseThread eventBaseThread_{false};
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<QueryRunnerContext> queryRunnerContext_;
};

} // namespace facebook::velox::exec::test

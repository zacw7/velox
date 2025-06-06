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

#include <optional>
#include <set>
#include <unordered_map>

#include "velox/exec/fuzzer/ReferenceQueryRunner.h"

namespace facebook::velox::exec::test {

class DuckQueryRunner : public ReferenceQueryRunner {
 public:
  explicit DuckQueryRunner(memory::MemoryPool* aggregatePool);

  RunnerType runnerType() const override {
    return RunnerType::kDuckQueryRunner;
  }

  /// Skip Timestamp, Varbinary, Unknown, and IntervalDayTime types. DuckDB
  /// doesn't support nanosecond precision for timestamps or casting from Bigint
  /// to Interval.
  ///
  /// TODO Investigate mismatches reported when comparing Varbinary.
  const std::vector<TypePtr>& supportedScalarTypes() const override;

  const std::unordered_map<std::string, DataSpec>&
  aggregationFunctionDataSpecs() const override;

  /// Specify names of aggregate function to exclude from the list of supported
  /// functions. Used to exclude functions that are non-determonistic, have bugs
  /// or whose semantics differ from Velox.
  void disableAggregateFunctions(const std::vector<std::string>& names);

  /// Supports AggregationNode and WindowNode with optional ProjectNode on top.
  /// Assumes that source of AggregationNode or Window Node is 'tmp' table.
  std::optional<std::string> toSql(const core::PlanNodePtr& plan) override;

  // Converts 'plan' into an SQL query and executes it. Result is returned as a
  // MaterializedRowMultiset with the ReferenceQueryErrorCode::kSuccess if
  // successful, or an std::nullopt with a ReferenceQueryErrorCode if the query
  // fails.
  std::pair<
      std::optional<std::multiset<std::vector<velox::variant>>>,
      ReferenceQueryErrorCode>
  execute(const core::PlanNodePtr& plan) override;

 private:
  std::unordered_set<std::string> aggregateFunctionNames_;
};

} // namespace facebook::velox::exec::test

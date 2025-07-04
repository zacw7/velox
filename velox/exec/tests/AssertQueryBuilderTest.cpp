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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::exec::test {

class AssertQueryBuilderTest : public HiveConnectorTestBase {};

TEST_F(AssertQueryBuilderTest, basic) {
  auto data = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});

  AssertQueryBuilder(
      PlanBuilder().values({data}).planNode(), duckDbQueryRunner_)
      .assertResults("VALUES (1), (2), (3)");

  AssertQueryBuilder(PlanBuilder().values({data}).planNode())
      .assertResults(data);
}

TEST_F(AssertQueryBuilderTest, serialExecution) {
  auto data = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});

  PlanBuilder builder;
  const auto& plan = builder.values({data}).planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .serialExecution(true)
      .assertResults("VALUES (1), (2), (3)");

  AssertQueryBuilder(plan).serialExecution(true).assertResults(data);
}

TEST_F(AssertQueryBuilderTest, orderedResults) {
  auto data = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});

  AssertQueryBuilder(
      PlanBuilder().values({data}).orderBy({"c0 DESC"}, true).planNode(),
      duckDbQueryRunner_)
      .assertResults("VALUES (3), (2), (1)", {{0}});
}

TEST_F(AssertQueryBuilderTest, concurrency) {
  auto data = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});

  AssertQueryBuilder(
      PlanBuilder().values({data}, true).planNode(), duckDbQueryRunner_)
      .maxDrivers(3)
      .assertResults("VALUES (1), (2), (3), (1), (2), (3), (1), (2), (3)");

  AssertQueryBuilder(PlanBuilder().values({data}, true).planNode())
      .maxDrivers(3)
      .assertResults({data, data, data});
}

TEST_F(AssertQueryBuilderTest, config) {
  auto data = makeRowVector({makeFlatVector<int32_t>({1, 2, 3})});

  AssertQueryBuilder(
      PlanBuilder().values({data}).project({"c0 * 2"}).planNode(),
      duckDbQueryRunner_)
      .config(core::QueryConfig::kExprEvalSimplified, "true")
      .assertResults("VALUES (2), (4), (6)");
}

TEST_F(AssertQueryBuilderTest, hiveSplits) {
  auto data = makeRowVector({"c0"}, {makeFlatVector<int32_t>({1, 2, 3})});

  auto file = TempFilePath::create();
  writeToFile(file->getPath(), {data});

  // Single leaf node.
  AssertQueryBuilder(
      PlanBuilder().tableScan(asRowType(data->type())).planNode(),
      duckDbQueryRunner_)
      .split(makeHiveConnectorSplit(file->getPath()))
      .assertResults("VALUES (1), (2), (3)");

  // Single leaf node with two splits.
  auto makeSplits = [](const std::string& path, size_t numRepeats = 1) {
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits(
        numRepeats, makeHiveConnectorSplit(path));
    return splits;
  };

  for (const auto withSequence : {false, true}) {
    AssertQueryBuilder(
        PlanBuilder().tableScan(asRowType(data->type())).planNode(),
        duckDbQueryRunner_)
        .splits(makeSplits(file->getPath(), 2))
        .addSplitWithSequence(withSequence)
        .assertResults("VALUES (1), (2), (3), (1), (2), (3)");
  }

  // Single leaf node with two splits and barrier execution.
  for (const auto withSequence : {false, true}) {
    auto splits = makeSplits(file->getPath(), 2);
    AssertQueryBuilder(
        PlanBuilder()
            .tableScan(asRowType(data->type()))
            .project({"c0 + 1"})
            .planNode(),
        duckDbQueryRunner_)
        .splits(splits)
        .addSplitWithSequence(withSequence)
        .barrierExecution(true)
        .serialExecution(true)
        .assertResults("VALUES (2), (3), (4), (2), (3), (4)");
  }

  // Split with partition key.
  connector::ColumnHandleMap assignments = {
      {"ds", partitionKey("ds", VARCHAR())},
      {"c0", regularColumn("c0", BIGINT())}};

  AssertQueryBuilder(
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c0", "ds"}, {INTEGER(), VARCHAR()}))
          .tableHandle(makeTableHandle())
          .assignments(assignments)
          .endTableScan()
          .planNode(),
      duckDbQueryRunner_)
      .split(HiveConnectorSplitBuilder(file->getPath())
                 .partitionKey("ds", "2022-05-10")
                 .build())
      .assertResults(
          "VALUES (1, '2022-05-10'), (2, '2022-05-10'), (3, '2022-05-10')");

  // Two leaf nodes.
  auto buildData = makeRowVector({makeFlatVector<int32_t>({2, 3})});
  auto buildFile = TempFilePath::create();
  writeToFile(buildFile->getPath(), {buildData});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId probeScanId;
  core::PlanNodeId buildScanId;
  auto joinPlan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(asRowType(data->type()))
                      .capturePlanNodeId(probeScanId)
                      .hashJoin(
                          {"c0"},
                          {"b_c0"},
                          PlanBuilder(planNodeIdGenerator)
                              .tableScan(asRowType(data->type()))
                              .capturePlanNodeId(buildScanId)
                              .project({"c0 as b_c0"})
                              .planNode(),
                          "",
                          {"c0", "b_c0"})
                      .singleAggregation({}, {"count(1)"})
                      .planNode();

  AssertQueryBuilder(joinPlan, duckDbQueryRunner_)
      .split(probeScanId, makeHiveConnectorSplit(file->getPath()))
      .split(buildScanId, makeHiveConnectorSplit(buildFile->getPath()))
      .assertResults("SELECT 2");
}

TEST_F(AssertQueryBuilderTest, encodedResults) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 1000;
  opts.nullRatio = 0.1;

  VectorFuzzer fuzzer(opts, pool_.get());

  // Dict(Array).
  auto input =
      makeRowVector({fuzzer.fuzzDictionary(fuzzer.fuzzFlat(ARRAY(INTEGER())))});
  auto flatInput = flatten<RowVector>(input);
  assertEqualResults({flatInput}, {input});

  // Const(Array).
  input = makeRowVector({fuzzer.fuzzConstant(ARRAY(INTEGER()))});
  flatInput = flatten<RowVector>(input);
  assertEqualResults({flatInput}, {input});

  // Dict(Map).
  input = makeRowVector(
      {fuzzer.fuzzDictionary(fuzzer.fuzzFlat(MAP(INTEGER(), VARCHAR())))});
  flatInput = flatten<RowVector>(input);
  assertEqualResults({flatInput}, {input});

  // Const(Map).
  input = makeRowVector({fuzzer.fuzzConstant(MAP(INTEGER(), VARCHAR()))});
  flatInput = flatten<RowVector>(input);
  assertEqualResults({flatInput}, {input});

  // Dict(Row).
  input = makeRowVector({fuzzer.fuzzDictionary(fuzzer.fuzzFlat(
      ROW({"c0", "c1", "c2", "c3"},
          {INTEGER(), VARCHAR(), BOOLEAN(), ARRAY(INTEGER())})))});
  flatInput = flatten<RowVector>(input);
  assertEqualResults({flatInput}, {input});

  // Const(Row).
  input = makeRowVector({fuzzer.fuzzConstant(
      ROW({"c0", "c1", "c2", "c3"},
          {INTEGER(), VARCHAR(), BOOLEAN(), ARRAY(INTEGER())}))});
  flatInput = flatten<RowVector>(input);
  assertEqualResults({flatInput}, {input});
}

TEST_F(AssertQueryBuilderTest, nestedArrayMapResults) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 1000;
  opts.nullRatio = 0.1;

  VectorFuzzer fuzzer(opts, pool_.get());

  // Array(Array).
  auto input = makeRowVector({fuzzer.fuzzFlat(ARRAY(ARRAY(BIGINT())))});
  assertEqualResults({input}, {input});

  // Map(Map).
  input = makeRowVector({fuzzer.fuzzDictionary(
      fuzzer.fuzzFlat(MAP(INTEGER(), MAP(INTEGER(), VARCHAR()))))});
  assertEqualResults({input}, {input});

  // Map(Array).
  input = makeRowVector({fuzzer.fuzzDictionary(
      fuzzer.fuzzFlat(MAP(INTEGER(), ARRAY(VARCHAR()))))});
  assertEqualResults({input}, {input});

  // Array(Map).
  input = makeRowVector({fuzzer.fuzzDictionary(
      fuzzer.fuzzFlat(ARRAY(MAP(INTEGER(), VARCHAR()))))});
  assertEqualResults({input}, {input});
}

} // namespace facebook::velox::exec::test

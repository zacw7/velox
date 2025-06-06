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
#include "velox/exec/fuzzer/AggregationFuzzer.h"

#include <boost/random/uniform_int_distribution.hpp>

#include "velox/common/base/Portability.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"

#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

#include "velox/exec/PartitionFunction.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/expression/fuzzer/FuzzerToolkit.h"
#include "velox/vector/VectorSaver.h"

DEFINE_bool(
    enable_sorted_aggregations,
    true,
    "When true, generates plans with aggregations over sorted inputs");

using facebook::velox::fuzzer::CallableSignature;
using facebook::velox::fuzzer::SignatureTemplate;

namespace facebook::velox::exec::test {

class AggregationFuzzerBase;

namespace {

class AggregationFuzzer : public AggregationFuzzerBase {
 public:
  AggregationFuzzer(
      AggregateFunctionSignatureMap signatureMap,
      size_t seed,
      const std::unordered_set<std::string>& functionsRequireSortedInput,
      const std::unordered_map<std::string, std::shared_ptr<ResultVerifier>>&
          customVerificationFunctions,
      const std::unordered_map<std::string, std::shared_ptr<InputGenerator>>&
          customInputGenerators,
      const std::unordered_map<std::string, DataSpec>& functionDataSpec,
      VectorFuzzer::Options::TimestampPrecision timestampPrecision,
      const std::unordered_map<std::string, std::string>& queryConfigs,
      const std::unordered_map<std::string, std::string>& hiveConfigs,
      bool orderableGroupKeys,
      std::unique_ptr<ReferenceQueryRunner> referenceQueryRunner);

  void go();
  void go(const std::string& planPath);

 private:
  struct Stats : public AggregationFuzzerBase::Stats {
    // Number of iterations using masked aggregation.
    size_t numMask{0};

    // Number of iterations using group-by aggregation.
    size_t numGroupBy{0};

    // Number of iterations using global aggregation.
    size_t numGlobal{0};

    // Number of iterations using distinct aggregation.
    size_t numDistinct{0};

    // Number of iterations using aggregations over distinct inputs.
    size_t numDistinctInputs{0};

    void print(size_t numIterations) const;
  };

  // Return 'true' if query plans failed.
  bool verifyAggregation(
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const std::vector<std::string>& masks,
      const std::vector<RowVectorPtr>& input,
      const std::vector<core::ExprPtr>& projections,
      bool customVerification,
      const std::shared_ptr<ResultVerifier>& customVerifier);

  // Return 'true' if query plans failed.
  bool verifySortedAggregation(
      const std::vector<std::string>& groupingKeys,
      const std::string& aggregate,
      const std::vector<std::string>& masks,
      const std::vector<RowVectorPtr>& input,
      const std::vector<core::ExprPtr>& projections,
      bool customVerification,
      const std::shared_ptr<ResultVerifier>& customVerifier);

  void verifyAggregation(const std::vector<PlanWithSplits>& plans);

  // Use the result of the first plan in the plans as the expected result to
  // compare or verify it with the results of other equivalent plans.
  bool compareEquivalentPlanResults(
      const std::vector<PlanWithSplits>& plans,
      bool customVerification,
      const std::vector<RowVectorPtr>& input,
      const std::shared_ptr<ResultVerifier>& customVerifier,
      int32_t maxDrivers = 2,
      bool testWithSpilling = true);

  // Return 'true' if query plans failed.
  bool verifyDistinctAggregation(
      const std::vector<std::string>& groupingKeys,
      const std::string& aggregate,
      const std::vector<std::string>& masks,
      const std::vector<RowVectorPtr>& input,
      const std::vector<core::ExprPtr>& projections,
      bool customVerification,
      const std::shared_ptr<ResultVerifier>& customVerifier);

  static bool hasPartialGroupBy(const core::PlanNodePtr& plan) {
    auto partialAgg = core::PlanNode::findFirstNode(
        plan.get(), [](const core::PlanNode* node) {
          if (auto aggregation =
                  dynamic_cast<const core::AggregationNode*>(node)) {
            return aggregation->step() ==
                core::AggregationNode::Step::kPartial &&
                !aggregation->groupingKeys().empty();
          }

          return false;
        });
    return partialAgg != nullptr;
  }

  void testPlans(
      const std::vector<PlanWithSplits>& plans,
      bool customVerification,
      const std::vector<std::shared_ptr<ResultVerifier>>& customVerifiers,
      const velox::fuzzer::ResultOrError& expected,
      int32_t maxDrivers = 2,
      bool testWithSpilling = true) {
    for (auto i = 0; i < plans.size(); ++i) {
      const auto& planWithSplits = plans[i];

      LOG(INFO) << "Testing plan #" << i;
      testPlan(
          planWithSplits,
          false /*injectSpill*/,
          false /*abandonPartial*/,
          customVerification,
          customVerifiers,
          expected,
          maxDrivers);

      if (testWithSpilling) {
        LOG(INFO) << "Testing plan #" << i << " with spilling";
        testPlan(
            planWithSplits,
            true /*injectSpill*/,
            false /*abandonPartial*/,
            customVerification,
            customVerifiers,
            expected,
            maxDrivers);
      }

      if (hasPartialGroupBy(planWithSplits.plan)) {
        LOG(INFO) << "Testing plan #" << i
                  << " with forced abandon-partial-aggregation";
        testPlan(
            planWithSplits,
            false /*injectSpill*/,
            true /*abandonPartial*/,
            customVerification,
            customVerifiers,
            expected,
            maxDrivers);
      }
    }
  }

  bool mustSortInput(const CallableSignature& signature) const;

  Stats stats_;
  const std::unordered_map<std::string, DataSpec> functionDataSpec_;
  const std::unordered_set<std::string> functionsRequireSortedInput_;
};
} // namespace

void aggregateFuzzer(
    AggregateFunctionSignatureMap signatureMap,
    size_t seed,
    const std::unordered_set<std::string>& functionsRequireSortedInput,
    const std::unordered_map<std::string, std::shared_ptr<ResultVerifier>>&
        customVerificationFunctions,
    const std::unordered_map<std::string, std::shared_ptr<InputGenerator>>&
        customInputGenerators,
    const std::unordered_map<std::string, DataSpec>& functionDataSpec,
    VectorFuzzer::Options::TimestampPrecision timestampPrecision,
    const std::unordered_map<std::string, std::string>& queryConfigs,
    const std::unordered_map<std::string, std::string>& hiveConfigs,
    bool orderableGroupKeys,
    const std::optional<std::string>& planPath,
    std::unique_ptr<ReferenceQueryRunner> referenceQueryRunner) {
  auto aggregationFuzzer = AggregationFuzzer(
      std::move(signatureMap),
      seed,
      functionsRequireSortedInput,
      customVerificationFunctions,
      customInputGenerators,
      functionDataSpec,
      timestampPrecision,
      queryConfigs,
      hiveConfigs,
      orderableGroupKeys,
      std::move(referenceQueryRunner));
  planPath.has_value() ? aggregationFuzzer.go(planPath.value())
                       : aggregationFuzzer.go();
}

namespace {

AggregationFuzzer::AggregationFuzzer(
    AggregateFunctionSignatureMap signatureMap,
    size_t seed,
    const std::unordered_set<std::string>& functionsRequireSortedInput,
    const std::unordered_map<std::string, std::shared_ptr<ResultVerifier>>&
        customVerificationFunctions,
    const std::unordered_map<std::string, std::shared_ptr<InputGenerator>>&
        customInputGenerators,
    const std::unordered_map<std::string, DataSpec>& functionDataSpec,
    VectorFuzzer::Options::TimestampPrecision timestampPrecision,
    const std::unordered_map<std::string, std::string>& queryConfigs,
    const std::unordered_map<std::string, std::string>& hiveConfigs,
    bool orderableGroupKeys,
    std::unique_ptr<ReferenceQueryRunner> referenceQueryRunner)
    : AggregationFuzzerBase{seed, customVerificationFunctions, customInputGenerators, timestampPrecision, queryConfigs, hiveConfigs, orderableGroupKeys, std::move(referenceQueryRunner)},
      functionDataSpec_{functionDataSpec},
      functionsRequireSortedInput_{functionsRequireSortedInput} {
  VELOX_CHECK(!signatureMap.empty(), "No function signatures available.");

  if (persistAndRunOnce_ && reproPersistPath_.empty()) {
    std::cout
        << "--repro_persist_path must be specified if --persist_and_run_once is specified"
        << std::endl;
    exit(1);
  }

  addAggregationSignatures(signatureMap);
  printStats(functionsStats);

  sortCallableSignatures(signatures_);
  sortSignatureTemplates(signatureTemplates_);

  signatureStats_.resize(signatures_.size() + signatureTemplates_.size());
}

void AggregationFuzzer::go(const std::string& planPath) {
  Type::registerSerDe();
  connector::hive::HiveTableHandle::registerSerDe();
  connector::hive::LocationHandle::registerSerDe();
  connector::hive::HiveColumnHandle::registerSerDe();
  connector::hive::HiveInsertTableHandle::registerSerDe();
  core::ITypedExpr::registerSerDe();
  core::PlanNode::registerSerDe();
  registerPartitionFunctionSerDe();

  LOG(INFO) << "Attempting to use serialized plan at: " << planPath;
  auto planString = restoreStringFromFile(planPath.c_str());
  auto parsedPlans = folly::parseJson(planString);
  std::vector<PlanWithSplits> plans(parsedPlans.size());
  std::transform(
      parsedPlans.begin(),
      parsedPlans.end(),
      plans.begin(),
      [&](const folly::dynamic& obj) { return deserialize(obj); });

  verifyAggregation(plans);
}

// Returns true if specified aggregate function can be applied to sorted inputs,
// i.e. function takes 1 or more arguments (count(1) doesn't qualify) and types
// of all arguments are orderable (no maps).
bool canSortInputs(const CallableSignature& signature) {
  if (signature.args.empty()) {
    return false;
  }

  for (const auto& arg : signature.args) {
    if (!arg->isOrderable()) {
      return false;
    }
  }

  return true;
}

// Returns true if specified aggregate function can be applied to distinct
// inputs. If 'orderableGroupKeys' is true the argument type must be orderable,
// otherwise it must be comparable.
bool supportsDistinctInputs(
    const CallableSignature& signature,
    bool orderableGroupKeys) {
  if (signature.args.empty()) {
    return false;
  }

  const auto& arg = signature.args.at(0);
  if (orderableGroupKeys) {
    return arg->isOrderable();
  }
  return arg->isComparable();
}

bool AggregationFuzzer::mustSortInput(
    const CallableSignature& signature) const {
  return functionsRequireSortedInput_.count(signature.name) > 0;
}

void AggregationFuzzer::go() {
  VELOX_CHECK(
      FLAGS_steps > 0 || FLAGS_duration_sec > 0,
      "Either --steps or --duration_sec needs to be greater than zero.");

  auto startTime = std::chrono::system_clock::now();
  size_t iteration = 0;

  auto vectorOptions = vectorFuzzer_.getOptions();
  while (!isDone(iteration, startTime)) {
    LOG(INFO) << "==============================> Started iteration "
              << iteration << " (seed: " << currentSeed_ << ")";

    // 10% of times test distinct aggregation.
    if (vectorFuzzer_.coinToss(0.1)) {
      ++stats_.numDistinct;

      std::vector<TypePtr> types;
      std::vector<std::string> names;

      auto groupingKeys = generateKeys("g", names, types);
      auto input = generateInputData(names, types, std::nullopt);
      auto [convertedInput, projections] =
          referenceQueryRunner_->inputProjections(input);

      logVectors(convertedInput);

      verifyAggregation(
          groupingKeys, {}, {}, convertedInput, projections, false, {});
    } else {
      // Pick a random signature.
      auto signatureWithStats = pickSignature();
      auto signature = signatureWithStats.first;
      if (mustSortInput(signature) &&
          !(FLAGS_enable_sorted_aggregations && canSortInputs(signature))) {
        continue;
      }
      signatureWithStats.second.numRuns++;
      stats_.functionNames.insert(signature.name);

      if (functionDataSpec_.count(signatureWithStats.first.name) > 0) {
        vectorOptions.dataSpec =
            functionDataSpec_.at(signatureWithStats.first.name);

      } else {
        vectorOptions.dataSpec = {true, true};
      }
      vectorFuzzer_.setOptions(vectorOptions);

      const bool sortedInputs = mustSortInput(signature) ||
          (FLAGS_enable_sorted_aggregations && canSortInputs(signature) &&
           vectorFuzzer_.coinToss(0.2));

      // Exclude approx_xxx aggregations since their verifiers may not be able
      // to verify the results. The approx_percentile verifier would discard
      // the distinct property when calculating the expected result, say the
      // expected result of the verifier would be approx_percentile(x), which
      // may be different from the actual result of approx_percentile(distinct
      // x).
      const bool distinctInputs = !sortedInputs &&
          (signature.name.find("approx_") == std::string::npos) &&
          supportsDistinctInputs(signature, orderableGroupKeys_) &&
          vectorFuzzer_.coinToss(0.2);

      std::vector<TypePtr> argTypes = signature.args;
      std::vector<std::string> argNames = makeNames(argTypes.size());
      auto call = makeFunctionCall(
          signature.name, argNames, sortedInputs, distinctInputs);

      // 20% of times use mask.
      std::vector<std::string> masks;
      if (vectorFuzzer_.coinToss(0.2)) {
        ++stats_.numMask;

        masks.push_back("m0");
        argTypes.push_back(BOOLEAN());
        argNames.push_back(masks.back());
      }

      // 10% of times use global aggregation (no grouping keys).
      std::vector<std::string> groupingKeys;
      if (vectorFuzzer_.coinToss(0.1)) {
        ++stats_.numGlobal;
      } else {
        ++stats_.numGroupBy;
        groupingKeys = generateKeys("g", argNames, argTypes);
      }

      auto input = generateInputData(argNames, argTypes, signature);
      auto [convertedInput, projections] =
          referenceQueryRunner_->inputProjections(input);

      logVectors(convertedInput);

      const bool customVerification =
          customVerificationFunctions_.count(signature.name) != 0;
      std::shared_ptr<ResultVerifier> customVerifier;
      if (customVerification) {
        customVerifier = customVerificationFunctions_.at(signature.name);
      }

      if (sortedInputs) {
        ++stats_.numSortedInputs;
        bool failed = verifySortedAggregation(
            groupingKeys,
            call,
            masks,
            convertedInput,
            projections,
            customVerification,
            customVerifier);
        if (failed) {
          signatureWithStats.second.numFailed++;
        }
      } else if (distinctInputs) {
        ++stats_.numDistinctInputs;
        bool failed = verifyDistinctAggregation(
            groupingKeys,
            call,
            masks,
            convertedInput,
            projections,
            customVerification,
            customVerifier);
        if (failed) {
          signatureWithStats.second.numFailed++;
        }
      } else {
        bool failed = verifyAggregation(
            groupingKeys,
            {call},
            masks,
            convertedInput,
            projections,
            customVerification,
            customVerifier);
        if (failed) {
          signatureWithStats.second.numFailed++;
        }
      }
    }
    LOG(INFO) << "==============================> Done with iteration "
              << iteration;

    if (persistAndRunOnce_) {
      LOG(WARNING)
          << "Iteration succeeded with --persist_and_run_once flag enabled "
             "(expecting crash failure)";
      exit(0);
    }

    reSeed();
    ++iteration;
  }

  stats_.print(iteration);

  printSignatureStats();
}

void makeAlternativePlansWithValues(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks,
    const std::vector<RowVectorPtr>& inputVectors,
    const std::vector<core::ExprPtr>& projections,
    std::vector<core::PlanNodePtr>& plans) {
  // Partial -> final aggregation plan.
  plans.push_back(PlanBuilder()
                      .values(inputVectors)
                      .projectExpressions(projections)
                      .partialAggregation(groupingKeys, aggregates, masks)
                      .finalAggregation()
                      .planNode());

  // Partial -> intermediate -> final aggregation plan.
  plans.push_back(PlanBuilder()
                      .values(inputVectors)
                      .projectExpressions(projections)
                      .partialAggregation(groupingKeys, aggregates, masks)
                      .intermediateAggregation()
                      .finalAggregation()
                      .planNode());

  // Partial -> local exchange -> final aggregation plan.
  auto numSources = std::min<size_t>(4, inputVectors.size());
  std::vector<std::vector<RowVectorPtr>> sourceInputs(numSources);
  for (auto i = 0; i < inputVectors.size(); ++i) {
    sourceInputs[i % numSources].push_back(inputVectors[i]);
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodePtr> sources;
  for (const auto& sourceInput : sourceInputs) {
    sources.push_back(PlanBuilder(planNodeIdGenerator)
                          .values({sourceInput})
                          .projectExpressions(projections)
                          .partialAggregation(groupingKeys, aggregates, masks)
                          .planNode());
  }

// There is a known issue where LocalPartition will send DictionaryVectors
// with the same underlying base Vector to multiple threads.  This triggers
// TSAN to report data races, particularly if that base Vector is from the
// TableScan and reused.  Don't run these tests when TSAN is enabled to avoid
// the false negatives.
#ifndef TSAN_BUILD
  plans.push_back(PlanBuilder(planNodeIdGenerator)
                      .localPartition(groupingKeys, sources)
                      .finalAggregation()
                      .planNode());
#endif
}

void makeAlternativePlansWithTableScan(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks,
    const RowTypePtr& inputRowType,
    const std::vector<core::ExprPtr>& projections,
    std::vector<core::PlanNodePtr>& plans) {
// There is a known issue where LocalPartition will send DictionaryVectors
// with the same underlying base Vector to multiple threads.  This triggers
// TSAN to report data races, particularly if that base Vector is from the
// TableScan and reused.  Don't run these tests when TSAN is enabled to avoid
// the false negatives.
#ifndef TSAN_BUILD
  // Partial -> final aggregation plan.
  plans.push_back(PlanBuilder()
                      .tableScan(inputRowType)
                      .projectExpressions(projections)
                      .partialAggregation(groupingKeys, aggregates, masks)
                      .localPartition(groupingKeys)
                      .finalAggregation()
                      .planNode());

  // Partial -> intermediate -> final aggregation plan.
  plans.push_back(PlanBuilder()
                      .tableScan(inputRowType)
                      .projectExpressions(projections)
                      .partialAggregation(groupingKeys, aggregates, masks)
                      .localPartition(groupingKeys)
                      .intermediateAggregation()
                      .finalAggregation()
                      .planNode());
#endif
}

void makeStreamingPlansWithValues(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks,
    const std::vector<RowVectorPtr>& inputVectors,
    const std::vector<core::ExprPtr>& projections,
    std::vector<core::PlanNodePtr>& plans) {
  // Single aggregation.
  plans.push_back(PlanBuilder()
                      .values(inputVectors)
                      .projectExpressions(projections)
                      .orderBy(groupingKeys, false)
                      .streamingAggregation(
                          groupingKeys,
                          aggregates,
                          masks,
                          core::AggregationNode::Step::kSingle,
                          false)
                      .planNode());

  // Partial -> final aggregation plan.
  plans.push_back(
      PlanBuilder()
          .values(inputVectors)
          .projectExpressions(projections)
          .orderBy(groupingKeys, false)
          .partialStreamingAggregation(groupingKeys, aggregates, masks)
          .finalAggregation()
          .planNode());

  // Partial -> intermediate -> final aggregation plan.
  plans.push_back(
      PlanBuilder()
          .values(inputVectors)
          .projectExpressions(projections)
          .orderBy(groupingKeys, false)
          .partialStreamingAggregation(groupingKeys, aggregates, masks)
          .intermediateAggregation()
          .finalAggregation()
          .planNode());

  // Partial -> local merge -> final aggregation plan.
  auto numSources = std::min<size_t>(4, inputVectors.size());
  std::vector<std::vector<RowVectorPtr>> sourceInputs(numSources);
  for (auto i = 0; i < inputVectors.size(); ++i) {
    sourceInputs[i % numSources].push_back(inputVectors[i]);
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodePtr> sources;
  for (const auto& sourceInput : sourceInputs) {
    sources.push_back(
        PlanBuilder(planNodeIdGenerator)
            .values({sourceInput})
            .projectExpressions(projections)
            .orderBy(groupingKeys, false)
            .partialStreamingAggregation(groupingKeys, aggregates, masks)
            .planNode());
  }
  plans.push_back(PlanBuilder(planNodeIdGenerator)
                      .localMerge(groupingKeys, sources)
                      .finalAggregation()
                      .planNode());
}

void makeStreamingPlansWithTableScan(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks,
    const RowTypePtr& inputRowType,
    const std::vector<core::ExprPtr>& projections,
    std::vector<core::PlanNodePtr>& plans) {
  // Single aggregation.
  plans.push_back(PlanBuilder()
                      .tableScan(inputRowType)
                      .projectExpressions(projections)
                      .orderBy(groupingKeys, false)
                      .streamingAggregation(
                          groupingKeys,
                          aggregates,
                          masks,
                          core::AggregationNode::Step::kSingle,
                          false)
                      .planNode());

  // Partial -> final aggregation plan.
  plans.push_back(
      PlanBuilder()
          .tableScan(inputRowType)
          .projectExpressions(projections)
          .orderBy(groupingKeys, false)
          .partialStreamingAggregation(groupingKeys, aggregates, masks)
          .finalAggregation()
          .planNode());

  // Partial -> intermediate -> final aggregation plan.
  plans.push_back(
      PlanBuilder()
          .tableScan(inputRowType)
          .projectExpressions(projections)
          .orderBy(groupingKeys, false)
          .partialStreamingAggregation(groupingKeys, aggregates, masks)
          .intermediateAggregation()
          .finalAggregation()
          .planNode());

  // Partial -> local merge -> final aggregation plan.
  plans.push_back(
      PlanBuilder()
          .tableScan(inputRowType)
          .projectExpressions(projections)
          .orderBy(groupingKeys, true)
          .partialStreamingAggregation(groupingKeys, aggregates, masks)
          .localMerge(groupingKeys)
          .finalAggregation()
          .planNode());
}

bool AggregationFuzzer::verifyAggregation(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks,
    const std::vector<RowVectorPtr>& input,
    const std::vector<core::ExprPtr>& projections,
    bool customVerification,
    const std::shared_ptr<ResultVerifier>& customVerifier) {
  auto firstPlan = PlanBuilder()
                       .values(input)
                       .projectExpressions(projections)
                       .singleAggregation(groupingKeys, aggregates, masks)
                       .planNode();

  if (customVerification && customVerifier != nullptr) {
    const auto& aggregationNode =
        std::dynamic_pointer_cast<const core::AggregationNode>(firstPlan);

    customVerifier->initialize(
        input,
        projections,
        groupingKeys,
        aggregationNode->aggregates()[0],
        aggregationNode->aggregateNames()[0]);
  }

  SCOPE_EXIT {
    if (customVerification && customVerifier != nullptr) {
      customVerifier->reset();
    }
  };

  // Create all the plans upfront.
  std::vector<PlanWithSplits> plans;
  plans.push_back({firstPlan, {}});

  auto directory = exec::test::TempDirectoryPath::create();

  // Alternate between using Values and TableScan node.

  const auto inputRowType = asRowType(input[0]->type());
  if (isTableScanSupported(inputRowType) && vectorFuzzer_.coinToss(0.5)) {
    auto splits = makeSplits(input, directory->getPath(), writerPool_);

    std::vector<core::PlanNodePtr> tableScanPlans;
    makeAlternativePlansWithTableScan(
        groupingKeys,
        aggregates,
        masks,
        inputRowType,
        projections,
        tableScanPlans);

    if (!groupingKeys.empty()) {
      // Use OrderBy + StreamingAggregation on original input.
      makeStreamingPlansWithTableScan(
          groupingKeys,
          aggregates,
          masks,
          inputRowType,
          projections,
          tableScanPlans);
    }

    for (const auto& plan : tableScanPlans) {
      plans.push_back({plan, splits});
    }
  } else {
    std::vector<core::PlanNodePtr> valuesPlans;
    makeAlternativePlansWithValues(
        groupingKeys, aggregates, masks, input, projections, valuesPlans);

    // Evaluate same plans on flat inputs.
    std::vector<RowVectorPtr> flatInput;
    for (const auto& vector : input) {
      auto flat = BaseVector::create<RowVector>(
          vector->type(), vector->size(), vector->pool());
      flat->copy(vector.get(), 0, 0, vector->size());
      flatInput.push_back(flat);
    }

    makeAlternativePlansWithValues(
        groupingKeys, aggregates, masks, flatInput, projections, valuesPlans);

    if (!groupingKeys.empty()) {
      // Use OrderBy + StreamingAggregation on original input.
      makeStreamingPlansWithValues(
          groupingKeys, aggregates, masks, input, projections, valuesPlans);

      // Use OrderBy + StreamingAggregation on flattened input.
      makeStreamingPlansWithValues(
          groupingKeys, aggregates, masks, flatInput, projections, valuesPlans);
    }

    for (const auto& plan : valuesPlans) {
      plans.push_back({plan, {}});
    }
  }

  if (persistAndRunOnce_) {
    persistReproInfo(plans, reproPersistPath_);
  }

  return compareEquivalentPlanResults(
      plans, customVerification, input, customVerifier);
}

bool AggregationFuzzer::verifySortedAggregation(
    const std::vector<std::string>& groupingKeys,
    const std::string& aggregate,
    const std::vector<std::string>& masks,
    const std::vector<RowVectorPtr>& input,
    const std::vector<core::ExprPtr>& projections,
    bool customVerification,
    const std::shared_ptr<ResultVerifier>& customVerifier) {
  auto firstPlan = PlanBuilder()
                       .values(input)
                       .projectExpressions(projections)
                       .singleAggregation(groupingKeys, {aggregate}, masks)
                       .planNode();

  bool aggregateOrderSensitive = false;

  if (customVerification && customVerifier != nullptr) {
    const auto& aggregationNode =
        std::dynamic_pointer_cast<const core::AggregationNode>(firstPlan);
    const auto& aggregateFunctionCall = aggregationNode->aggregates()[0];
    const std::string& aggregateFunctionName =
        aggregateFunctionCall.call->name();

    customVerifier->initialize(
        input,
        projections,
        groupingKeys,
        aggregateFunctionCall,
        aggregationNode->aggregateNames()[0]);

    auto* aggregateFunctionEntry =
        getAggregateFunctionEntry(aggregateFunctionName);
    aggregateOrderSensitive = aggregateFunctionEntry->metadata.orderSensitive;
  }

  SCOPE_EXIT {
    if (customVerification && customVerifier != nullptr) {
      customVerifier->reset();
    }
  };

  std::vector<PlanWithSplits> plans;
  plans.push_back({firstPlan, {}});

  if (!groupingKeys.empty()) {
    plans.push_back(
        {PlanBuilder()
             .values(input)
             .projectExpressions(projections)
             .orderBy(groupingKeys, false)
             .streamingAggregation(
                 groupingKeys,
                 {aggregate},
                 masks,
                 core::AggregationNode::Step::kSingle,
                 false)
             .planNode(),
         {}});
  }

  std::shared_ptr<exec::test::TempDirectoryPath> directory;
  const auto inputRowType = asRowType(input[0]->type());
  if (isTableScanSupported(inputRowType)) {
    directory = exec::test::TempDirectoryPath::create();
    auto splits = makeSplits(input, directory->getPath(), writerPool_);

    plans.push_back(
        {PlanBuilder()
             .tableScan(inputRowType)
             .projectExpressions(projections)
             .singleAggregation(groupingKeys, {aggregate}, masks)
             .planNode(),
         splits});

    if (!groupingKeys.empty()) {
      plans.push_back(
          {PlanBuilder()
               .tableScan(inputRowType)
               .projectExpressions(projections)
               .orderBy(groupingKeys, false)
               .streamingAggregation(
                   groupingKeys,
                   {aggregate},
                   masks,
                   core::AggregationNode::Step::kSingle,
                   false)
               .planNode(),
           splits});
    }
  }

  if (customVerification &&
      (!aggregateOrderSensitive || customVerifier == nullptr ||
       customVerifier->supportsVerify() || customVerifier->supportsCompare())) {
    // We have custom verification enabled and:
    // 1) the aggregate function is not order sensitive (sorting the input won't
    //    have an effect on the output) or
    // 2) the custom verifier is null (we've deliberately turned off
    //    verification of this aggregation) or
    // 3) the custom verifier supports verification (it can't compare the
    //    results of the aggregation with the reference DB)
    // 4) the custom verifier supports compare.
    // keep the custom verifier enabled.
    return compareEquivalentPlanResults(
        plans, customVerification, input, customVerifier, 1);
  } else {
    // If custom verification is not enabled, then compare the results directly.
    return compareEquivalentPlanResults(plans, false, input, nullptr, 1);
  }
}

// verifyAggregation(std::vector<core::PlanNodePtr> plans) is tied to plan
// created by previous verifyAggregation function. Changes in nodes there will
// require corresponding changes here.
void AggregationFuzzer::verifyAggregation(
    const std::vector<PlanWithSplits>& plans) {
  VELOX_CHECK_GT(plans.size(), 0);
  const auto& plan = plans.front().plan;

  const auto node = dynamic_cast<const core::AggregationNode*>(plan.get());
  VELOX_CHECK_NOT_NULL(node);

  // Get groupingKeys.
  auto groupingKeys = node->groupingKeys();
  std::vector<std::string> groupingKeyNames;
  groupingKeyNames.reserve(groupingKeys.size());

  for (const auto& key : groupingKeys) {
    groupingKeyNames.push_back(key->name());
  }

  // Get masks.
  std::vector<std::string> maskNames;
  maskNames.reserve(node->aggregates().size());

  for (const auto& aggregate : node->aggregates()) {
    if (aggregate.mask) {
      maskNames.push_back(aggregate.mask->name());
    }
  }

  // Search for the source node starting from the AggregationNode
  core::PlanNodePtr source = plan;

  while (!dynamic_cast<const core::ValuesNode*>(source.get())) {
    // The AggregationNode and any of it's preceding nodes can only have a
    // single source. This will fail if a node with multiple sources is found
    // (which would require updating the logic to handle), or if no ValuesNode
    // is found.
    VELOX_CHECK_EQ(source->sources().size(), 1);
    source = source->sources()[0];
  }

  // Get inputs.
  std::vector<RowVectorPtr> input;

  auto valueNode = dynamic_cast<const core::ValuesNode*>(source.get());
  VELOX_CHECK_NOT_NULL(valueNode);
  auto values = valueNode->values();
  input.insert(input.end(), values.begin(), values.end());

  auto resultOrError = execute(plan);
  if (resultOrError.exceptionPtr) {
    ++stats_.numFailed;
  }

  // Get aggregations and determine if order dependent.
  const int32_t numAggregates = node->aggregates().size();

  std::vector<std::string> aggregateStrings;
  aggregateStrings.reserve(numAggregates);

  bool customVerification = false;
  std::vector<std::shared_ptr<ResultVerifier>> customVerifiers(numAggregates);
  for (auto aggregate : node->aggregates()) {
    aggregateStrings.push_back(aggregate.call->toString());

    const auto& name = aggregate.call->name();
    auto it = customVerificationFunctions_.find(name);
    if (it != customVerificationFunctions_.end()) {
      customVerification = true;
      customVerifiers.push_back(it->second);
    } else {
      customVerifiers.push_back(nullptr);
    }
  }

  std::optional<MaterializedRowMultiset> expectedResult;
  if (!customVerification) {
    auto referenceResult =
        computeReferenceResults(plan, referenceQueryRunner_.get());
    stats_.updateReferenceQueryStats(referenceResult.second);
    expectedResult = referenceResult.first;
  }

  if (expectedResult && resultOrError.result) {
    ++stats_.numVerified;
    VELOX_CHECK(
        assertEqualResults(
            expectedResult.value(), plan->outputType(), {resultOrError.result}),
        "Velox and reference DB results don't match");
    LOG(INFO) << "Verified results against reference DB";
  }

  // Test all plans.
  testPlans(plans, customVerification, customVerifiers, resultOrError);
}

void AggregationFuzzer::Stats::print(size_t numIterations) const {
  LOG(ERROR) << "Total masked aggregations: "
             << printPercentageStat(numMask, numIterations);
  LOG(ERROR) << "Total global aggregations: "
             << printPercentageStat(numGlobal, numIterations);
  LOG(ERROR) << "Total group-by aggregations: "
             << printPercentageStat(numGroupBy, numIterations);
  LOG(ERROR) << "Total distinct aggregations: "
             << printPercentageStat(numDistinct, numIterations);
  LOG(ERROR) << "Total aggregations over distinct inputs: "
             << printPercentageStat(numDistinctInputs, numIterations);
  AggregationFuzzerBase::Stats::print(numIterations);
}

bool AggregationFuzzer::compareEquivalentPlanResults(
    const std::vector<PlanWithSplits>& plans,
    bool customVerification,
    const std::vector<RowVectorPtr>& input,
    const std::shared_ptr<ResultVerifier>& customVerifier,
    int32_t maxDrivers,
    bool testWithSpilling) {
  try {
    auto firstPlan = plans.at(0).plan;
    auto resultOrError = execute(firstPlan);
    if (resultOrError.exceptionPtr) {
      ++stats_.numFailed;
    }

    // If Velox successfully executes a plan, we attempt to verify
    // the plan against the reference DB as follows:
    // 1) If deterministic function (i.e. customVerification is false)
    //    then try and have the reference DB execute the plan and assert
    //    results are equal.
    // 2) If Non deterministic function, and if the reference query runner
    //    supports Velox vectors then we have the reference DB execute the plan
    //    and use ResultVerifier::compare api (if supported ) to validate the
    //    results.

    if (resultOrError.result != nullptr) {
      if (!customVerification) {
        auto referenceResult =
            computeReferenceResults(firstPlan, referenceQueryRunner_.get());
        stats_.updateReferenceQueryStats(referenceResult.second);
        auto expectedResult = referenceResult.first;

        if (expectedResult) {
          ++stats_.numVerified;
          VELOX_CHECK(
              assertEqualResults(
                  expectedResult.value(),
                  firstPlan->outputType(),
                  {resultOrError.result}),
              "Velox and reference DB results don't match, plan: {}",
              firstPlan->toString(true, true));
          LOG(INFO) << "Verified results against reference DB";
        }
      } else if (referenceQueryRunner_->supportsVeloxVectorResults()) {
        if (isSupportedType(firstPlan->outputType()) &&
            isSupportedType(input.front()->type())) {
          auto referenceResult = computeReferenceResultsAsVector(
              firstPlan, referenceQueryRunner_.get());
          stats_.updateReferenceQueryStats(referenceResult.second);

          if (referenceResult.first) {
            velox::fuzzer::ResultOrError expected;
            expected.result = fuzzer::mergeRowVectors(
                referenceResult.first.value(), pool_.get());

            compare(
                resultOrError, customVerification, {customVerifier}, expected);
            ++stats_.numVerified;
          }
        }
      }
    }

    testPlans(
        plans,
        customVerification,
        {customVerifier},
        resultOrError,
        maxDrivers,
        testWithSpilling);

    return resultOrError.exceptionPtr != nullptr;
  } catch (...) {
    if (!reproPersistPath_.empty()) {
      persistReproInfo(plans, reproPersistPath_);
    }
    throw;
  }
}

bool AggregationFuzzer::verifyDistinctAggregation(
    const std::vector<std::string>& groupingKeys,
    const std::string& aggregate,
    const std::vector<std::string>& masks,
    const std::vector<RowVectorPtr>& input,
    const std::vector<core::ExprPtr>& projections,
    bool customVerification,
    const std::shared_ptr<ResultVerifier>& customVerifier) {
  const auto firstPlan =
      PlanBuilder()
          .values(input)
          .projectExpressions(projections)
          .singleAggregation(groupingKeys, {aggregate}, masks)
          .planNode();

  if (customVerification) {
    if (customVerification && customVerifier != nullptr) {
      const auto& aggregationNode =
          std::dynamic_pointer_cast<const core::AggregationNode>(firstPlan);

      customVerifier->initialize(
          input,
          projections,
          groupingKeys,
          aggregationNode->aggregates()[0],
          aggregationNode->aggregateNames()[0]);
    }
  }

  SCOPE_EXIT {
    if (customVerification && customVerifier != nullptr) {
      customVerifier->reset();
    }
  };

  // Create all the plans upfront.
  std::vector<PlanWithSplits> plans;
  plans.push_back({firstPlan, {}});

  if (!groupingKeys.empty()) {
    plans.push_back(
        {PlanBuilder()
             .values(input)
             .projectExpressions(projections)
             .orderBy(groupingKeys, false)
             .streamingAggregation(
                 groupingKeys,
                 {aggregate},
                 masks,
                 core::AggregationNode::Step::kSingle,
                 false)
             .planNode(),
         {}});
  }

  // Alternate between using Values and TableScan node.

  std::shared_ptr<exec::test::TempDirectoryPath> directory;
  const auto inputRowType = asRowType(input[0]->type());
  if (isTableScanSupported(inputRowType) && vectorFuzzer_.coinToss(0.5)) {
    directory = exec::test::TempDirectoryPath::create();
    auto splits = makeSplits(input, directory->getPath(), writerPool_);

    plans.push_back(
        {PlanBuilder()
             .tableScan(inputRowType)
             .projectExpressions(projections)
             .singleAggregation(groupingKeys, {aggregate}, masks)
             .planNode(),
         splits});

    if (!groupingKeys.empty()) {
      plans.push_back(
          {PlanBuilder()
               .tableScan(inputRowType)
               .projectExpressions(projections)
               .orderBy(groupingKeys, false)
               .streamingAggregation(
                   groupingKeys,
                   {aggregate},
                   masks,
                   core::AggregationNode::Step::kSingle,
                   false)
               .planNode(),
           splits});
    }
  }

  if (persistAndRunOnce_) {
    persistReproInfo(plans, reproPersistPath_);
  }

  // Distinct aggregation must run single-threaded or data must be partitioned
  // on group-by keys among threads.
  return compareEquivalentPlanResults(
      plans, customVerification, input, customVerifier, 1, false);
}

} // namespace
} // namespace facebook::velox::exec::test

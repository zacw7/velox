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
#include "velox/exec/fuzzer/AggregationFuzzerBase.h"

#include <boost/random/uniform_int_distribution.hpp>
#include "velox/common/base/Fs.h"
#include "velox/common/base/VeloxException.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/Spill.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/expression/fuzzer/ArgumentTypeFuzzer.h"
#include "velox/vector/VectorSaver.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int32(steps, 10, "Number of plans to generate and execute.");

DEFINE_int32(
    duration_sec,
    0,
    "For how long it should run (in seconds). If zero, "
    "it executes exactly --steps iterations and exits.");

DEFINE_int32(
    batch_size,
    100,
    "The number of elements on each generated vector.");

DEFINE_int32(num_batches, 10, "The number of generated vectors.");

DEFINE_int32(
    max_num_varargs,
    5,
    "The maximum number of variadic arguments fuzzer will generate for "
    "functions that accept variadic arguments. Fuzzer will generate up to "
    "max_num_varargs arguments for the variadic list in addition to the "
    "required arguments by the function.");

DEFINE_double(
    null_ratio,
    0.1,
    "Chance of adding a null constant to the plan, or null value in a vector "
    "(expressed as double from 0 to 1).");

DEFINE_string(
    repro_persist_path,
    "",
    "Directory path for persistence of data and SQL when fuzzer fails for "
    "future reproduction. Empty string disables this feature.");

DEFINE_bool(
    persist_and_run_once,
    false,
    "Persist repro info before evaluation and only run one iteration. "
    "This is to rerun with the seed number and persist repro info upon a "
    "crash failure. Only effective if repro_persist_path is set.");

DEFINE_bool(
    log_signature_stats,
    false,
    "Log statistics about function signatures");

DEFINE_bool(
    enable_oom_injection,
    false,
    "When enabled OOMs will randomly be triggered while executing query "
    "plans. The goal of this mode is to ensure unexpected exceptions "
    "aren't thrown and the process isn't killed in the process of cleaning "
    "up after failures. Therefore, results are not compared when this is "
    "enabled. Note that this option only works in debug builds.");

namespace facebook::velox::exec::test {

int32_t AggregationFuzzerBase::randInt(int32_t min, int32_t max) {
  return boost::random::uniform_int_distribution<int32_t>(min, max)(rng_);
}

bool AggregationFuzzerBase::isSupportedType(const TypePtr& type) const {
  // Date / IntervalDayTime/ Unknown are not currently supported by DWRF.
  if (type->isDate() || type->isIntervalDayTime() || type->isUnKnown()) {
    return false;
  }

  for (auto i = 0; i < type->size(); ++i) {
    if (!isSupportedType(type->childAt(i))) {
      return false;
    }
  }

  return true;
}

bool AggregationFuzzerBase::addSignature(
    const std::string& name,
    const FunctionSignaturePtr& signature) {
  ++functionsStats.numSignatures;

  if (signature->variableArity()) {
    LOG(WARNING) << "Skipping variadic function signature: " << name
                 << signature->toString();
    return false;
  }

  if (!signature->variables().empty()) {
    bool skip = false;
    std::unordered_set<std::string> typeVariables;
    for (auto& [variableName, variable] : signature->variables()) {
      if (variable.isIntegerParameter()) {
        LOG(WARNING) << "Skipping generic function signature: " << name
                     << signature->toString();
        skip = true;
        break;
      }

      typeVariables.insert(variableName);
    }
    if (skip) {
      return false;
    }

    signatureTemplates_.push_back(
        {name, signature.get(), std::move(typeVariables)});
  } else {
    CallableSignature callable{
        .name = name,
        .args = {},
        .returnType =
            SignatureBinder::tryResolveType(signature->returnType(), {}, {}),
        .constantArgs = {}};
    VELOX_CHECK_NOT_NULL(callable.returnType);

    // Process each argument and figure out its type.
    for (const auto& arg : signature->argumentTypes()) {
      auto resolvedType = SignatureBinder::tryResolveType(arg, {}, {});
      VELOX_CHECK_NOT_NULL(resolvedType);

      // SignatureBinder::tryResolveType produces ROW types with empty
      // field names. These won't work with TableScan.
      if (resolvedType->isRow()) {
        std::vector<std::string> names;
        for (auto i = 0; i < resolvedType->size(); ++i) {
          names.push_back(fmt::format("field{}", i));
        }

        std::vector<TypePtr> types = resolvedType->asRow().children();

        resolvedType = ROW(std::move(names), std::move(types));
      }

      callable.args.emplace_back(resolvedType);
    }

    signatures_.emplace_back(callable);
  }

  ++functionsStats.numSupportedSignatures;
  return true;
}

void AggregationFuzzerBase::addAggregationSignatures(
    const AggregateFunctionSignatureMap& signatureMap) {
  for (auto& [name, signatures] : signatureMap) {
    ++functionsStats.numFunctions;
    bool hasSupportedSignature = false;
    for (auto& signature : signatures) {
      hasSupportedSignature |= addSignature(name, signature);
    }
    if (hasSupportedSignature) {
      ++functionsStats.numSupportedFunctions;
    }
  }
}

std::pair<CallableSignature, AggregationFuzzerBase::SignatureStats&>
AggregationFuzzerBase::pickSignature() {
  size_t idx = boost::random::uniform_int_distribution<uint32_t>(
      0, signatures_.size() + signatureTemplates_.size() - 1)(rng_);
  CallableSignature signature;
  if (idx < signatures_.size()) {
    signature = signatures_[idx];
  } else {
    const auto& signatureTemplate =
        signatureTemplates_[idx - signatures_.size()];
    signature.name = signatureTemplate.name;
    velox::fuzzer::ArgumentTypeFuzzer typeFuzzer(
        *signatureTemplate.signature,
        rng_,
        referenceQueryRunner_->supportedScalarTypes());
    VELOX_CHECK(typeFuzzer.fuzzArgumentTypes(FLAGS_max_num_varargs));
    signature.args = typeFuzzer.argumentTypes();
  }

  return {signature, signatureStats_[idx]};
}

std::vector<std::string> AggregationFuzzerBase::generateKeys(
    const std::string& prefix,
    std::vector<std::string>& names,
    std::vector<TypePtr>& types) {
  auto numKeys = boost::random::uniform_int_distribution<uint32_t>(1, 5)(rng_);
  std::vector<std::string> keys;
  for (auto i = 0; i < numKeys; ++i) {
    keys.push_back(fmt::format("{}{}", prefix, i));

    // Pick random, possibly complex, type.
    if (orderableGroupKeys_) {
      types.push_back(vectorFuzzer_.randOrderableType(supportedKeyTypes_, 2));
    } else {
      types.push_back(vectorFuzzer_.randType(supportedKeyTypes_, 2));
    }
    names.push_back(keys.back());
  }
  return keys;
}

std::vector<std::string> AggregationFuzzerBase::generateSortingKeys(
    const std::string& prefix,
    std::vector<std::string>& names,
    std::vector<TypePtr>& types,
    bool rangeFrame,
    const std::vector<TypePtr>& scalarTypes,
    std::optional<uint32_t> numKeys) {
  std::vector<std::string> keys;
  vector_size_t maxDepth;
  std::vector<TypePtr> sortingKeyTypes = scalarTypes;

  // If frame has k-RANGE bound, only one sorting key should be present, and it
  // should be a scalar type which supports '+', '-' arithmetic operations.
  if (rangeFrame) {
    numKeys = 1;
    sortingKeyTypes = {
        TINYINT(),
        SMALLINT(),
        INTEGER(),
        BIGINT(),
        HUGEINT(),
        REAL(),
        DOUBLE()};
    maxDepth = 0;
  } else {
    if (!numKeys.has_value()) {
      numKeys = boost::random::uniform_int_distribution<uint32_t>(1, 5)(rng_);
    }
    // Pick random, possibly complex, type.
    maxDepth = 2;
  }

  for (auto i = 0; i < numKeys.value(); ++i) {
    keys.push_back(fmt::format("{}{}", prefix, i));
    types.push_back(vectorFuzzer_.randOrderableType(sortingKeyTypes, maxDepth));
    names.push_back(keys.back());
  }

  return keys;
}

std::shared_ptr<InputGenerator> AggregationFuzzerBase::findInputGenerator(
    const CallableSignature& signature) {
  auto generatorIt = customInputGenerators_.find(signature.name);
  if (generatorIt != customInputGenerators_.end()) {
    return generatorIt->second;
  }

  return nullptr;
}

std::vector<RowVectorPtr> AggregationFuzzerBase::generateInputData(
    std::vector<std::string> names,
    std::vector<TypePtr> types,
    const std::optional<CallableSignature>& signature) {
  std::shared_ptr<InputGenerator> generator;
  if (signature.has_value()) {
    generator = findInputGenerator(signature.value());
  }

  const auto size = vectorFuzzer_.getOptions().vectorSize;

  auto inputType = ROW(std::move(names), std::move(types));
  std::vector<RowVectorPtr> input;
  for (auto i = 0; i < FLAGS_num_batches; ++i) {
    std::vector<VectorPtr> children;

    if (generator != nullptr) {
      children = generator->generate(
          signature->args, vectorFuzzer_, rng_, pool_.get());
    }

    for (auto j = children.size(); j < inputType->size(); ++j) {
      children.push_back(vectorFuzzer_.fuzz(inputType->childAt(j), size));
    }

    input.push_back(std::make_shared<RowVector>(
        pool_.get(), inputType, nullptr, size, std::move(children)));
  }

  if (generator != nullptr) {
    generator->reset();
  }

  return input;
}

std::vector<RowVectorPtr> AggregationFuzzerBase::generateInputDataWithRowNumber(
    std::vector<std::string> names,
    std::vector<TypePtr> types,
    const std::vector<std::string>& partitionKeys,
    const std::vector<std::string>& windowFrameBounds,
    const std::vector<std::string>& sortingKeys,
    const CallableSignature& signature) {
  names.push_back("row_number");
  types.push_back(INTEGER());

  auto generator = findInputGenerator(signature);

  std::vector<RowVectorPtr> input;
  vector_size_t size = vectorFuzzer_.getOptions().vectorSize;
  velox::test::VectorMaker vectorMaker{pool_.get()};
  int64_t rowNumber = 0;

  std::unordered_set<std::string> partitionKeySet{
      partitionKeys.begin(), partitionKeys.end()};
  std::unordered_set<std::string> windowFrameBoundsSet{
      windowFrameBounds.begin(), windowFrameBounds.end()};
  std::unordered_set<std::string> sortingKeySet{
      sortingKeys.begin(), sortingKeys.end()};

  for (auto j = 0; j < FLAGS_num_batches; ++j) {
    std::vector<VectorPtr> children;

    if (generator != nullptr) {
      children =
          generator->generate(signature.args, vectorFuzzer_, rng_, pool_.get());
    }

    // Some window functions like 'rank' have semantics influenced by "peer"
    // rows. Peer rows are rows in the same partition having the same order by
    // key. In rank and dense_rank functions, peer rows have the same function
    // result value. This code influences the fuzzer to generate such data.
    //
    // To build such rows the code separates the notions of "peer" groups and
    // "partition" groups during data generation. A number of peers are chosen
    // between (1, size) of the input. Rows with the same peer number have the
    // same order by keys. This means that there are sets of rows in the input
    // data which will have the same order by key.
    //
    // Each peer is then mapped to a partition group. Rows in the same partition
    // group have the same partition keys. So a partition can contain a group of
    // rows with the same order by key and there can be multiple such groups
    // (each with different order by keys) in one partition.
    //
    // This style of data generation is preferable for window functions. The
    // input data so generated could look as follows:
    //
    //   numRows = 6, numPeerGroups = 3, numPartitions = 2,
    //   columns = {p0: VARCHAR, s0: INTEGER}, partitioningKeys = {p0},
    //   sortingKeys = {s0}
    //     row1: 'APPLE'   2
    //     row2: 'APPLE'   2
    //     row3: 'APPLE'   2
    //     row4: 'APPLE'   8
    //     row5: 'ORANGE'  5
    //     row6: 'ORANGE'  5
    //
    // In the above example, the sets of rows belonging to the same peer group
    // are {row1, row2, row3}, {row4}, and {row5, row6}. The sets of rows
    // belonging to the same partition are {row1, row2, row3, row4} and
    // {row5, row6}.
    auto numPeerGroups = size ? randInt(1, size) : 1;
    auto sortingIndices = vectorFuzzer_.fuzzIndices(size, numPeerGroups);
    auto rawSortingIndices = sortingIndices->as<vector_size_t>();
    auto sortingNulls = vectorFuzzer_.fuzzNulls(size);

    auto numPartitions = randInt(1, numPeerGroups);
    auto peerGroupToPartitionIndices =
        vectorFuzzer_.fuzzIndices(numPeerGroups, numPartitions);
    auto rawPeerGroupToPartitionIndices =
        peerGroupToPartitionIndices->as<vector_size_t>();
    auto partitionIndices =
        AlignedBuffer::allocate<vector_size_t>(size, pool_.get());
    auto rawPartitionIndices = partitionIndices->asMutable<vector_size_t>();
    auto partitionNulls = vectorFuzzer_.fuzzNulls(size);
    for (auto i = 0; i < size; i++) {
      auto peerGroup = rawSortingIndices[i];
      rawPartitionIndices[i] = rawPeerGroupToPartitionIndices[peerGroup];
    }

    for (auto i = children.size(); i < types.size() - 1; ++i) {
      if (partitionKeySet.find(names[i]) != partitionKeySet.end()) {
        // The partition keys are built with a dictionary over a smaller set of
        // values. This is done to introduce some repetition of key values for
        // windowing.
        auto baseVector = vectorFuzzer_.fuzz(types[i], numPartitions);
        children.push_back(BaseVector::wrapInDictionary(
            partitionNulls, partitionIndices, size, baseVector));
      } else if (
          windowFrameBoundsSet.find(names[i]) != windowFrameBoundsSet.end()) {
        // Frame bound columns cannot have NULLs.
        children.push_back(vectorFuzzer_.fuzzNotNull(types[i], size));
      } else if (sortingKeySet.find(names[i]) != sortingKeySet.end()) {
        auto baseVector = vectorFuzzer_.fuzz(types[i], numPeerGroups);
        children.push_back(BaseVector::wrapInDictionary(
            sortingNulls, sortingIndices, size, baseVector));
      } else {
        children.push_back(vectorFuzzer_.fuzz(types[i], size));
      }
    }
    children.push_back(vectorMaker.flatVector<int32_t>(
        size, [&](auto /*row*/) { return rowNumber++; }));
    input.push_back(vectorMaker.rowVector(names, children));
  }

  if (generator != nullptr) {
    generator->reset();
  }

  return input;
}

AggregationFuzzerBase::PlanWithSplits AggregationFuzzerBase::deserialize(
    const folly::dynamic& obj) {
  auto plan = velox::ISerializable::deserialize<core::PlanNode>(
      obj["plan"], pool_.get());

  std::vector<exec::Split> splits;
  if (obj.count("splits") > 0) {
    auto paths =
        ISerializable::deserialize<std::vector<std::string>>(obj["splits"]);
    for (const auto& path : paths) {
      splits.push_back(makeSplit(path));
    }
  }

  return PlanWithSplits{plan, splits};
}

void AggregationFuzzerBase::printSignatureStats() {
  if (!FLAGS_log_signature_stats) {
    return;
  }

  for (auto i = 0; i < signatureStats_.size(); ++i) {
    const auto& stats = signatureStats_[i];
    if (stats.numRuns == 0) {
      continue;
    }

    if (stats.numFailed * 1.0 / stats.numRuns < 0.5) {
      continue;
    }

    if (i < signatures_.size()) {
      LOG(INFO) << "Signature #" << i << " failed " << stats.numFailed
                << " out of " << stats.numRuns
                << " times: " << signatures_[i].toString();
    } else {
      const auto& signatureTemplate =
          signatureTemplates_[i - signatures_.size()];
      LOG(INFO) << "Signature template #" << i << " failed " << stats.numFailed
                << " out of " << stats.numRuns
                << " times: " << signatureTemplate.name << "("
                << signatureTemplate.signature->toString() << ")";
    }
  }
}

velox::fuzzer::ResultOrError AggregationFuzzerBase::execute(
    const core::PlanNodePtr& plan,
    const std::vector<exec::Split>& splits,
    bool injectSpill,
    bool abandonPartial,
    int32_t maxDrivers) {
  LOG(INFO) << "Executing query plan: " << std::endl
            << plan->toString(true, true);

  velox::fuzzer::ResultOrError resultOrError;
  try {
    std::shared_ptr<TempDirectoryPath> spillDirectory;
    AssertQueryBuilder builder(plan);

    builder.configs(queryConfigs_);

    int32_t spillPct{0};
    if (injectSpill) {
      spillDirectory = exec::test::TempDirectoryPath::create();
      builder.spillDirectory(spillDirectory->getPath())
          .config(core::QueryConfig::kSpillEnabled, "true")
          .config(core::QueryConfig::kAggregationSpillEnabled, "true")
          .config(core::QueryConfig::kMaxSpillRunRows, randInt(32, 1L << 30));
      // Randomized the spill injection with a percentage less than 100.
      spillPct = 20;
    }

    if (abandonPartial) {
      builder.config(core::QueryConfig::kAbandonPartialAggregationMinRows, "1")
          .config(core::QueryConfig::kAbandonPartialAggregationMinPct, "0")
          .config(core::QueryConfig::kMaxPartialAggregationMemory, "0")
          .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, "0");
    }

    if (!splits.empty()) {
      builder.splits(splits);
    }

    ScopedOOMInjector oomInjector(
        []() -> bool { return folly::Random::oneIn(10); },
        10); // Check the condition every 10 ms.
    if (FLAGS_enable_oom_injection) {
      oomInjector.enable();
    }

    TestScopedSpillInjection scopedSpillInjection(spillPct);
    resultOrError.result =
        builder.maxDrivers(maxDrivers).copyResults(pool_.get());
  } catch (VeloxUserError&) {
    // NOTE: velox user exception is accepted as it is caused by the invalid
    // fuzzer test inputs.
    resultOrError.exceptionPtr = std::current_exception();
  } catch (VeloxRuntimeError& e) {
    if (FLAGS_enable_oom_injection &&
        e.errorCode() == facebook::velox::error_code::kMemCapExceeded &&
        e.message() == ScopedOOMInjector::kErrorMessage) {
      // If we enabled OOM injection we expect the exception thrown by the
      // ScopedOOMInjector. Set the exceptionPtr, in case anything up stream
      // attempts to use the results if exceptionPtr is not set.
      resultOrError.exceptionPtr = std::current_exception();
    } else {
      throw e;
    }
  }

  return resultOrError;
}

void AggregationFuzzerBase::testPlan(
    const PlanWithSplits& planWithSplits,
    bool injectSpill,
    bool abandonPartial,
    bool customVerification,
    const std::vector<std::shared_ptr<ResultVerifier>>& customVerifiers,
    const velox::fuzzer::ResultOrError& expected,
    int32_t maxDrivers) {
  try {
    auto actual = execute(
        planWithSplits.plan,
        planWithSplits.splits,
        injectSpill,
        abandonPartial,
        maxDrivers);
    compare(actual, customVerification, customVerifiers, expected);
  } catch (...) {
    LOG(ERROR) << "Failed while testing plan: "
               << planWithSplits.plan->toString(true, true);
    throw;
  }
}

void AggregationFuzzerBase::compare(
    const velox::fuzzer::ResultOrError& actual,
    bool customVerification,
    const std::vector<std::shared_ptr<ResultVerifier>>& customVerifiers,
    const velox::fuzzer::ResultOrError& expected) {
  // Compare results or exceptions (if any). Fail is anything is different.
  if (FLAGS_enable_oom_injection) {
    // If OOM injection is enabled and we've made it this far and the test
    // is considered a success.  We don't bother checking the results.
    return;
  }

  // Compare results or exceptions (if any). Fail if anything is different.
  if (expected.exceptionPtr || actual.exceptionPtr) {
    // Throws in case exceptions are not compatible.
    velox::fuzzer::compareExceptions(
        expected.exceptionPtr, actual.exceptionPtr);
    return;
  }

  if (!customVerification) {
    VELOX_CHECK(
        assertEqualResults({expected.result}, {actual.result}),
        "Logically equivalent plans produced different results");
    return;
  }

  VELOX_CHECK_NOT_NULL(expected.result);
  VELOX_CHECK_NOT_NULL(actual.result);

  VELOX_CHECK_EQ(
      expected.result->size(),
      actual.result->size(),
      "Logically equivalent plans produced different number of rows");

  for (auto& verifier : customVerifiers) {
    if (verifier == nullptr) {
      continue;
    }

    if (verifier->supportsCompare()) {
      VELOX_CHECK(
          verifier->compare(expected.result, actual.result),
          "Logically equivalent plans produced different results");
      LOG(INFO) << "Verified through custom verifier.";
    } else if (verifier->supportsVerify()) {
      VELOX_CHECK(
          verifier->verify(actual.result),
          "Result of a logically equivalent plan failed custom verification");
      LOG(INFO) << "Verified through custom verifier.";
    } else {
      VELOX_UNREACHABLE(
          "Custom verifier must support either 'compare' or 'verify' API.");
    }
  }
}

namespace {
void writeToFile(
    const std::string& path,
    const VectorPtr& vector,
    memory::MemoryPool* pool) {
  dwrf::WriterOptions options;
  options.schema = vector->type();
  options.memoryPool = pool;
  auto writeFile = std::make_unique<LocalWriteFile>(path, true, false);
  auto sink =
      std::make_unique<dwio::common::WriteFileSink>(std::move(writeFile), path);
  dwrf::Writer writer(std::move(sink), options);
  writer.write(vector);
  writer.close();
}
} // namespace

void AggregationFuzzerBase::Stats::updateReferenceQueryStats(
    ReferenceQueryErrorCode errorCode) {
  if (errorCode == ReferenceQueryErrorCode::kReferenceQueryFail) {
    ++numReferenceQueryFailed;
  } else if (errorCode == ReferenceQueryErrorCode::kReferenceQueryUnsupported) {
    ++numReferenceQueryNotSupported;
  } else {
    VELOX_CHECK(
        errorCode == ReferenceQueryErrorCode::kSuccess,
        "Error should be handled by branches above.");
  }
}

void AggregationFuzzerBase::Stats::print(size_t numIterations) const {
  LOG(ERROR) << "Total functions tested: " << functionNames.size();
  LOG(ERROR) << "Total iterations requiring sorted inputs: "
             << printPercentageStat(numSortedInputs, numIterations);
  LOG(ERROR) << "Total iterations verified against reference DB: "
             << printPercentageStat(numVerified, numIterations);
  LOG(ERROR)
      << "Total functions not verified (verification skipped / not supported by reference DB / reference DB failed): "
      << printPercentageStat(numVerificationSkipped, numIterations) << " / "
      << printPercentageStat(numReferenceQueryNotSupported, numIterations)
      << " / " << printPercentageStat(numReferenceQueryFailed, numIterations);
  LOG(ERROR) << "Total failed functions: "
             << printPercentageStat(numFailed, numIterations);
}

std::string printPercentageStat(size_t n, size_t total) {
  return fmt::format("{} ({:.2f}%)", n, (double)n / total * 100);
}

void printStats(const AggregationFuzzerBase::FunctionsStats& stats) {
  LOG(ERROR) << fmt::format(
      "Total functions: {} ({} signatures)",
      stats.numFunctions,
      stats.numSignatures);
  LOG(ERROR) << "Functions with at least one supported signature: "
             << printPercentageStat(
                    stats.numSupportedFunctions, stats.numFunctions);

  size_t numNotSupportedFunctions =
      stats.numFunctions - stats.numSupportedFunctions;
  LOG(ERROR) << "Functions with no supported signature: "
             << printPercentageStat(
                    numNotSupportedFunctions, stats.numFunctions);
  LOG(ERROR) << "Supported function signatures: "
             << printPercentageStat(
                    stats.numSupportedSignatures, stats.numSignatures);

  size_t numNotSupportedSignatures =
      stats.numSignatures - stats.numSupportedSignatures;
  LOG(ERROR) << "Unsupported function signatures: "
             << printPercentageStat(
                    numNotSupportedSignatures, stats.numSignatures);
}

std::string makeFunctionCall(
    const std::string& name,
    const std::vector<std::string>& argNames,
    bool sortedInputs,
    bool distinctInputs,
    bool ignoreNulls) {
  std::ostringstream call;
  call << name << "(";

  const auto args = folly::join(", ", argNames);
  if (sortedInputs) {
    call << args << " ORDER BY " << args;
  } else if (distinctInputs) {
    call << "distinct " << args;
  } else {
    call << args;
  }
  if (ignoreNulls) {
    call << " IGNORE NULLS";
  }
  call << ")";

  return call.str();
}

std::vector<std::string> makeNames(size_t n) {
  std::vector<std::string> names;
  for (auto i = 0; i < n; ++i) {
    names.push_back(fmt::format("c{}", i));
  }
  return names;
}

folly::dynamic serialize(
    const AggregationFuzzerBase::PlanWithSplits& planWithSplits,
    const std::string& dirPath,
    std::unordered_map<std::string, std::string>& filePaths) {
  folly::dynamic obj = folly::dynamic::object();
  obj["plan"] = planWithSplits.plan->serialize();
  if (planWithSplits.splits.empty()) {
    return obj;
  }

  folly::dynamic jsonSplits = folly::dynamic::array();
  jsonSplits.reserve(planWithSplits.splits.size());
  for (const auto& split : planWithSplits.splits) {
    const auto filePath =
        std::dynamic_pointer_cast<connector::hive::HiveConnectorSplit>(
            split.connectorSplit)
            ->filePath;
    if (filePaths.count(filePath) == 0) {
      const auto newFilePath = fmt::format("{}/{}", dirPath, filePaths.size());
      fs::copy(filePath, newFilePath);
      filePaths.insert({filePath, newFilePath});
    }
    jsonSplits.push_back(filePaths.at(filePath));
  }
  obj["splits"] = jsonSplits;
  return obj;
}

void persistReproInfo(
    const std::vector<AggregationFuzzerBase::PlanWithSplits>& plans,
    const std::string& basePath) {
  if (!common::generateFileDirectory(basePath.c_str())) {
    return;
  }

  // Create a new directory
  const auto dirPathOptional =
      common::generateTempFolderPath(basePath.c_str(), "aggregationVerifier");
  if (!dirPathOptional.has_value()) {
    LOG(ERROR)
        << "Failed to create directory for persisting plans using base path: "
        << basePath;
    return;
  }

  const auto dirPath = dirPathOptional.value();

  // Save plans and splits.
  const std::string planPath = fmt::format("{}/{}", dirPath, kPlanNodeFileName);
  std::unordered_map<std::string, std::string> filePaths;
  try {
    folly::dynamic array = folly::dynamic::array();
    array.reserve(plans.size());
    for (auto planWithSplits : plans) {
      array.push_back(serialize(planWithSplits, dirPath, filePaths));
    }
    auto planJson = folly::toJson(array);
    saveStringToFile(planJson, planPath.c_str());
    LOG(INFO) << "Persisted aggregation plans to " << planPath;
  } catch (std::exception& e) {
    LOG(ERROR) << "Failed to store aggregation plans to " << planPath << ": "
               << e.what();
  }
}

std::vector<std::string> retrieveWindowFunctionName(
    const core::PlanNodePtr& node) {
  auto windowNode = std::dynamic_pointer_cast<const core::WindowNode>(node);
  VELOX_CHECK_NOT_NULL(windowNode);
  std::vector<std::string> functionNames;
  for (const auto& function : windowNode->windowFunctions()) {
    functionNames.push_back(function.functionCall->name());
  }
  return functionNames;
}

} // namespace facebook::velox::exec::test

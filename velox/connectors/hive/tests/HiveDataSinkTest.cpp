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

#include <gtest/gtest.h>
#include "velox/common/caching/AsyncDataCache.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <folly/init/Init.h>
#include <re2/re2.h>
#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/writer/FlushPolicy.h"
#include "velox/dwio/dwrf/writer/Writer.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#endif

#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::connector::hive {
namespace {

using namespace facebook::velox::common;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::testutil;

class HiveDataSinkTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
#ifdef VELOX_ENABLE_PARQUET
    parquet::registerParquetReaderFactory();
    parquet::registerParquetWriterFactory();
#endif
    Type::registerSerDe();
    HiveSortingColumn::registerSerDe();
    HiveBucketProperty::registerSerDe();

    rowType_ =
        ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
            {BIGINT(),
             INTEGER(),
             SMALLINT(),
             REAL(),
             DOUBLE(),
             VARCHAR(),
             BOOLEAN()});

    setupMemoryPools();

    spillExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(
        std::thread::hardware_concurrency());
  }

  void TearDown() override {
    connectorQueryCtx_.reset();
    connectorPool_.reset();
    opPool_.reset();
    root_.reset();
    HiveConnectorTestBase::TearDown();
  }

  std::vector<RowVectorPtr> createVectors(int vectorSize, int numVectors) {
    VectorFuzzer::Options options;
    options.vectorSize = vectorSize;
    VectorFuzzer fuzzer(options, pool());
    std::vector<RowVectorPtr> vectors;
    for (int i = 0; i < numVectors; ++i) {
      vectors.push_back(fuzzer.fuzzInputRow(rowType_));
    }
    return vectors;
  }

  std::unique_ptr<SpillConfig> getSpillConfig(
      const std::string& spillPath,
      uint64_t writerFlushThreshold) {
    return std::make_unique<SpillConfig>(
        [spillPath]() -> const std::string& { return spillPath; },
        [&](uint64_t) {},
        "",
        0,
        0,
        /*readBufferSize=*/1 << 20,
        spillExecutor_.get(),
        10,
        20,
        0,
        0,
        0,
        0,
        writerFlushThreshold,
        "none");
  }

  void setupMemoryPools() {
    connectorQueryCtx_.reset();
    connectorPool_.reset();
    opPool_.reset();
    root_.reset();

    root_ = memory::memoryManager()->addRootPool(
        "HiveDataSinkTest", 1L << 30, exec::MemoryReclaimer::create());
    opPool_ = root_->addLeafChild("operator");
    connectorPool_ =
        root_->addAggregateChild("connector", exec::MemoryReclaimer::create());

    connectorQueryCtx_ = std::make_unique<connector::ConnectorQueryCtx>(
        opPool_.get(),
        connectorPool_.get(),
        connectorSessionProperties_.get(),
        nullptr,
        common::PrefixSortConfig(),
        nullptr,
        nullptr,
        "query.HiveDataSinkTest",
        "task.HiveDataSinkTest",
        "planNodeId.HiveDataSinkTest",
        0,
        "");
  }

  std::shared_ptr<connector::hive::HiveInsertTableHandle>
  createHiveInsertTableHandle(
      const RowTypePtr& outputRowType,
      const std::string& outputDirectoryPath,
      dwio::common::FileFormat fileFormat = dwio::common::FileFormat::DWRF,
      const std::vector<std::string>& partitionedBy = {},
      const std::shared_ptr<connector::hive::HiveBucketProperty>&
          bucketProperty = nullptr,
      const std::shared_ptr<dwio::common::WriterOptions>& writerOptions =
          nullptr,
      const bool ensureFiles = false) {
    return makeHiveInsertTableHandle(
        outputRowType->names(),
        outputRowType->children(),
        partitionedBy,
        bucketProperty,
        makeLocationHandle(
            outputDirectoryPath,
            std::nullopt,
            connector::hive::LocationHandle::TableType::kNew),
        fileFormat,
        CompressionKind::CompressionKind_ZSTD,
        {},
        writerOptions,
        ensureFiles);
  }

  std::shared_ptr<HiveDataSink> createDataSink(
      const RowTypePtr& rowType,
      const std::string& outputDirectoryPath,
      dwio::common::FileFormat fileFormat = dwio::common::FileFormat::DWRF,
      const std::vector<std::string>& partitionedBy = {},
      const std::shared_ptr<connector::hive::HiveBucketProperty>&
          bucketProperty = nullptr,
      const std::shared_ptr<dwio::common::WriterOptions>& writerOptions =
          nullptr,
      const bool ensureFiles = false) {
    return std::make_shared<HiveDataSink>(
        rowType,
        createHiveInsertTableHandle(
            rowType,
            outputDirectoryPath,
            fileFormat,
            partitionedBy,
            bucketProperty,
            writerOptions,
            ensureFiles),
        connectorQueryCtx_.get(),
        CommitStrategy::kNoCommit,
        connectorConfig_);
  }

  std::vector<std::string> listFiles(const std::string& dirPath) {
    std::vector<std::string> files;
    for (auto& dirEntry : fs::recursive_directory_iterator(dirPath)) {
      if (dirEntry.is_regular_file()) {
        files.push_back(dirEntry.path().string());
      }
    }
    return files;
  }

  void verifyWrittenData(const std::string& dirPath, int32_t numFiles = 1) {
    const std::vector<std::string> filePaths = listFiles(dirPath);
    ASSERT_EQ(filePaths.size(), numFiles);
    std::vector<std::shared_ptr<connector::ConnectorSplit>> splits;
    std::for_each(filePaths.begin(), filePaths.end(), [&](auto filePath) {
      splits.push_back(makeHiveConnectorSplit(filePath));
    });
    HiveConnectorTestBase::assertQuery(
        PlanBuilder().tableScan(rowType_).planNode(),
        splits,
        fmt::format("SELECT * FROM tmp"));
  }

  void setConnectorQueryContext(
      std::unique_ptr<ConnectorQueryCtx> connectorQueryCtx) {
    connectorQueryCtx_ = std::move(connectorQueryCtx);
  }

  const std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();

  std::shared_ptr<memory::MemoryPool> root_;
  std::shared_ptr<memory::MemoryPool> opPool_;
  std::shared_ptr<memory::MemoryPool> connectorPool_;
  RowTypePtr rowType_;
  std::shared_ptr<config::ConfigBase> connectorSessionProperties_ =
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>(),
          /*mutable=*/true);
  std::unique_ptr<ConnectorQueryCtx> connectorQueryCtx_;
  std::shared_ptr<HiveConfig> connectorConfig_ =
      std::make_shared<HiveConfig>(std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  std::unique_ptr<folly::IOThreadPoolExecutor> spillExecutor_;
};

TEST_F(HiveDataSinkTest, hiveSortingColumn) {
  struct {
    std::string sortColumn;
    core::SortOrder sortOrder;
    bool badColumn;
    std::string exceptionString;
    std::string expectedToString;

    std::string debugString() const {
      return fmt::format(
          "sortColumn {} sortOrder {} badColumn {} exceptionString {} expectedToString {}",
          sortColumn,
          sortOrder.toString(),
          badColumn,
          exceptionString,
          expectedToString);
    }
  } testSettings[] = {
      {"a",
       core::SortOrder{true, true},
       false,
       "",
       "[COLUMN[a] ORDER[ASC NULLS FIRST]]"},
      {"a",
       core::SortOrder{false, false},
       false,
       "",
       "[COLUMN[a] ORDER[DESC NULLS LAST]]"},
      {"",
       core::SortOrder{true, true},
       true,
       "hive sort column must be set",
       ""},
      {"a",
       core::SortOrder{true, false},
       true,
       "Bad hive sort order: [COLUMN[a] ORDER[ASC NULLS LAST]]",
       ""},
      {"a",
       core::SortOrder{false, true},
       true,
       "Bad hive sort order: [COLUMN[a] ORDER[DESC NULLS FIRST]]",
       ""}};
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    if (testData.badColumn) {
      VELOX_ASSERT_THROW(
          HiveSortingColumn(testData.sortColumn, testData.sortOrder),
          testData.exceptionString);
      continue;
    }
    const HiveSortingColumn column(testData.sortColumn, testData.sortOrder);
    ASSERT_EQ(column.sortOrder(), testData.sortOrder);
    ASSERT_EQ(column.sortColumn(), testData.sortColumn);
    ASSERT_EQ(column.toString(), testData.expectedToString);
    auto obj = column.serialize();
    const auto deserializedColumn = HiveSortingColumn::deserialize(obj, pool());
    ASSERT_EQ(obj, deserializedColumn->serialize());
  }
}

TEST_F(HiveDataSinkTest, hiveBucketProperty) {
  const std::vector<std::string> columns = {"a", "b", "c"};
  const std::vector<TypePtr> types = {INTEGER(), VARCHAR(), BIGINT()};
  const std::vector<std::shared_ptr<const HiveSortingColumn>> sortedColumns = {
      std::make_shared<HiveSortingColumn>("d", core::SortOrder{false, false}),
      std::make_shared<HiveSortingColumn>("e", core::SortOrder{false, false}),
      std::make_shared<HiveSortingColumn>("f", core::SortOrder{true, true})};
  struct {
    HiveBucketProperty::Kind kind;
    std::vector<std::string> bucketedBy;
    std::vector<TypePtr> bucketedTypes;
    uint32_t bucketCount;
    std::vector<std::shared_ptr<const HiveSortingColumn>> sortedBy;
    bool badProperty;
    std::string exceptionString;
    std::string expectedToString;
  } testSettings[] = {
      {HiveBucketProperty::Kind::kPrestoNative,
       {columns[0]},
       {types[0], types[1]},
       4,
       {},
       true,
       "The number of hive bucket columns and types do not match",
       ""},
      {HiveBucketProperty::Kind::kPrestoNative,
       {columns[0], columns[1]},
       {types[0]},
       4,
       {},
       true,
       "The number of hive bucket columns and types do not match",
       ""},
      {HiveBucketProperty::Kind::kPrestoNative,
       {columns[0], columns[1]},
       {},
       4,
       {},
       true,
       "The number of hive bucket columns and types do not match",
       ""},
      {HiveBucketProperty::Kind::kPrestoNative,
       {},
       {types[0]},
       4,
       {},
       true,
       "Hive bucket columns must be set",
       ""},
      {HiveBucketProperty::Kind::kPrestoNative,
       {columns[0], columns[1]},
       {types[0], types[1]},
       0,
       {},
       true,
       "Hive bucket count can't be zero",
       ""},
      {HiveBucketProperty::Kind::kPrestoNative,
       {columns[0], columns[1]},
       {types[0], types[1]},
       4,
       {},
       false,
       "",
       "\nHiveBucketProperty[<PRESTO_NATIVE 4>\n"
       "\tBucket Columns:\n"
       "\t\ta\n"
       "\t\tb\n"
       "\tBucket Types:\n"
       "\t\tINTEGER\n"
       "\t\tVARCHAR\n"
       "]\n"},
      {HiveBucketProperty::Kind::kPrestoNative,
       {columns[0]},
       {types[0]},
       4,
       {},
       false,
       "",
       "\nHiveBucketProperty[<PRESTO_NATIVE 4>\n\tBucket Columns:\n\t\ta\n\tBucket Types:\n\t\tINTEGER\n]\n"},
      {HiveBucketProperty::Kind::kPrestoNative,
       {columns[0], columns[1]},
       {types[0], types[1]},
       4,
       {{sortedColumns[0]}},
       false,
       "",
       "\nHiveBucketProperty[<PRESTO_NATIVE 4>\n\tBucket Columns:\n\t\ta\n\t\tb\n\tBucket Types:\n\t\tINTEGER\n\t\tVARCHAR\n\tSortedBy Columns:\n\t\t[COLUMN[d] ORDER[DESC NULLS LAST]]\n]\n"},
      {HiveBucketProperty::Kind::kPrestoNative,
       {columns[0]},
       {types[0]},
       4,
       {{sortedColumns[0], sortedColumns[2]}},
       false,
       "",
       "\nHiveBucketProperty[<PRESTO_NATIVE 4>\n\tBucket Columns:\n\t\ta\n\tBucket Types:\n\t\tINTEGER\n\tSortedBy Columns:\n\t\t[COLUMN[d] ORDER[DESC NULLS LAST]]\n\t\t[COLUMN[f] ORDER[ASC NULLS FIRST]]\n]\n"},

      {HiveBucketProperty::Kind::kPrestoNative,
       {columns[0]},
       {types[0], types[1]},
       4,
       {},
       true,
       "The number of hive bucket columns and types do not match",
       ""},
      {HiveBucketProperty::Kind::kHiveCompatible,
       {columns[0], columns[1]},
       {types[0]},
       4,
       {},
       true,
       "The number of hive bucket columns and types do not match",
       ""},
      {HiveBucketProperty::Kind::kHiveCompatible,
       {columns[0], columns[1]},
       {},
       4,
       {},
       true,
       "The number of hive bucket columns and types do not match",
       ""},
      {HiveBucketProperty::Kind::kHiveCompatible,
       {},
       {types[0]},
       4,
       {},
       true,
       "Hive bucket columns must be set",
       ""},
      {HiveBucketProperty::Kind::kHiveCompatible,
       {columns[0], columns[1]},
       {types[0], types[1]},
       0,
       {},
       true,
       "Hive bucket count can't be zero",
       ""},
      {HiveBucketProperty::Kind::kHiveCompatible,
       {columns[0], columns[1]},
       {types[0], types[1]},
       4,
       {},
       false,
       "",
       "\nHiveBucketProperty[<HIVE_COMPATIBLE 4>\n"
       "\tBucket Columns:\n"
       "\t\ta\n"
       "\t\tb\n"
       "\tBucket Types:\n"
       "\t\tINTEGER\n"
       "\t\tVARCHAR\n"
       "]\n"},
      {HiveBucketProperty::Kind::kHiveCompatible,
       {columns[0]},
       {types[0]},
       4,
       {},
       false,
       "",
       "\nHiveBucketProperty[<HIVE_COMPATIBLE 4>\n\tBucket Columns:\n\t\ta\n\tBucket Types:\n\t\tINTEGER\n]\n"},
      {HiveBucketProperty::Kind::kHiveCompatible,
       {columns[0], columns[1]},
       {types[0], types[1]},
       4,
       {{sortedColumns[0]}},
       false,
       "",
       "\nHiveBucketProperty[<HIVE_COMPATIBLE 4>\n\tBucket Columns:\n\t\ta\n\t\tb\n\tBucket Types:\n\t\tINTEGER\n\t\tVARCHAR\n\tSortedBy Columns:\n\t\t[COLUMN[d] ORDER[DESC NULLS LAST]]\n]\n"},
      {HiveBucketProperty::Kind::kHiveCompatible,
       {columns[0]},
       {types[0]},
       4,
       {{sortedColumns[0], sortedColumns[2]}},
       false,
       "",
       "\nHiveBucketProperty[<HIVE_COMPATIBLE 4>\n\tBucket Columns:\n\t\ta\n\tBucket Types:\n\t\tINTEGER\n\tSortedBy Columns:\n\t\t[COLUMN[d] ORDER[DESC NULLS LAST]]\n\t\t[COLUMN[f] ORDER[ASC NULLS FIRST]]\n]\n"},
  };
  for (const auto& testData : testSettings) {
    if (testData.badProperty) {
      VELOX_ASSERT_THROW(
          HiveBucketProperty(
              testData.kind,
              testData.bucketCount,
              testData.bucketedBy,
              testData.bucketedTypes,
              testData.sortedBy),
          testData.exceptionString);
      continue;
    }
    HiveBucketProperty hiveProperty(
        testData.kind,
        testData.bucketCount,
        testData.bucketedBy,
        testData.bucketedTypes,
        testData.sortedBy);
    ASSERT_EQ(hiveProperty.kind(), testData.kind);
    ASSERT_EQ(hiveProperty.sortedBy(), testData.sortedBy);
    ASSERT_EQ(hiveProperty.bucketedBy(), testData.bucketedBy);
    ASSERT_EQ(hiveProperty.bucketedTypes(), testData.bucketedTypes);
    ASSERT_EQ(hiveProperty.toString(), testData.expectedToString);

    auto obj = hiveProperty.serialize();
    const auto deserializedProperty =
        HiveBucketProperty::deserialize(obj, pool());
    ASSERT_EQ(obj, deserializedProperty->serialize());
  }
}

TEST_F(HiveDataSinkTest, basic) {
  const auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createDataSink(rowType_, outputDirectory->getPath());
  auto stats = dataSink->stats();
  ASSERT_TRUE(stats.empty()) << stats.toString();
  ASSERT_EQ(
      stats.toString(),
      "numWrittenBytes 0B numWrittenFiles 0 spillRuns[0] spilledInputBytes[0B] "
      "spilledBytes[0B] spilledRows[0] spilledPartitions[0] spilledFiles[0] "
      "spillFillTimeNanos[0ns] spillSortTimeNanos[0ns] spillExtractVectorTime[0ns] spillSerializationTimeNanos[0ns] "
      "spillWrites[0] spillFlushTimeNanos[0ns] spillWriteTimeNanos[0ns] "
      "maxSpillExceededLimitCount[0] spillReadBytes[0B] spillReads[0] "
      "spillReadTimeNanos[0ns] spillReadDeserializationTimeNanos[0ns]");

  const int numBatches = 10;
  const auto vectors = createVectors(500, numBatches);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }
  stats = dataSink->stats();
  ASSERT_FALSE(stats.empty());
  ASSERT_GT(stats.numWrittenBytes, 0);
  ASSERT_EQ(stats.numWrittenFiles, 0);
  ASSERT_TRUE(dataSink->finish());
  ASSERT_TRUE(dataSink->finish());
  const auto partitions = dataSink->close();
  stats = dataSink->stats();
  ASSERT_FALSE(stats.empty());
  ASSERT_EQ(partitions.size(), 1);

  createDuckDbTable(vectors);
  verifyWrittenData(outputDirectory->getPath());
}

TEST_F(HiveDataSinkTest, basicBucket) {
  const auto outputDirectory = TempDirectoryPath::create();

  const int32_t numBuckets = 4;
  auto bucketProperty = std::make_shared<HiveBucketProperty>(
      HiveBucketProperty::Kind::kHiveCompatible,
      numBuckets,
      std::vector<std::string>{"c0"},
      std::vector<TypePtr>{BIGINT()},
      std::vector<std::shared_ptr<const HiveSortingColumn>>{
          std::make_shared<HiveSortingColumn>(
              "c1", core::SortOrder{false, false})});
  connectorSessionProperties_->set(
      HiveConfig::kSortWriterFinishTimeSliceLimitMsSession, "1");
  auto dataSink = createDataSink(
      rowType_,
      outputDirectory->getPath(),
      dwio::common::FileFormat::DWRF,
      {},
      bucketProperty);
  auto stats = dataSink->stats();
  ASSERT_TRUE(stats.empty()) << stats.toString();
  ASSERT_EQ(
      stats.toString(),
      "numWrittenBytes 0B numWrittenFiles 0 spillRuns[0] spilledInputBytes[0B] "
      "spilledBytes[0B] spilledRows[0] spilledPartitions[0] spilledFiles[0] "
      "spillFillTimeNanos[0ns] spillSortTimeNanos[0ns] spillExtractVectorTime[0ns] spillSerializationTimeNanos[0ns] "
      "spillWrites[0] spillFlushTimeNanos[0ns] spillWriteTimeNanos[0ns] "
      "maxSpillExceededLimitCount[0] spillReadBytes[0B] spillReads[0] "
      "spillReadTimeNanos[0ns] spillReadDeserializationTimeNanos[0ns]");

  const int numBatches = 10;
  const auto vectors = createVectors(500, numBatches);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }
  stats = dataSink->stats();
  ASSERT_FALSE(stats.empty());
  ASSERT_GT(stats.numWrittenBytes, 0);
  ASSERT_EQ(stats.numWrittenFiles, 0);
  VELOX_ASSERT_THROW(
      dataSink->close(), "Unexpected state transition from RUNNING to CLOSED");
  while (!dataSink->finish()) {
  }
  const auto partitions = dataSink->close();
  stats = dataSink->stats();
  ASSERT_FALSE(stats.empty());
  ASSERT_EQ(partitions.size(), numBuckets);

  createDuckDbTable(vectors);
  verifyWrittenData(outputDirectory->getPath(), numBuckets);
}

TEST_F(HiveDataSinkTest, close) {
  for (bool empty : {true, false}) {
    SCOPED_TRACE(fmt::format("Data sink is empty: {}", empty));
    const auto outputDirectory = TempDirectoryPath::create();
    auto dataSink = createDataSink(rowType_, outputDirectory->getPath());

    auto vectors = createVectors(500, 1);

    if (!empty) {
      dataSink->appendData(vectors[0]);
      ASSERT_GT(dataSink->stats().numWrittenBytes, 0);
    } else {
      ASSERT_EQ(dataSink->stats().numWrittenBytes, 0);
    }
    ASSERT_TRUE(dataSink->finish());
    const auto partitions = dataSink->close();
    // Can't append after close.
    VELOX_ASSERT_THROW(
        dataSink->appendData(vectors.back()), "Hive data sink is not running");
    VELOX_ASSERT_THROW(
        dataSink->close(), "Unexpected state transition from CLOSED to CLOSED");
    VELOX_ASSERT_THROW(
        dataSink->abort(),
        "Unexpected state transition from CLOSED to ABORTED");

    const auto stats = dataSink->stats();
    if (!empty) {
      ASSERT_EQ(partitions.size(), 1);
      ASSERT_GT(stats.numWrittenBytes, 0);
      createDuckDbTable(vectors);
      verifyWrittenData(outputDirectory->getPath());
    } else {
      ASSERT_TRUE(partitions.empty());
      ASSERT_EQ(stats.numWrittenBytes, 0);
    }
  }
}

TEST_F(HiveDataSinkTest, abort) {
  for (bool empty : {true, false}) {
    SCOPED_TRACE(fmt::format("Data sink is empty: {}", empty));
    const auto outputDirectory = TempDirectoryPath::create();
    auto dataSink = createDataSink(rowType_, outputDirectory->getPath());

    auto vectors = createVectors(1, 1);
    int initialBytes = 0;
    if (!empty) {
      dataSink->appendData(vectors.back());
      initialBytes = dataSink->stats().numWrittenBytes;
      ASSERT_GT(initialBytes, 0);
    } else {
      initialBytes = dataSink->stats().numWrittenBytes;
      ASSERT_EQ(initialBytes, 0);
    }
    dataSink->abort();
    const auto stats = dataSink->stats();
    ASSERT_TRUE(stats.empty());
    // Can't close after abort.
    VELOX_ASSERT_THROW(
        dataSink->close(),
        "Unexpected state transition from ABORTED to CLOSED");
    VELOX_ASSERT_THROW(
        dataSink->abort(),
        "Unexpected state transition from ABORTED to ABORTED");
    // Can't append after abort.
    VELOX_ASSERT_THROW(
        dataSink->appendData(vectors.back()), "Hive data sink is not running");
  }
}

DEBUG_ONLY_TEST_F(HiveDataSinkTest, memoryReclaim) {
  const int numBatches = 200;
  auto vectors = createVectors(500, 200);

  struct {
    dwio::common::FileFormat format;
    bool sortWriter;
    bool writerSpillEnabled;
    uint64_t writerFlushThreshold;
    bool expectedWriterReclaimEnabled;
    bool expectedWriterReclaimed;

    std::string debugString() const {
      return fmt::format(
          "format: {}, sortWriter: {}, writerSpillEnabled: {}, writerFlushThreshold: {}, expectedWriterReclaimEnabled: {}, expectedWriterReclaimed: {}",
          dwio::common::toString(format),
          sortWriter,
          writerSpillEnabled,
          succinctBytes(writerFlushThreshold),
          expectedWriterReclaimEnabled,
          expectedWriterReclaimed);
    }
  } testSettings[] = {
      {dwio::common::FileFormat::DWRF, true, true, 1 << 30, true, true},
      {dwio::common::FileFormat::DWRF, true, true, 1, true, true},
      {dwio::common::FileFormat::DWRF, true, false, 1 << 30, false, false},
      {dwio::common::FileFormat::DWRF, true, false, 1, false, false},
      {dwio::common::FileFormat::DWRF, false, true, 1 << 30, true, false},
      {dwio::common::FileFormat::DWRF, false, true, 1, true, true},
      {dwio::common::FileFormat::DWRF, false, false, 1 << 30, false, false},
      {dwio::common::FileFormat::DWRF, false, false, 1, false, false},
  // Add Parquet with https://github.com/facebookincubator/velox/issues/5560
#if 0
      {dwio::common::FileFormat::PARQUET, true, true, 1 << 30, false, false},
      {dwio::common::FileFormat::PARQUET, true, true, 1, false, false},
      {dwio::common::FileFormat::PARQUET, true, false, 1 << 30, false, false},
      {dwio::common::FileFormat::PARQUET, true, false, 1, false, false},
      {dwio::common::FileFormat::PARQUET, false, true, 1 << 30, false, false},
      {dwio::common::FileFormat::PARQUET, false, true, 1, false, false},
      {dwio::common::FileFormat::PARQUET, false, false, 1 << 30, false, false},
      {dwio::common::FileFormat::PARQUET, false, false, 1, false, false}
#endif
  };
  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwrf::Writer::MemoryReclaimer::reclaimableBytes",
      std::function<void(dwrf::Writer*)>([&](dwrf::Writer* writer) {
        // Release before reclaim to make it not able to reclaim from reserved
        // memory.
        writer->getContext().releaseMemoryReservation();
      }));
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    setupMemoryPools();

    const auto outputDirectory = TempDirectoryPath::create();
    std::shared_ptr<HiveBucketProperty> bucketProperty;
    std::vector<std::string> partitionBy;
    if (testData.sortWriter) {
      partitionBy = {"c6"};
      bucketProperty = std::make_shared<HiveBucketProperty>(
          HiveBucketProperty::Kind::kHiveCompatible,
          4,
          std::vector<std::string>{"c0"},
          std::vector<TypePtr>{BIGINT()},
          std::vector<std::shared_ptr<const HiveSortingColumn>>{
              std::make_shared<HiveSortingColumn>(
                  "c1", core::SortOrder{false, false})});
    }
    std::shared_ptr<TempDirectoryPath> spillDirectory;
    std::unique_ptr<SpillConfig> spillConfig;
    if (testData.writerSpillEnabled) {
      spillDirectory = exec::test::TempDirectoryPath::create();
      spillConfig = getSpillConfig(
          spillDirectory->getPath(), testData.writerFlushThreshold);
      auto connectorQueryCtx = std::make_unique<connector::ConnectorQueryCtx>(
          opPool_.get(),
          connectorPool_.get(),
          connectorSessionProperties_.get(),
          spillConfig.get(),
          common::PrefixSortConfig(),
          nullptr,
          nullptr,
          "query.HiveDataSinkTest",
          "task.HiveDataSinkTest",
          "planNodeId.HiveDataSinkTest",
          0,
          "");
      setConnectorQueryContext(std::move(connectorQueryCtx));
    } else {
      auto connectorQueryCtx = std::make_unique<connector::ConnectorQueryCtx>(
          opPool_.get(),
          connectorPool_.get(),
          connectorSessionProperties_.get(),
          nullptr,
          common::PrefixSortConfig(),
          nullptr,
          nullptr,
          "query.HiveDataSinkTest",
          "task.HiveDataSinkTest",
          "planNodeId.HiveDataSinkTest",
          0,
          "");
      setConnectorQueryContext(std::move(connectorQueryCtx));
    }

    auto dataSink = createDataSink(
        rowType_,
        outputDirectory->getPath(),
        testData.format,
        partitionBy,
        bucketProperty);
    auto* hiveDataSink = static_cast<HiveDataSink*>(dataSink.get());
    ASSERT_EQ(
        hiveDataSink->canReclaim(), testData.expectedWriterReclaimEnabled);
    for (int i = 0; i < numBatches; ++i) {
      dataSink->appendData(vectors[i]);
    }
    memory::MemoryArbitrator::Stats oldStats =
        memory::memoryManager()->arbitrator()->stats();
    uint64_t reclaimableBytes{0};
    if (testData.expectedWriterReclaimed) {
      reclaimableBytes = root_->reclaimableBytes().value();
      ASSERT_GT(reclaimableBytes, 0);
      memory::testingRunArbitration();
      memory::MemoryArbitrator::Stats curStats =
          memory::memoryManager()->arbitrator()->stats();
      ASSERT_GT(curStats.reclaimedUsedBytes - oldStats.reclaimedUsedBytes, 0);
      // We expect dwrf writer set numNonReclaimableAttempts counter.
      ASSERT_LE(
          curStats.numNonReclaimableAttempts -
              oldStats.numNonReclaimableAttempts,
          1);
    } else {
      ASSERT_FALSE(root_->reclaimableBytes().has_value());
      memory::testingRunArbitration();
      memory::MemoryArbitrator::Stats curStats =
          memory::memoryManager()->arbitrator()->stats();
      ASSERT_EQ(curStats.reclaimedUsedBytes - oldStats.reclaimedUsedBytes, 0);
    }
    while (!dataSink->finish()) {
    }
    const auto partitions = dataSink->close();
    if (testData.sortWriter && testData.expectedWriterReclaimed) {
      ASSERT_FALSE(dataSink->stats().spillStats.empty());
    } else {
      ASSERT_TRUE(dataSink->stats().spillStats.empty());
    }
    ASSERT_GE(partitions.size(), 1);
  }
}

TEST_F(HiveDataSinkTest, memoryReclaimAfterClose) {
  const int numBatches = 10;
  const auto vectors = createVectors(500, 10);

  struct {
    dwio::common::FileFormat format;
    bool sortWriter;
    bool writerSpillEnabled;
    bool close;
    bool expectedWriterReclaimEnabled;

    std::string debugString() const {
      return fmt::format(
          "format: {}, sortWriter: {}, writerSpillEnabled: {}, close: {}, expectedWriterReclaimEnabled: {}",
          dwio::common::toString(format),
          sortWriter,
          writerSpillEnabled,
          close,
          expectedWriterReclaimEnabled);
    }
  } testSettings[] = {
      {dwio::common::FileFormat::DWRF, true, true, true, true},
      {dwio::common::FileFormat::DWRF, true, false, true, false},
      {dwio::common::FileFormat::DWRF, true, true, false, true},
      {dwio::common::FileFormat::DWRF, true, false, false, false},
      {dwio::common::FileFormat::DWRF, false, true, true, true},
      {dwio::common::FileFormat::DWRF, false, false, true, false},
      {dwio::common::FileFormat::DWRF, false, true, false, true},
      {dwio::common::FileFormat::DWRF, false, false, false, false}
      // Add parquet file format after fix
      // https://github.com/facebookincubator/velox/issues/5560
  };
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    std::unordered_map<std::string, std::string> connectorConfig;
    // Always allow memory reclaim from the file writer/
    connectorConfig.emplace(
        "file_writer_flush_threshold_bytes", folly::to<std::string>(0));
    // Avoid internal the stripe flush while data write.
    connectorConfig.emplace("hive.orc.writer.stripe-max-size", "1GB");
    connectorConfig.emplace("hive.orc.writer.dictionary-max-memory", "1GB");

    connectorConfig_ = std::make_shared<HiveConfig>(
        std::make_shared<config::ConfigBase>(std::move(connectorConfig)));
    const auto outputDirectory = TempDirectoryPath::create();
    std::shared_ptr<HiveBucketProperty> bucketProperty;
    std::vector<std::string> partitionBy;
    if (testData.sortWriter) {
      partitionBy = {"c6"};
      bucketProperty = std::make_shared<HiveBucketProperty>(
          HiveBucketProperty::Kind::kHiveCompatible,
          4,
          std::vector<std::string>{"c0"},
          std::vector<TypePtr>{BIGINT()},
          std::vector<std::shared_ptr<const HiveSortingColumn>>{
              std::make_shared<HiveSortingColumn>(
                  "c1", core::SortOrder{false, false})});
    }
    std::shared_ptr<TempDirectoryPath> spillDirectory;
    std::unique_ptr<SpillConfig> spillConfig;
    if (testData.writerSpillEnabled) {
      spillDirectory = exec::test::TempDirectoryPath::create();
      spillConfig = getSpillConfig(spillDirectory->getPath(), 0);
      auto connectorQueryCtx = std::make_unique<connector::ConnectorQueryCtx>(
          opPool_.get(),
          connectorPool_.get(),
          connectorSessionProperties_.get(),
          spillConfig.get(),
          common::PrefixSortConfig(),
          nullptr,
          nullptr,
          "query.HiveDataSinkTest",
          "task.HiveDataSinkTest",
          "planNodeId.HiveDataSinkTest",
          0,
          "");
      setConnectorQueryContext(std::move(connectorQueryCtx));
    } else {
      auto connectorQueryCtx = std::make_unique<connector::ConnectorQueryCtx>(
          opPool_.get(),
          connectorPool_.get(),
          connectorSessionProperties_.get(),
          nullptr,
          common::PrefixSortConfig(),
          nullptr,
          nullptr,
          "query.HiveDataSinkTest",
          "task.HiveDataSinkTest",
          "planNodeId.HiveDataSinkTest",
          0,
          "");
      setConnectorQueryContext(std::move(connectorQueryCtx));
    }

    auto dataSink = createDataSink(
        rowType_,
        outputDirectory->getPath(),
        testData.format,
        partitionBy,
        bucketProperty);
    auto* hiveDataSink = static_cast<HiveDataSink*>(dataSink.get());
    ASSERT_EQ(
        hiveDataSink->canReclaim(), testData.expectedWriterReclaimEnabled);

    for (int i = 0; i < numBatches; ++i) {
      dataSink->appendData(vectors[i]);
    }
    if (testData.close) {
      ASSERT_TRUE(dataSink->finish());
      const auto partitions = dataSink->close();
      ASSERT_GE(partitions.size(), 1);
    } else {
      dataSink->abort();
      ASSERT_TRUE(dataSink->stats().empty());
    }

    memory::MemoryReclaimer::Stats stats;
    uint64_t reclaimableBytes{0};
    if (testData.expectedWriterReclaimEnabled) {
      reclaimableBytes = root_->reclaimableBytes().value();
      if (testData.close) {
        // NOTE: file writer might not release all the memory on close
        // immediately.
        ASSERT_GE(reclaimableBytes, 0);
      } else {
        ASSERT_EQ(reclaimableBytes, 0);
      }
    } else {
      ASSERT_FALSE(root_->reclaimableBytes().has_value());
    }
    ASSERT_EQ(root_->reclaim(1L << 30, 0, stats), 0);
    ASSERT_EQ(stats.reclaimExecTimeUs, 0);
    ASSERT_EQ(stats.reclaimedBytes, 0);
    if (testData.expectedWriterReclaimEnabled) {
      ASSERT_GE(stats.numNonReclaimableAttempts, 0);
    } else {
      ASSERT_EQ(stats.numNonReclaimableAttempts, 0);
    }
  }
}

DEBUG_ONLY_TEST_F(HiveDataSinkTest, sortWriterAbortDuringFinish) {
  const auto outputDirectory = TempDirectoryPath::create();
  const int32_t numBuckets = 4;
  auto bucketProperty = std::make_shared<HiveBucketProperty>(
      HiveBucketProperty::Kind::kHiveCompatible,
      numBuckets,
      std::vector<std::string>{"c0"},
      std::vector<TypePtr>{BIGINT()},
      std::vector<std::shared_ptr<const HiveSortingColumn>>{
          std::make_shared<HiveSortingColumn>(
              "c1", core::SortOrder{false, false})});
  connectorSessionProperties_->set(
      HiveConfig::kSortWriterFinishTimeSliceLimitMsSession, "1");
  connectorSessionProperties_->set(
      HiveConfig::kSortWriterMaxOutputRowsSession, "100");
  auto dataSink = createDataSink(
      rowType_,
      outputDirectory->getPath(),
      dwio::common::FileFormat::DWRF,
      {},
      bucketProperty);
  const int numBatches{10};
  const auto vectors = createVectors(500, numBatches);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwrf::Writer::write",
      std::function<void(dwrf::Writer*)>([&](dwrf::Writer* /*unused*/) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }));

  for (int i = 0;; ++i) {
    ASSERT_FALSE(dataSink->finish());
    if (i == 2) {
      dataSink->abort();
      break;
    }
  }
  const auto stats = dataSink->stats();
  ASSERT_TRUE(stats.empty());
}

TEST_F(HiveDataSinkTest, sortWriterMemoryReclaimDuringFinish) {
  const auto outputDirectory = TempDirectoryPath::create();
  const int32_t numBuckets = 4;
  auto bucketProperty = std::make_shared<HiveBucketProperty>(
      HiveBucketProperty::Kind::kHiveCompatible,
      numBuckets,
      std::vector<std::string>{"c0"},
      std::vector<TypePtr>{BIGINT()},
      std::vector<std::shared_ptr<const HiveSortingColumn>>{
          std::make_shared<HiveSortingColumn>(
              "c1", core::SortOrder{false, false})});
  std::shared_ptr<TempDirectoryPath> spillDirectory =
      exec::test::TempDirectoryPath::create();
  std::unique_ptr<SpillConfig> spillConfig =
      getSpillConfig(spillDirectory->getPath(), 1);
  connectorSessionProperties_->set(
      HiveConfig::kSortWriterFinishTimeSliceLimitMsSession, "1");
  connectorSessionProperties_->set(
      HiveConfig::kSortWriterMaxOutputRowsSession, "100");
  auto connectorQueryCtx = std::make_unique<connector::ConnectorQueryCtx>(
      opPool_.get(),
      connectorPool_.get(),
      connectorSessionProperties_.get(),
      spillConfig.get(),
      common::PrefixSortConfig(),
      nullptr,
      nullptr,
      "query.HiveDataSinkTest",
      "task.HiveDataSinkTest",
      "planNodeId.HiveDataSinkTest",
      0,
      "");
  setConnectorQueryContext(std::move(connectorQueryCtx));
  auto dataSink = createDataSink(
      rowType_,
      outputDirectory->getPath(),
      dwio::common::FileFormat::DWRF,
      {},
      bucketProperty);
  const int numBatches{10};
  const auto vectors = createVectors(500, numBatches);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  for (int i = 0; !dataSink->finish(); ++i) {
    if (i == 2) {
      ASSERT_GT(root_->reclaimableBytes().value(), 0);
      const memory::MemoryArbitrator::Stats prevStats =
          memory::memoryManager()->arbitrator()->stats();
      memory::testingRunArbitration();
      memory::MemoryArbitrator::Stats curStats =
          memory::memoryManager()->arbitrator()->stats();
      ASSERT_GT(curStats.reclaimedUsedBytes - prevStats.reclaimedUsedBytes, 0);
    }
  }
  const auto partitions = dataSink->close();
  const auto stats = dataSink->stats();
  ASSERT_FALSE(stats.empty());
  ASSERT_EQ(partitions.size(), numBuckets);

  createDuckDbTable(vectors);
  verifyWrittenData(outputDirectory->getPath(), numBuckets);
}

DEBUG_ONLY_TEST_F(HiveDataSinkTest, sortWriterFailureTest) {
  auto vectors = createVectors(500, 10);

  const auto outputDirectory = TempDirectoryPath::create();
  const std::vector<std::string> partitionBy{"c6"};
  const auto bucketProperty = std::make_shared<HiveBucketProperty>(
      HiveBucketProperty::Kind::kHiveCompatible,
      4,
      std::vector<std::string>{"c0"},
      std::vector<TypePtr>{BIGINT()},
      std::vector<std::shared_ptr<const HiveSortingColumn>>{
          std::make_shared<HiveSortingColumn>(
              "c1", core::SortOrder{false, false})});
  const std::shared_ptr<TempDirectoryPath> spillDirectory =
      exec::test::TempDirectoryPath::create();
  std::unique_ptr<SpillConfig> spillConfig =
      getSpillConfig(spillDirectory->getPath(), 0);
  // Triggers the memory reservation in sort buffer.
  spillConfig->minSpillableReservationPct = 1'000;
  auto connectorQueryCtx = std::make_unique<connector::ConnectorQueryCtx>(
      opPool_.get(),
      connectorPool_.get(),
      connectorSessionProperties_.get(),
      spillConfig.get(),
      common::PrefixSortConfig(),
      nullptr,
      nullptr,
      "query.HiveDataSinkTest",
      "task.HiveDataSinkTest",
      "planNodeId.HiveDataSinkTest",
      0,
      "");
  setConnectorQueryContext(std::move(connectorQueryCtx));

  auto dataSink = createDataSink(
      rowType_,
      outputDirectory->getPath(),
      dwio::common::FileFormat::DWRF,
      partitionBy,
      bucketProperty);
  for (auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  SCOPED_TESTVALUE_SET(
      "facebook::velox::dwrf::Writer::write",
      std::function<void(memory::MemoryPool*)>(
          [&](memory::MemoryPool* pool) { VELOX_FAIL("inject failure"); }));

  VELOX_ASSERT_THROW(dataSink->finish(), "inject failure");
}

TEST_F(HiveDataSinkTest, insertTableHandleToString) {
  const int32_t numBuckets = 4;
  auto bucketProperty = std::make_shared<HiveBucketProperty>(
      HiveBucketProperty::Kind::kHiveCompatible,
      numBuckets,
      std::vector<std::string>{"c5"},
      std::vector<TypePtr>{VARCHAR()},
      std::vector<std::shared_ptr<const HiveSortingColumn>>{
          std::make_shared<HiveSortingColumn>(
              "c5", core::SortOrder{false, false})});
  auto insertTableHandle = createHiveInsertTableHandle(
      rowType_,
      "/path/to/test",
      dwio::common::FileFormat::DWRF,
      {"c5", "c6"},
      bucketProperty);
  ASSERT_EQ(
      insertTableHandle->toString(),
      "HiveInsertTableHandle [dwrf zstd], [inputColumns: [ HiveColumnHandle [name: c0, columnType: Regular, dataType: BIGINT, requiredSubfields: [ ]] HiveColumnHandle [name: c1, columnType: Regular, dataType: INTEGER, requiredSubfields: [ ]] HiveColumnHandle [name: c2, columnType: Regular, dataType: SMALLINT, requiredSubfields: [ ]] HiveColumnHandle [name: c3, columnType: Regular, dataType: REAL, requiredSubfields: [ ]] HiveColumnHandle [name: c4, columnType: Regular, dataType: DOUBLE, requiredSubfields: [ ]] HiveColumnHandle [name: c5, columnType: PartitionKey, dataType: VARCHAR, requiredSubfields: [ ]] HiveColumnHandle [name: c6, columnType: PartitionKey, dataType: BOOLEAN, requiredSubfields: [ ]] ], locationHandle: LocationHandle [targetPath: /path/to/test, writePath: /path/to/test, tableType: kNew, tableFileName: ], bucketProperty: \nHiveBucketProperty[<HIVE_COMPATIBLE 4>\n\tBucket Columns:\n\t\tc5\n\tBucket Types:\n\t\tVARCHAR\n\tSortedBy Columns:\n\t\t[COLUMN[c5] ORDER[DESC NULLS LAST]]\n]\n, fileNameGenerator: HiveInsertFileNameGenerator]");
}

#ifdef VELOX_ENABLE_PARQUET
TEST_F(HiveDataSinkTest, flushPolicyWithParquet) {
  const auto outputDirectory = TempDirectoryPath::create();
  auto flushPolicyFactory = []() {
    return std::make_unique<parquet::DefaultFlushPolicy>(1234, 0);
  };
  auto writeOptions = std::make_shared<parquet::WriterOptions>();
  writeOptions->flushPolicyFactory = flushPolicyFactory;
  auto dataSink = createDataSink(
      rowType_,
      outputDirectory->getPath(),
      dwio::common::FileFormat::PARQUET,
      {},
      nullptr,
      writeOptions);

  const int numBatches = 10;
  const auto vectors = createVectors(500, numBatches);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  dwio::common::ReaderOptions readerOpts{pool_.get()};
  const std::vector<std::string> filePaths =
      listFiles(outputDirectory->getPath());
  auto bufferedInput = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(filePaths[0]), readerOpts.memoryPool());

  auto reader = std::make_unique<facebook::velox::parquet::ParquetReader>(
      std::move(bufferedInput), readerOpts);
  auto fileMeta = reader->fileMetaData();
  EXPECT_EQ(fileMeta.numRowGroups(), 10);
  EXPECT_EQ(fileMeta.rowGroup(0).numRows(), 500);
}
#endif

TEST_F(HiveDataSinkTest, flushPolicyWithDWRF) {
  const auto outputDirectory = TempDirectoryPath::create();
  auto flushPolicyFactory = []() {
    return std::make_unique<dwrf::DefaultFlushPolicy>(1234, 0);
  };

  auto writeOptions = std::make_shared<dwrf::WriterOptions>();
  writeOptions->flushPolicyFactory = flushPolicyFactory;
  auto dataSink = createDataSink(
      rowType_,
      outputDirectory->getPath(),
      dwio::common::FileFormat::DWRF,
      {},
      nullptr,
      writeOptions);

  const int numBatches = 10;
  const auto vectors = createVectors(500, numBatches);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();

  dwio::common::ReaderOptions readerOpts{pool_.get()};
  const std::vector<std::string> filePaths =
      listFiles(outputDirectory->getPath());
  auto bufferedInput = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(filePaths[0]), readerOpts.memoryPool());

  auto reader = std::make_unique<facebook::velox::dwrf::DwrfReader>(
      readerOpts, std::move(bufferedInput));
  ASSERT_EQ(reader->getNumberOfStripes(), 10);
  ASSERT_EQ(reader->getRowsPerStripe()[0], 500);
}

TEST_F(HiveDataSinkTest, ensureFilesNoData) {
  const auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createDataSink(
      rowType_,
      outputDirectory->getPath(),
      dwio::common::FileFormat::DWRF,
      {}, // partitionBy
      nullptr, // bucketProperty
      nullptr, // writeOptions
      true // ensureFiles
  );

  ASSERT_TRUE(dataSink->finish());

  auto partitions = dataSink->close();
  auto stats = dataSink->stats();
  ASSERT_FALSE(stats.empty());
  ASSERT_GT(stats.numWrittenBytes, 0);
  ASSERT_EQ(stats.numWrittenFiles, 1);
  ASSERT_EQ(partitions.size(), 1);

  std::vector<RowVectorPtr> vectors{RowVector::createEmpty(rowType_, pool())};
  createDuckDbTable(vectors);
  verifyWrittenData(outputDirectory->getPath());
}

TEST_F(HiveDataSinkTest, ensureFilesWithData) {
  const auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createDataSink(
      rowType_,
      outputDirectory->getPath(),
      dwio::common::FileFormat::DWRF,
      {}, // partitionBy
      nullptr, // bucketProperty
      nullptr, // writeOptions
      true // ensureFiles
  );

  const int numBatches = 10;
  const auto vectors = createVectors(500, numBatches);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }

  ASSERT_TRUE(dataSink->finish());

  auto partitions = dataSink->close();
  auto stats = dataSink->stats();
  ASSERT_FALSE(stats.empty());
  ASSERT_GT(stats.numWrittenBytes, 0);
  ASSERT_EQ(stats.numWrittenFiles, 1);
  ASSERT_EQ(partitions.size(), 1);

  createDuckDbTable(vectors);
  verifyWrittenData(outputDirectory->getPath());
}

TEST_F(HiveDataSinkTest, ensureFilesUnsupported) {
  VELOX_ASSERT_THROW(
      makeHiveInsertTableHandle(
          rowType_->names(),
          rowType_->children(),
          {rowType_->names()[0]}, // partitionedBy
          nullptr, // bucketProperty
          makeLocationHandle(
              "/path/to/test",
              std::nullopt,
              connector::hive::LocationHandle::TableType::kNew),
          dwio::common::FileFormat::DWRF,
          CompressionKind::CompressionKind_ZSTD,
          {}, // serdeParameters
          nullptr, // writeOptions
          true // ensureFiles
          ),
      "ensureFiles is not supported with partition keys in the data");

  VELOX_ASSERT_THROW(
      makeHiveInsertTableHandle(
          rowType_->names(),
          rowType_->children(),
          {}, // partitionedBy
          {std::make_shared<HiveBucketProperty>(
              HiveBucketProperty::Kind::kPrestoNative,
              1,
              std::vector<std::string>{rowType_->names()[0]},
              std::vector<TypePtr>{rowType_->children()[0]},
              std::vector<std::shared_ptr<const HiveSortingColumn>>{})},
          makeLocationHandle(
              "/path/to/test",
              std::nullopt,
              connector::hive::LocationHandle::TableType::kNew),
          dwio::common::FileFormat::DWRF,
          CompressionKind::CompressionKind_ZSTD,
          {}, // serdeParameters
          nullptr, // writeOptions
          true // ensureFiles
          ),
      "ensureFiles is not supported with bucketing");
}

TEST_F(HiveDataSinkTest, raceWithCacheEviction) {
  /// This test ensures that LRU cache staleness and StringIdMap cache
  /// eviction do not cause issues with file reads.
  std::atomic<bool> stop{false};
  auto cacheCleaner = std::async(std::launch::async, [&] {
    auto cache = cache::AsyncDataCache::getInstance();
    auto hiveConnector = std::dynamic_pointer_cast<HiveConnector>(
        getConnector(exec::test::kHiveConnectorId));
    while (!stop) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      cache->clear();
      hiveConnector->clearFileHandleCache();
    }
  });

  const auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createDataSink(rowType_, outputDirectory->getPath());
  const auto vectors = createVectors(500 /*vectorSize*/, 10 /*numVectors*/);
  for (const auto& vector : vectors) {
    dataSink->appendData(vector);
  }
  ASSERT_TRUE(dataSink->finish());
  ASSERT_FALSE(dataSink->close().empty());

  createDuckDbTable(vectors);
  verifyWrittenData(outputDirectory->getPath());

  stop = true;
  cacheCleaner.get();
}

#ifdef VELOX_ENABLE_PARQUET
TEST_F(HiveDataSinkTest, lazyVectorForParquet) {
  // This test ensures that lazy vector is handled correctly in HiveDataSink.
  VectorFuzzer::Options options{.vectorSize = 100};
  VectorFuzzer fuzzer(options, pool());

  auto lazyVector = fuzzer.wrapInLazyVector(fuzzer.fuzzFlat(BIGINT(), 100));
  auto lazyMapVector = fuzzer.wrapInLazyVector(fuzzer.fuzzMap(
      fuzzer.fuzzFlat(BIGINT(), 100), fuzzer.fuzzFlat(VARCHAR(), 100), 100));

  auto rowType = ROW({"c0", "c1"}, {BIGINT(), MAP(BIGINT(), VARCHAR())});
  std::vector<VectorPtr> children;
  children.emplace_back(lazyVector);
  children.emplace_back(lazyMapVector);
  auto row = std::make_shared<RowVector>(
      pool(), rowType, nullptr, 100, std::move(children));

  const auto outputDirectory = TempDirectoryPath::create();
  auto dataSink = createDataSink(
      rowType, outputDirectory->getPath(), dwio::common::FileFormat::PARQUET);

  dataSink->appendData(row);
  ASSERT_TRUE(dataSink->finish());
  dataSink->close();
}
#endif

} // namespace
} // namespace facebook::velox::connector::hive

// This main is needed for some tests on linux.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // Signal handler required for ThreadDebugInfoTest
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}

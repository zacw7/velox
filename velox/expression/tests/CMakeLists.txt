# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(
  velox_expression_test
  ArrayViewTest.cpp
  ArrayWriterTest.cpp
  CastExprTest.cpp
  CoalesceTest.cpp
  ConjunctTest.cpp
  ConstantFlatVectorReaderTest.cpp
  CustomTypeTest.cpp
  ExprCompilerTest.cpp
  ExprEncodingsTest.cpp
  ExprStatsTest.cpp
  ExprTest.cpp
  ExprToSubfieldFilterTest.cpp
  EvalCtxTest.cpp
  EvalErrorsTest.cpp
  EvalSimplifiedTest.cpp
  FunctionCallToSpecialFormTest.cpp
  GenericViewTest.cpp
  GenericWriterTest.cpp
  Main.cpp
  MapViewTest.cpp
  MapWriterTest.cpp
  PeeledEncodingTest.cpp
  ReverseSignatureBinderTest.cpp
  RowViewTest.cpp
  RowWriterTest.cpp
  SignatureBinderTest.cpp
  SimpleFunctionCallNullFreeTest.cpp
  SimpleFunctionInitTest.cpp
  SimpleFunctionPresetNullsTest.cpp
  SimpleFunctionTest.cpp
  StringWriterTest.cpp
  TryExprTest.cpp
  VariadicViewTest.cpp
  VectorReaderTest.cpp)
if(VELOX_ENABLE_GEO)
  target_compile_definitions(velox_expression_test PRIVATE VELOX_ENABLE_GEO)
endif()

add_test(
  NAME velox_expression_test
  COMMAND velox_expression_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  velox_expression_test
  velox_aggregates
  velox_hive_connector
  velox_dwio_common
  velox_dwio_common_exception
  velox_exec_test_lib
  velox_expression
  velox_expression_test_utility
  velox_functions_lib
  velox_functions_prestosql
  velox_functions_test_lib
  velox_parse_parser
  velox_parse_expression
  velox_presto_serializer
  velox_type_test_lib
  velox_vector_test_lib
  velox_vector_fuzzer
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
  Folly::folly
  gflags::gflags
  glog::glog
  fmt::fmt)

add_library(velox_expression_verifier ExpressionVerifier.cpp)

target_link_libraries(
  velox_expression_verifier
  velox_vector_test_lib
  velox_vector_fuzzer
  velox_fuzzer_util
  velox_type
  velox_expression_test_utility)

add_library(velox_expression_runner ExpressionRunner.cpp)
target_link_libraries(
  velox_expression_runner
  velox_expression_verifier
  velox_fuzzer_util
  velox_functions_prestosql
  velox_functions_spark
  velox_parse_parser
  GTest::gtest)

add_executable(velox_expression_runner_test ExpressionRunnerTest.cpp)
target_link_libraries(
  velox_expression_runner_test
  velox_expression_runner
  velox_fuzzer_util
  velox_exec_test_lib
  velox_function_registry
  velox_file
  velox_hive_connector
  velox_fuzzer_util
  GTest::gtest
  GTest::gtest_main)

add_executable(velox_expression_verifier_unit_test
               ExpressionVerifierUnitTest.cpp)
target_link_libraries(
  velox_expression_verifier_unit_test
  velox_expression_verifier
  velox_file
  velox_temp_path
  velox_parse_expression
  velox_parse_parser
  velox_parse_utils
  velox_type
  velox_vector_test_lib
  GTest::gtest
  GTest::gtest_main)

add_executable(velox_expression_runner_unit_test ExpressionRunnerUnitTest.cpp)
target_link_libraries(
  velox_expression_runner_unit_test
  velox_dwio_common_test_utils
  velox_expression
  velox_expression_runner
  velox_expression_fuzzer
  velox_expression_verifier
  velox_function_registry
  velox_temp_path
  velox_exec_test_lib
  velox_vector_test_lib
  GTest::gtest
  GTest::gtest_main)

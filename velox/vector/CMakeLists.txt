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
velox_add_library(
  velox_vector
  BaseVector.cpp
  ComplexVector.cpp
  ConstantVector.cpp
  DecodedVector.cpp
  EncodedVectorCopy.cpp
  FlatMapVector.cpp
  FlatVector.cpp
  LazyVector.cpp
  SelectivityVector.cpp
  SequenceVector.cpp
  SimpleVector.cpp
  StringVectorBuffer.cpp
  VariantToVector.cpp
  VectorEncoding.cpp
  VectorMap.cpp
  VectorPool.cpp
  VectorPrinter.cpp
  VectorSaver.cpp
  VectorStream.cpp)

velox_link_libraries(
  velox_vector
  velox_encode
  velox_memory
  velox_row_fast
  velox_time
  velox_type
  velox_buffer)

add_subdirectory(arrow)
add_subdirectory(fuzzer)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
elseif(${VELOX_BUILD_TEST_UTILS} OR ${VELOX_BUILD_VECTOR_TEST_UTILS})
  add_subdirectory(tests/utils)
endif()

if(${VELOX_ENABLE_BENCHMARKS})
  add_subdirectory(benchmarks)
endif()

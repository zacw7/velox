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

add_executable(velox_text_writer_test
               TextWriterTest.cpp BufferedWriterSinkTest.cpp FileReaderUtil.cpp)

add_test(
  NAME velox_text_writer_test
  COMMAND velox_text_writer_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  velox_text_writer_test
  velox_dwio_text_writer
  velox_dwio_text_reader
  velox_dwio_common_test_utils
  velox_dwio_text_reader_register
  velox_dwio_text_writer_register
  velox_link_libs
  Boost::regex
  Folly::folly
  ${TEST_LINK_LIBS}
  GTest::gtest
  fmt::fmt)

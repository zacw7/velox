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

#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/json/SIMDJsonUtil.h"

namespace facebook::velox::functions::sparksql {

/// json_object_keys(jsonString) -> array(string)
///
/// Returns all the keys of the outermost JSON object as an array if a valid
/// JSON object is given.
template <typename T>
struct JsonObjectKeysFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<Varchar>>& out,
      const arg_type<Varchar>& json) {
    simdjson::ondemand::document jsonDoc;

    simdjson::padded_string paddedJson(json.data(), json.size());
    // The result is NULL if the given string is not a valid JSON string.
    if (simdjsonParse(paddedJson).get(jsonDoc)) {
      return false;
    }

    // The result is NULL if the given string is not a valid JSON string.
    if (isFatal(jsonDoc.type().error())) {
      return false;
    }

    // The result is NULL if the given string is not a JSON object string.
    if (jsonDoc.type() != simdjson::ondemand::json_type::object) {
      return false;
    }

    simdjson::ondemand::object jsonObject;
    // The result is NULL if the given string is not a valid JSON object string.
    if (jsonDoc.get_object().get(jsonObject)) {
      return false;
    }

    for (auto field : jsonObject) {
      if (isFatal(field.error())) {
        // On-Demand only fully validates the values used and the structure
        // leading to it.
        return false;
      }
      out.add_item().copy_from(std::string_view(field.unescaped_key()));
    }
    return true;
  }

 private:
  // TODO: After upgrade simdjson v3.12.3, we could replace with simdjson
  // function relevant simdjson function:
  // https://github.com/simdjson/simdjson/blob/master/include/simdjson/error-inl.h#L10-L12
  bool isFatal(simdjson::error_code error) noexcept {
    // Indicates the document is not valid JSON.
    return error == simdjson::error_code::TAPE_ERROR ||
        error == simdjson::error_code::INCOMPLETE_ARRAY_OR_OBJECT;
  }
};

} // namespace facebook::velox::functions::sparksql

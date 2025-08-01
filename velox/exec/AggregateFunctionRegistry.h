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

#include <string>
#include <vector>

#include "velox/type/Type.h"

namespace facebook::velox::exec {

/// Given a name of aggregate function and argument types, returns a pair of the
/// return type and intermediate type if the function exists. Throws if function
/// doesn't exist or doesn't support specified argument types.
///
/// @return a pair of {finalType, intermediateType}
std::pair<TypePtr, TypePtr> resolveAggregateFunction(
    const std::string& name,
    const std::vector<TypePtr>& argTypes);

} // namespace facebook::velox::exec

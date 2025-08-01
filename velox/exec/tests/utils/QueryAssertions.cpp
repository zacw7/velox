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
#include <chrono>

#include "duckdb/common/types.hpp" // @manual
#include "velox/duckdb/conversion/DuckConversion.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/vector/VariantToVector.h"
#include "velox/vector/VectorTypeUtils.h"

using facebook::velox::duckdb::duckdbTimestampToVelox;
using facebook::velox::duckdb::veloxTimestampToDuckDB;

DEFINE_int32(max_error_rows, 10, "Max number of listed error rows");

namespace facebook::velox::exec::test {
namespace {

static const std::string kDuckDbTimestampWarning =
    "Note: DuckDB only supports timestamps of millisecond precision. If this "
    "test involves timestamp inputs, please make sure you use the right"
    " precision.";

template <TypeKind kind>
::duckdb::Value duckValueAt(const VectorPtr& vector, vector_size_t index) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  return ::duckdb::Value(vector->as<SimpleVector<T>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::TINYINT>(
    const VectorPtr& vector,
    vector_size_t index) {
  return ::duckdb::Value::TINYINT(
      vector->as<SimpleVector<int8_t>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::SMALLINT>(
    const VectorPtr& vector,
    vector_size_t index) {
  return ::duckdb::Value::SMALLINT(
      vector->as<SimpleVector<int16_t>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::BOOLEAN>(
    const VectorPtr& vector,
    vector_size_t index) {
  return ::duckdb::Value::BOOLEAN(
      vector->as<SimpleVector<bool>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::VARCHAR>(
    const VectorPtr& vector,
    vector_size_t index) {
  // DuckDB requires zero-ending string
  auto copy = vector->as<SimpleVector<StringView>>()->valueAt(index).str();
  return ::duckdb::Value(copy);
}

template <>
::duckdb::Value duckValueAt<TypeKind::VARBINARY>(
    const VectorPtr& vector,
    vector_size_t index) {
  // DuckDB requires zero-ending string
  auto stringView = vector->as<SimpleVector<StringView>>()->valueAt(index);
  return ::duckdb::Value::BLOB(
      reinterpret_cast<const uint8_t*>(stringView.begin()), stringView.size());
}

template <>
::duckdb::Value duckValueAt<TypeKind::TIMESTAMP>(
    const VectorPtr& vector,
    vector_size_t index) {
  using T = typename KindToFlatVector<TypeKind::TIMESTAMP>::WrapperType;
  return ::duckdb::Value::TIMESTAMP(
      veloxTimestampToDuckDB(vector->as<SimpleVector<T>>()->valueAt(index)));
}

template <>
::duckdb::Value duckValueAt<TypeKind::INTEGER>(
    const VectorPtr& vector,
    vector_size_t index) {
  auto type = vector->type();
  if (type->isDate()) {
    return ::duckdb::Value::DATE(::duckdb::Date::EpochDaysToDate(
        vector->as<SimpleVector<int32_t>>()->valueAt(index)));
  }
  return ::duckdb::Value(vector->as<SimpleVector<int32_t>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::BIGINT>(
    const VectorPtr& vector,
    vector_size_t index) {
  using T = typename KindToFlatVector<TypeKind::BIGINT>::WrapperType;
  auto type = vector->type();
  if (type->isShortDecimal()) {
    const auto& decimalType = type->asShortDecimal();
    return ::duckdb::Value::DECIMAL(
        vector->as<SimpleVector<T>>()->valueAt(index),
        decimalType.precision(),
        decimalType.scale());
  }

  if (type->isIntervalDayTime()) {
    static constexpr int64_t kMicrosecondsInDay =
        1000L * 1000L * 60L * 60L * 24L;
    const auto interval = vector->as<SimpleVector<int64_t>>()->valueAt(index);
    const int64_t microseconds = interval % kMicrosecondsInDay;
    const int64_t days = interval / kMicrosecondsInDay;
    return ::duckdb::Value::INTERVAL(0, days, microseconds);
  }

  return ::duckdb::Value(vector->as<SimpleVector<T>>()->valueAt(index));
}

template <>
::duckdb::Value duckValueAt<TypeKind::HUGEINT>(
    const VectorPtr& vector,
    vector_size_t index) {
  using T = typename KindToFlatVector<TypeKind::HUGEINT>::WrapperType;
  auto val = vector->as<SimpleVector<T>>()->valueAt(index);
  auto duckVal = ::duckdb::hugeint_t();
  duckVal.lower = (val << 64) >> 64;
  duckVal.upper = (val >> 64);
  if (vector->type()->isLongDecimal()) {
    auto type = vector->type()->asLongDecimal();
    return ::duckdb::Value::DECIMAL(
        std::move(duckVal), type.precision(), type.scale());
  }
  // Flat vector is HUGEINT type and not the logical decimal type.
  return ::duckdb::Value::HUGEINT(std::move(duckVal));
}

template <>
::duckdb::Value duckValueAt<TypeKind::ARRAY>(
    const VectorPtr& vector,
    int32_t row) {
  auto arrayVector = vector->wrappedVector()->as<ArrayVector>();
  auto arrayRow = vector->wrappedIndex(row);
  auto& elements = arrayVector->elements();
  auto offset = arrayVector->offsetAt(arrayRow);
  auto size = arrayVector->sizeAt(arrayRow);

  if (size == 0) {
    return ::duckdb::Value::EMPTYLIST(duckdb::fromVeloxType(elements->type()));
  }

  ::duckdb::vector<::duckdb::Value> array;
  array.reserve(size);
  for (auto i = 0; i < size; i++) {
    auto innerRow = offset + i;
    if (elements->isNullAt(innerRow)) {
      array.emplace_back(
          ::duckdb::Value(duckdb::fromVeloxType(elements->type())));
    } else {
      array.emplace_back(VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          duckValueAt, elements->typeKind(), elements, innerRow));
    }
  }

  return ::duckdb::Value::LIST(array);
}

template <>
::duckdb::Value duckValueAt<TypeKind::ROW>(
    const VectorPtr& vector,
    int32_t row) {
  auto rowVector = vector->wrappedVector()->as<RowVector>();
  auto rowRow = vector->wrappedIndex(row);
  auto rowType = asRowType(rowVector->type());

  ::duckdb::vector<std::pair<std::string, ::duckdb::Value>> fields;
  for (auto i = 0; i < rowType->size(); ++i) {
    if (rowVector->childAt(i)->isNullAt(rowRow)) {
      fields.push_back({rowType->nameOf(i), ::duckdb::Value(nullptr)});
    } else {
      fields.push_back(
          {rowType->nameOf(i),
           VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
               duckValueAt,
               rowType->childAt(i)->kind(),
               rowVector->childAt(i),
               rowRow)});
    }
  }

  return ::duckdb::Value::STRUCT(fields);
}

template <>
::duckdb::Value duckValueAt<TypeKind::MAP>(
    const VectorPtr& vector,
    int32_t row) {
  auto mapVector = vector->wrappedVector()->as<MapVector>();
  auto mapRow = vector->wrappedIndex(row);
  const auto& mapKeys = mapVector->mapKeys();
  const auto& mapValues = mapVector->mapValues();
  auto offset = mapVector->offsetAt(mapRow);
  auto size = mapVector->sizeAt(mapRow);
  auto mapType = ::duckdb::ListType::GetChildType(::duckdb::LogicalType::MAP(
      duckdb::fromVeloxType(mapKeys->type()),
      duckdb::fromVeloxType(mapValues->type())));
  if (size == 0) {
    return ::duckdb::Value::MAP(mapType, ::duckdb::vector<::duckdb::Value>());
  }

  ::duckdb::vector<::duckdb::Value> duckMap;
  for (auto i = 0; i < size; i++) {
    auto innerRow = offset + i;
    auto key = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        duckValueAt, mapKeys->typeKind(), mapKeys, innerRow);
    ::duckdb::Value value;
    if (mapValues->isNullAt(innerRow)) {
      value = ::duckdb::Value(duckdb::fromVeloxType(mapValues->type()));
    } else {
      value = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          duckValueAt, mapValues->typeKind(), mapValues, innerRow);
    }
    ::duckdb::child_list_t<::duckdb::Value> mapStruct;
    mapStruct.push_back(make_pair("key", key));
    mapStruct.push_back(make_pair("value", value));
    duckMap.push_back(::duckdb::Value::STRUCT(mapStruct));
  }
  return ::duckdb::Value::MAP(mapType, duckMap);
}

template <TypeKind kind>
variant variantAt(::duckdb::DataChunk* dataChunk, int32_t row, int32_t column) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  return variant(dataChunk->GetValue(column, row).GetValue<T>());
}

template <>
variant variantAt<TypeKind::VARCHAR>(
    ::duckdb::DataChunk* dataChunk,
    int32_t row,
    int32_t column) {
  return variant(
      StringView(::duckdb::StringValue::Get(dataChunk->GetValue(column, row))));
}

template <>
velox::variant variantAt<TypeKind::HUGEINT>(
    ::duckdb::DataChunk* dataChunk,
    int32_t row,
    int32_t column) {
  auto unscaledValue =
      dataChunk->GetValue(column, row).GetValue<::duckdb::hugeint_t>();
  return variant(HugeInt::build(unscaledValue.upper, unscaledValue.lower));
}

template <>
variant variantAt<TypeKind::VARBINARY>(
    ::duckdb::DataChunk* dataChunk,
    int32_t row,
    int32_t column) {
  return variant::binary(
      StringView(::duckdb::StringValue::Get(dataChunk->GetValue(column, row))));
}

template <>
variant variantAt<TypeKind::TIMESTAMP>(
    ::duckdb::DataChunk* dataChunk,
    int32_t row,
    int32_t column) {
  return variant::timestamp(duckdbTimestampToVelox(
      dataChunk->GetValue(column, row).GetValue<::duckdb::timestamp_t>()));
}

template <TypeKind kind>
variant variantAt(const ::duckdb::Value& value) {
  if (value.type() == ::duckdb::LogicalType::INTERVAL) {
    return ::duckdb::Interval::GetMicro(value.GetValue<::duckdb::interval_t>());
  } else if (value.type() == ::duckdb::LogicalType::DATE) {
    return ::duckdb::Date::EpochDays(value.GetValue<::duckdb::date_t>());
  } else {
    // NOTE: duckdb only support native cpp type for GetValue so we need to use
    // DeepCopiedType instead of WrapperType here.
    using T = typename TypeTraits<kind>::DeepCopiedType;
    return variant(value.GetValue<T>());
  }
}

template <>
velox::variant variantAt<TypeKind::HUGEINT>(const ::duckdb::Value& value) {
  auto hugeInt = ::duckdb::HugeIntValue::Get(value);
  return velox::variant(HugeInt::build(hugeInt.upper, hugeInt.lower));
}

template <>
variant variantAt<TypeKind::TIMESTAMP>(const ::duckdb::Value& value) {
  return variant::timestamp(
      duckdbTimestampToVelox(value.GetValue<::duckdb::timestamp_t>()));
}

template <>
variant variantAt<TypeKind::VARCHAR>(const ::duckdb::Value& value) {
  return variant(StringView(::duckdb::StringValue::Get(value)));
}

template <>
variant variantAt<TypeKind::VARBINARY>(const ::duckdb::Value& value) {
  return variant::binary(StringView(::duckdb::StringValue::Get(value)));
}

variant nullVariant(const TypePtr& type) {
  return variant(type->kind());
}

variant rowVariantAt(const ::duckdb::Value& vector, const TypePtr& rowType) {
  std::vector<variant> values;
  const auto& structValue = ::duckdb::StructValue::GetChildren(vector);
  for (size_t i = 0; i < structValue.size(); ++i) {
    auto currChild = structValue[i];
    auto currType = rowType->childAt(i);
    // TODO: Add support for ARRAY and MAP children types.
    if (currChild.IsNull()) {
      values.push_back(nullVariant(currType));
    } else if (currType->kind() == TypeKind::ROW) {
      values.push_back(rowVariantAt(currChild, currType));
    } else {
      auto value = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          variantAt, currType->kind(), currChild);
      values.push_back(value);
    }
  }
  return variant::row(std::move(values));
}

variant mapVariantAt(const ::duckdb::Value& vector, const TypePtr& mapType) {
  std::map<variant, variant> map;

  auto mapTypePtr = dynamic_cast<const MapType*>(mapType.get());
  auto keyType = mapTypePtr->keyType();
  auto valueType = mapTypePtr->valueType();

  VELOX_CHECK_EQ(vector.type().id(), ::duckdb::LogicalTypeId::MAP);

  const auto& valueList = ::duckdb::ListValue::GetChildren(vector);
  for (int i = 0; i < valueList.size(); i++) {
    // TODO: Add support for complex key and value types.
    variant variantKey;
    variant variantValue;
    auto value = ::duckdb::StructValue::GetChildren(valueList[i]);
    // Map key cannot be null.
    VELOX_CHECK(!value[0].IsNull());
    variantKey = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        variantAt, keyType->kind(), value[0]);

    if (value[1].IsNull()) {
      variantValue = nullVariant(valueType);
    } else {
      variantValue = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          variantAt, valueType->kind(), value[1]);
    }
    map.insert({variantKey, variantValue});
  }
  return variant::map(map);
}

variant arrayVariantAt(
    const ::duckdb::Value& vector,
    const TypePtr& arrayType) {
  std::vector<variant> array;

  const auto& elementList = ::duckdb::ListValue::GetChildren(vector);

  auto arrayTypePtr = dynamic_cast<const ArrayType*>(arrayType.get());
  auto elementType = arrayTypePtr->elementType();
  for (int i = 0; i < elementList.size(); i++) {
    // TODO: Add support for MAP and ROW element types.
    if (elementList[i].IsNull()) {
      array.push_back(nullVariant(elementType));
    } else if (elementType->kind() == TypeKind::ARRAY) {
      array.push_back(
          arrayVariantAt(elementList[i], arrayTypePtr->elementType()));
    } else {
      auto variant = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          variantAt, elementType->kind(), elementList[i]);
      array.push_back(variant);
    }
  }
  return variant::array(std::move(array));
}

std::vector<MaterializedRow> materialize(
    ::duckdb::DataChunk* dataChunk,
    const RowTypePtr& rowType) {
  VELOX_CHECK_EQ(
      rowType->size(), dataChunk->GetTypes().size(), "Wrong number of columns");

  auto size = dataChunk->size();
  std::vector<MaterializedRow> rows;
  rows.reserve(size);

  // Pre-compute null values for all columns.
  std::vector<variant> nulls;
  for (size_t j = 0; j < rowType->size(); ++j) {
    nulls.emplace_back(nullVariant(rowType->childAt(j)));
  }

  for (size_t i = 0; i < size; ++i) {
    MaterializedRow row;
    row.reserve(rowType->size());
    for (size_t j = 0; j < rowType->size(); ++j) {
      auto type = rowType->childAt(j);
      auto typeKind = type->kind();
      if (dataChunk->GetValue(j, i).IsNull()) {
        row.push_back(nulls[j]);
      } else if (typeKind == TypeKind::ARRAY) {
        row.push_back(arrayVariantAt(dataChunk->GetValue(j, i), type));
      } else if (typeKind == TypeKind::MAP) {
        row.push_back(mapVariantAt(dataChunk->GetValue(j, i), type));
      } else if (typeKind == TypeKind::ROW) {
        row.push_back(rowVariantAt(dataChunk->GetValue(j, i), type));
      } else if (type->isDecimal()) {
        row.push_back(duckdb::decimalVariant(dataChunk->GetValue(j, i)));
      } else if (type->isIntervalDayTime()) {
        auto value = variant(::duckdb::Interval::GetMicro(
            dataChunk->GetValue(j, i).GetValue<::duckdb::interval_t>()));
        row.push_back(value);
      } else if (type->isDate()) {
        auto value = variant(::duckdb::Date::EpochDays(
            dataChunk->GetValue(j, i).GetValue<::duckdb::date_t>()));
        row.push_back(value);
      } else {
        auto value = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            variantAt, typeKind, dataChunk, i, j);
        row.push_back(value);
      }
    }
    rows.push_back(row);
  }
  return rows;
}

MaterializedRow getColumns(
    const MaterializedRow& row,
    const std::vector<uint32_t>& columnIndices) {
  MaterializedRow columns;
  columns.reserve(columnIndices.size());
  for (auto& index : columnIndices) {
    columns.push_back(row[index]);
  }
  return columns;
}

void printRow(
    const MaterializedRow& row,
    const TypePtr& type,
    std::ostringstream& out) {
  out << row[0].toJson(type->childAt(0));
  for (int32_t i = 1; i < row.size(); ++i) {
    out << " | " << row[i].toJson(type->childAt(i));
  }
}

std::string toString(const MaterializedRow& row, const TypePtr& type) {
  std::ostringstream oss;
  printRow(row, type, oss);
  return oss.str();
}

struct MaterializedRowEpsilonComparator {
 public:
  /// Construct a comparator for MaterializedRows for epsilon comparison.
  /// @param columns A vector of all column indices of the MaterializedRows to
  /// be compared with non-floating-poin column indices coming first, followed
  /// by `numFloatingPointColumns` floating-point column indices.
  MaterializedRowEpsilonComparator(
      uint32_t numFloatingPointColumns,
      const std::vector<column_index_t>& columns)
      : numFloatingPointColumns_{numFloatingPointColumns}, columns_{columns} {
    VELOX_CHECK_GT(numFloatingPointColumns, 0);
    VELOX_CHECK_GE(columns_.size(), numFloatingPointColumns);
  }

  /// Returns the results of comparing 'expected' and 'actual' with epsilon
  /// precision for floating-point columns. Comparing with epsilon is supported
  /// only when non-floating-point columns form unique key in both expected and
  /// actual datasets. Returns std::nullopt if epsilon comparison is not
  /// possible. The caller may fall back on exact comparison in that case.
  std::optional<bool> areEqual(
      const MaterializedRowMultiset& expected,
      const MaterializedRowMultiset& actual);

  /// Returns user-friendly diff message generated by epsilon comparison of rows
  /// sorted by non-floating point columns. Should be called only after
  /// areEqual() returned false.
  std::string getUserFriendlyDiff(const TypePtr& type) const;

 private:
  bool customLessThan(
      const MaterializedRow& lhs,
      const MaterializedRow& rhs,
      bool (*lessThan)(const variant& lhs, const variant& rhs)) const;

  bool lessThanWithEpsilon(
      const MaterializedRow& lhs,
      const MaterializedRow& rhs) const;

  bool equalWithEpsilon(const MaterializedRow& lhs, const MaterializedRow& rhs)
      const;

  // Return true if 'left' and 'right' rows have same values in
  // non-floating-point columns.
  bool equalKeys(const MaterializedRow& left, const MaterializedRow& right)
      const;

  /// Return true if there is only one row with the same values at
  /// non-floating-point columns.
  /// @param sortedRows Rows sorted by non-floating-point columns.
  bool hasUniqueKeys(const std::vector<MaterializedRow>& sortedRows) const;

  // Sorts two lists of rows by non-floating point columns. Returns true if
  // non-floating point columns contain unique combinations of values in both
  // input sets. The sorted results are stored in expectedSorted_ and
  // actualSorted_ in ascending order.
  bool sortByUniqueKey(
      const MaterializedRowMultiset& expected,
      const MaterializedRowMultiset& actual);

  bool notEqual_{false};

  const uint32_t numFloatingPointColumns_;
  const std::vector<column_index_t>& columns_;

  std::vector<MaterializedRow> expectedSorted_;
  std::vector<MaterializedRow> actualSorted_;
};

bool MaterializedRowEpsilonComparator::customLessThan(
    const MaterializedRow& lhs,
    const MaterializedRow& rhs,
    bool (*lessThan)(const variant& lhs, const variant& rhs)) const {
  for (auto i : columns_) {
    if (!lessThan(lhs[i], rhs[i]) && !lessThan(rhs[i], lhs[i])) {
      continue;
    }
    // The 1st non-equal element determines if 'lhs' is smaller or not.
    return lessThan(lhs[i], rhs[i]);
  }
  return lhs.size() < rhs.size();
}

bool MaterializedRowEpsilonComparator::lessThanWithEpsilon(
    const MaterializedRow& lhs,
    const MaterializedRow& rhs) const {
  return customLessThan(lhs, rhs, [](const variant& lhs, const variant& rhs) {
    return lhs.lessThanWithEpsilon(rhs);
  });
}

bool MaterializedRowEpsilonComparator::equalWithEpsilon(
    const MaterializedRow& lhs,
    const MaterializedRow& rhs) const {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  auto size = lhs.size();
  for (auto i = 0; i < size; ++i) {
    if (!lhs[i].equalsWithEpsilon(rhs[i])) {
      return false;
    }
  }
  return true;
}

std::string makeErrorMessage(
    const std::vector<MaterializedRow>& missingRows,
    const std::vector<MaterializedRow>& extraRows,
    size_t expectedSize,
    size_t actualSize,
    const TypePtr& type) {
  std::ostringstream message;
  message << "Expected " << expectedSize << ", got " << actualSize << std::endl;
  message << extraRows.size() << " extra rows, " << missingRows.size()
          << " missing rows" << std::endl;

  auto extraRowsToPrint =
      std::min((size_t)FLAGS_max_error_rows, extraRows.size());
  message << extraRowsToPrint << " of extra rows:" << std::endl;

  for (int32_t i = 0; i < extraRowsToPrint; i++) {
    message << "\t";
    printRow(extraRows[i], type, message);
    message << std::endl;
  }
  message << std::endl;

  auto missingRowsToPrint =
      std::min((size_t)FLAGS_max_error_rows, missingRows.size());
  message << missingRowsToPrint << " of missing rows:" << std::endl;
  for (int32_t i = 0; i < missingRowsToPrint; i++) {
    message << "\t";
    printRow(missingRows[i], type, message);
    message << std::endl;
  }
  message << std::endl;
  return message.str();
}

std::string MaterializedRowEpsilonComparator::getUserFriendlyDiff(
    const TypePtr& type) const {
  VELOX_CHECK(
      notEqual_, "This method must be called after compare() returned false.");

  std::vector<MaterializedRow> extraRows;
  std::vector<MaterializedRow> missingRows;
  int32_t actualIndex = 0;
  int32_t expectedIndex = 0;

  while (expectedIndex < expectedSorted_.size() &&
         actualIndex < actualSorted_.size()) {
    const auto& expectedRow = expectedSorted_[expectedIndex];
    const auto& actualRow = actualSorted_[actualIndex];
    if (equalWithEpsilon(expectedRow, actualRow)) {
      ++expectedIndex;
      ++actualIndex;
    } else if (lessThanWithEpsilon(expectedRow, actualRow)) {
      missingRows.push_back(expectedRow);
      ++expectedIndex;
    } else {
      extraRows.push_back(actualRow);
      ++actualIndex;
    }
  }
  for (; actualIndex < actualSorted_.size(); ++actualIndex) {
    extraRows.push_back(actualSorted_[actualIndex]);
  }
  for (; expectedIndex < expectedSorted_.size(); ++expectedIndex) {
    missingRows.push_back(expectedSorted_[expectedIndex]);
  }

  return makeErrorMessage(
      missingRows,
      extraRows,
      expectedSorted_.size(),
      actualSorted_.size(),
      type);
}

bool MaterializedRowEpsilonComparator::equalKeys(
    const MaterializedRow& left,
    const MaterializedRow& right) const {
  for (auto i = 0; i < columns_.size() - numFloatingPointColumns_; ++i) {
    auto column = columns_[i];
    if (left[column] != right[column]) {
      return false;
    }
  }
  return true;
}

bool MaterializedRowEpsilonComparator::hasUniqueKeys(
    const std::vector<MaterializedRow>& sortedRows) const {
  for (auto i = 1; i < sortedRows.size(); ++i) {
    if (equalKeys(sortedRows[i], sortedRows[i - 1])) {
      return false;
    }
  }
  return true;
}

bool MaterializedRowEpsilonComparator::sortByUniqueKey(
    const MaterializedRowMultiset& expected,
    const MaterializedRowMultiset& actual) {
  std::copy(
      expected.begin(), expected.end(), std::back_inserter(expectedSorted_));
  std::copy(actual.begin(), actual.end(), std::back_inserter(actualSorted_));
  auto lessThan = [&](const MaterializedRow& lhs, const MaterializedRow& rhs) {
    return customLessThan(lhs, rhs, [](const variant& lhs, const variant& rhs) {
      return lhs < rhs;
    });
  };
  std::sort(expectedSorted_.begin(), expectedSorted_.end(), lessThan);
  std::sort(actualSorted_.begin(), actualSorted_.end(), lessThan);

  // Check that every group grouped by non-floating-point columns has only
  // one row.
  return hasUniqueKeys(expectedSorted_) && hasUniqueKeys(actualSorted_);
}

std::string toTypeString(const MaterializedRow& row) {
  std::ostringstream out;
  out << "ROW(";
  const auto numColumns = row.size();
  for (auto i = 0; i < numColumns; ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << mapTypeKindToName(row[i].kind());
  }
  out << ")";
  return out.str();
}

bool equalTypeKinds(const MaterializedRow& left, const MaterializedRow& right) {
  if (left.size() != right.size()) {
    return false;
  }

  const auto numColumns = left.size();
  for (auto i = 0; i < numColumns; ++i) {
    if (left[i].kind() != right[i].kind()) {
      return false;
    }
  }
  return true;
}

std::optional<bool> MaterializedRowEpsilonComparator::areEqual(
    const MaterializedRowMultiset& expected,
    const MaterializedRowMultiset& actual) {
  VELOX_CHECK(!expected.empty());
  VELOX_CHECK(!actual.empty());
  VELOX_CHECK(equalTypeKinds(*expected.begin(), *actual.begin()));

  if (!sortByUniqueKey(expected, actual)) {
    return std::nullopt;
  }

  const auto numColumns = columns_.size();
  const auto size = expectedSorted_.size();
  if (size != actualSorted_.size()) {
    notEqual_ = true;
    return false;
  }

  // Compare row-by-row with epsilon.
  for (auto i = 0; i < size; ++i) {
    for (auto j = 0; j < numColumns; ++j) {
      if (!expectedSorted_[i][j].equalsWithEpsilon(actualSorted_[i][j])) {
        notEqual_ = true;
        return false;
      }
    }
  }
  return true;
}

std::string generateUserFriendlyDiff(
    const MaterializedRowMultiset& expectedRows,
    const MaterializedRowMultiset& actualRows,
    const TypePtr& type) {
  std::vector<MaterializedRow> extraRows;
  std::set_difference(
      actualRows.begin(),
      actualRows.end(),
      expectedRows.begin(),
      expectedRows.end(),
      std::inserter(extraRows, extraRows.end()));

  std::vector<MaterializedRow> missingRows;
  std::set_difference(
      expectedRows.begin(),
      expectedRows.end(),
      actualRows.begin(),
      actualRows.end(),
      std::inserter(missingRows, missingRows.end()));

  return makeErrorMessage(
      missingRows, extraRows, expectedRows.size(), actualRows.size(), type);
}

void verifyDuckDBResult(const DuckDBQueryResult& result, std::string_view sql) {
  VELOX_CHECK(
      !result->HasError(),
      "DuckDB query failed: {}\n{}",
      result->GetError(),
      sql);
}

} // namespace

std::vector<MaterializedRow> materialize(const RowVectorPtr& vector) {
  auto size = vector->size();
  std::vector<MaterializedRow> rows;
  rows.reserve(size);

  auto numColumns = vector->childrenSize();
  std::vector<VectorPtr> simpleVectors(numColumns);

  // variantAt() assumes you can upcast to SimpleVector, so we need to take
  // the inner vector out of lazies first.
  for (size_t i = 0; i < numColumns; ++i) {
    simpleVectors[i] = BaseVector::loadedVectorShared(vector->childAt(i));
  }

  for (size_t i = 0; i < size; ++i) {
    MaterializedRow row;
    row.reserve(numColumns);
    for (size_t j = 0; j < numColumns; ++j) {
      row.push_back(vectorToVariant(simpleVectors[j], i));
    }
    rows.push_back(row);
  }

  return rows;
}

void DuckDbQueryRunner::createTable(
    const std::string& name,
    const std::vector<RowVectorPtr>& data) {
  auto query = fmt::format("DROP TABLE IF EXISTS {}", name);
  execute(query);

  auto& rowType = data[0]->type()->as<TypeKind::ROW>();
  ::duckdb::Connection con(db_);
  auto sql = duckdb::makeCreateTableSql(name, rowType);
  auto res = con.Query(sql);
  verifyDuckDBResult(res, sql);

  ::duckdb::Appender appender(con, name);
  for (auto& vector : data) {
    for (int32_t row = 0; row < vector->size(); row++) {
      appender.BeginRow();
      for (int32_t column = 0; column < rowType.size(); column++) {
        auto columnVector = vector->childAt(column);
        auto type = rowType.childAt(column);
        if (columnVector->isNullAt(row)) {
          appender.Append(nullptr);
        } else if (type->isArray()) {
          appender.Append(duckValueAt<TypeKind::ARRAY>(columnVector, row));
        } else if (type->isMap()) {
          appender.Append(duckValueAt<TypeKind::MAP>(columnVector, row));
        } else if (type->isRow()) {
          appender.Append(duckValueAt<TypeKind::ROW>(columnVector, row));
        } else if (rowType.childAt(column)->isShortDecimal()) {
          appender.Append(duckValueAt<TypeKind::BIGINT>(columnVector, row));
        } else if (rowType.childAt(column)->isLongDecimal()) {
          appender.Append(duckValueAt<TypeKind::HUGEINT>(columnVector, row));
        } else {
          VELOX_CHECK(
              type->equivalent(*columnVector->type()),
              "{} vs. {}",
              type->toString(),
              columnVector->toString());
          auto value = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
              duckValueAt, type->kind(), columnVector, row);
          appender.Append(value);
        }
      }
      appender.EndRow();
    }
  }
}

DuckDBQueryResult DuckDbQueryRunner::execute(const std::string& sql) {
  ::duckdb::Connection con(db_);
  // Changing the default null order of NULLS FIRST used by DuckDB. Velox uses
  // NULLS LAST.
  con.Query("PRAGMA default_null_order='NULLS LAST'");
  auto duckDbResult = con.Query(sql);
  verifyDuckDBResult(duckDbResult, sql);
  return std::move(duckDbResult);
}

void DuckDbQueryRunner::execute(
    const std::string& sql,
    const RowTypePtr& resultRowType,
    std::function<void(std::vector<MaterializedRow>&)> resultCallback) {
  auto duckDbResult = execute(sql);

  for (;;) {
    auto dataChunk = duckDbResult->Fetch();
    verifyDuckDBResult(duckDbResult, sql);
    if (!dataChunk || (dataChunk->size() == 0)) {
      break;
    }
    auto rows = materialize(dataChunk.get(), resultRowType);
    resultCallback(rows);
  }
}

std::shared_ptr<Task> assertQuery(
    const core::PlanNodePtr& plan,
    const std::string& duckDbSql,
    DuckDbQueryRunner& duckDbQueryRunner,
    std::optional<std::vector<uint32_t>> sortingKeys) {
  return assertQuery(
      plan,
      [](TaskCursor* taskCursor) { taskCursor->setNoMoreSplits(); },
      duckDbSql,
      duckDbQueryRunner,
      sortingKeys);
}

std::shared_ptr<Task> assertQueryReturnsEmptyResult(
    const core::PlanNodePtr& plan) {
  CursorParameters params;
  params.planNode = plan;
  auto [cursor, results] = readCursor(params);
  assertEmptyResults(results);
  return cursor->task();
}

std::shared_ptr<Task> assertQueryReturnsEmptyResult(
    const CursorParameters& params) {
  VELOX_DCHECK_NOT_NULL(params.planNode);
  auto [cursor, results] = readCursor(params);
  assertEmptyResults(results);
  return cursor->task();
}

void assertEmptyResults(const std::vector<RowVectorPtr>& results) {
  size_t totalCount = 0;
  for (const auto& vector : results) {
    totalCount += vector->size();
  }
  EXPECT_EQ(0, totalCount) << "Expected empty result but received "
                           << totalCount << " rows";
}

// Compare left and right without epsilon and returns true if they are equal.
static bool compareMaterializedRows(
    const MaterializedRowMultiset& left,
    const MaterializedRowMultiset& right) {
  if (left.size() != right.size()) {
    return false;
  }

  for (auto& it : left) {
    if (right.count(it) != left.count(it)) {
      return false;
    }
  }

  // In case left has duplicate values and there are values in right but not in
  // left, check the other way around. E.g., left = {1, 1, 2}, right = {1, 2,
  // 3}.
  for (auto& it : right) {
    if (left.count(it) != right.count(it)) {
      return false;
    }
  }

  return true;
}

bool assertEqualResults(
    const std::vector<RowVectorPtr>& expected,
    const std::vector<RowVectorPtr>& actual) {
  MaterializedRowMultiset expectedRows;
  for (auto vector : expected) {
    auto rows = materialize(vector);
    std::copy(
        rows.begin(),
        rows.end(),
        std::inserter(expectedRows, expectedRows.end()));
  }
  const auto& expectedType =
      (expected.size() == 0) ? nullptr : expected.at(0)->type();
  return assertEqualResults(expectedRows, expectedType, actual);
}

bool assertEqualResults(
    const core::PlanNodePtr& plan1,
    const core::PlanNodePtr& plan2) {
  CursorParameters params1;
  params1.planNode = plan1;
  auto [cursor1, results1] = readCursor(params1);

  CursorParameters params2;
  params2.planNode = plan2;
  auto [cursor2, results2] = readCursor(params2);
  return assertEqualResults(results1, results2);
}

void assertEqualTypeAndNumRows(
    const TypePtr& expectedType,
    vector_size_t expectedNumRows,
    const std::vector<RowVectorPtr>& actual) {
  size_t actualNumRows = 0;
  for (const auto& result : actual) {
    EXPECT_EQ(*expectedType, *result->type());
    actualNumRows += result->size();
  }
  EXPECT_EQ(expectedNumRows, actualNumRows);
}

bool containsFloatingPoint(const TypePtr& type) {
  if (type->isPrimitiveType()) {
    return type->isReal() || type->isDouble();
  } else if (type->isArray()) {
    return containsFloatingPoint(type->as<TypeKind::ARRAY>().elementType());
  } else if (type->isMap()) {
    // We currently don't support comparing maps with floating-point keys with
    // epsilon. This is because fuzzer can generate floating-point keys that are
    // very close, causing one key incorrectly match another during the
    // comparison.
    return containsFloatingPoint(type->as<TypeKind::MAP>().valueType());
  } else if (type->isRow()) {
    for (auto& child : type->as<TypeKind::ROW>().children()) {
      if (containsFloatingPoint(child)) {
        return true;
      }
    }
  }
  return false;
}

/// Returns the number of floating-point columns and a list of columns indices
/// with floating-point columns placed at the end.
std::tuple<uint32_t, std::vector<velox::column_index_t>>
findFloatingPointColumns(const TypePtr& type) {
  const auto rowType = asRowType(type);
  uint32_t numFloatingPointColumns = 0;
  std::vector<velox::column_index_t> indices;
  for (auto i = 0; i < rowType->children().size(); ++i) {
    if (containsFloatingPoint(rowType->childAt(i))) {
      ++numFloatingPointColumns;
    } else {
      indices.push_back(i);
    }
  }

  for (auto i = 0; i < rowType->children().size(); ++i) {
    if (containsFloatingPoint(rowType->childAt(i))) {
      indices.push_back(i);
    }
  }
  return std::make_tuple(numFloatingPointColumns, indices);
}

// Compare actualRows with expectedRows and return whether they match. Compare
// actualRows and expectedRows with epsilon if needed and allowed. Otherwise,
// compare their values directly. The underlying assumption is that aggregation
// results can be sorted by unique keys and floating-point values in them are
// computed in different ways and hence require epsilon comparison. For results
// of other operations, floating-point values are likely copied from inputs and
// hence can be compared directly.
bool assertEqualResults(
    const MaterializedRowMultiset& expectedRows,
    const TypePtr& expectedType,
    const MaterializedRowMultiset& actualRows,
    const TypePtr& actualType,
    const std::string& message) {
  const auto& type = (!expectedRows.empty()) ? expectedType : actualType;
  if (expectedRows.empty() != actualRows.empty()) {
    ADD_FAILURE() << generateUserFriendlyDiff(expectedRows, actualRows, type)
                  << message;
    return false;
  }

  if (expectedRows.empty()) {
    return true;
  }

  if (!equalTypeKinds(*expectedRows.begin(), *actualRows.begin())) {
    ADD_FAILURE() << "Types of expected and actual results do not match: "
                  << toTypeString(*expectedRows.begin()) << " vs. "
                  << toTypeString(*actualRows.begin());
    return false;
  }

  auto [numFloatingPointColumns, columns] = findFloatingPointColumns(type);
  if (numFloatingPointColumns) {
    MaterializedRowEpsilonComparator comparator{
        numFloatingPointColumns, columns};
    if (auto result = comparator.areEqual(expectedRows, actualRows)) {
      if (!result.value()) {
        ADD_FAILURE() << comparator.getUserFriendlyDiff(expectedType)
                      << message;
        return false;
      }
      return true;
    }
  }

  // Compare the results directly without epsilon. This may cause false alarm
  // if there are floating-point columns that are computed during the
  // evaluation.
  if (not compareMaterializedRows(expectedRows, actualRows)) {
    std::string note = numFloatingPointColumns > 0
        ? "\nNote: results are compared without epsilon because values at non-floating-point columns do not form unique keys."
        : "";
    ADD_FAILURE() << generateUserFriendlyDiff(
                         expectedRows, actualRows, expectedType)
                  << message << note;
    return false;
  }
  return true;
}

MaterializedRowMultiset materialize(const std::vector<RowVectorPtr>& vectors) {
  MaterializedRowMultiset materialized;
  for (auto vector : vectors) {
    auto rows = materialize(vector);
    std::copy(
        rows.begin(),
        rows.end(),
        std::inserter(materialized, materialized.end()));
  }
  return materialized;
}

bool assertEqualResults(
    const MaterializedRowMultiset& expectedRows,
    const TypePtr& expectedType,
    const std::vector<RowVectorPtr>& actual) {
  const auto& actualType =
      (actual.size() == 0) ? nullptr : actual.at(0)->type();
  return assertEqualResults(
      expectedRows,
      expectedType,
      materialize(actual),
      actualType,
      "Unexpected results");
}

void assertResults(
    const std::vector<RowVectorPtr>& results,
    const RowTypePtr& resultType,
    const std::string& duckDbSql,
    DuckDbQueryRunner& duckDbQueryRunner) {
  MaterializedRowMultiset actualRows;
  MaterializedRowMultiset expectedRows;
  for (const auto& vector : results) {
    auto rows = materialize(vector);
    std::copy(
        rows.begin(), rows.end(), std::inserter(actualRows, actualRows.end()));
  }

  if (!duckDbSql.empty()) {
    expectedRows = duckDbQueryRunner.execute(duckDbSql, resultType);
  }
  const auto& actualType =
      (results.size() == 0) ? nullptr : results.at(0)->type();
  assertEqualResults(
      expectedRows,
      resultType,
      actualRows,
      actualType,
      kDuckDbTimestampWarning + "\nDuckDB query: " + duckDbSql);
}

// To handle the case when the sorting keys are not unique and the order
// within the same key is not deterministic, we partition the results with
// sorting keys and verify the partitions are equal using set-diff.
using OrderedPartition = std::pair<MaterializedRow, MaterializedRowMultiset>;

// Special function to compare ordered partitions in a way that
// we compare all floating point values inside using 'epsilon' constant.
// Returns true if equal. valueType is the type of expected.second.
static bool compareOrderedPartitions(
    const OrderedPartition& expected,
    const OrderedPartition& actual,
    const RowTypePtr& valueType) {
  if (expected.first.size() != actual.first.size() or
      expected.second.size() != actual.second.size()) {
    return false;
  }

  for (size_t i = 0; i < expected.first.size(); ++i) {
    if (not expected.first[i].equalsWithEpsilon(actual.first[i])) {
      return false;
    }
  }

  if (expected.second.empty()) {
    return true;
  }

  if (!equalTypeKinds(*expected.second.begin(), *actual.second.begin())) {
    ADD_FAILURE() << "Types of expected and actual results do not match";
    return false;
  }

  // valueType is needed by findFloatingPointColumns() to avoid having to infer
  // the type from expected.second.
  auto [numFloatingPointColumns, columns] = findFloatingPointColumns(valueType);
  if (numFloatingPointColumns) {
    MaterializedRowEpsilonComparator comparator{
        numFloatingPointColumns, columns};
    if (auto result = comparator.areEqual(expected.second, actual.second)) {
      return result.value();
    }
  }
  // Compare the results directly without epsilon. This may cause false alarm
  // if there are floating-point columns that are computed during the
  // evaluation.
  return compareMaterializedRows(expected.second, actual.second);
}

// Special function to compare vectors of ordered partitions in a way that
// we compare all floating point values inside using 'epsilon' constant.
// Returns true if equal. valueType is the type of expected[i].second.
static bool compareOrderedPartitionsVectors(
    const std::vector<OrderedPartition>& expected,
    const std::vector<OrderedPartition>& actual,
    const RowTypePtr& valueType) {
  if (expected.size() != actual.size()) {
    return false;
  }

  for (size_t i = 0; i < expected.size(); ++i) {
    if (not compareOrderedPartitions(expected[i], actual[i], valueType)) {
      return false;
    }
  }

  return true;
}

void assertResultsOrdered(
    const std::vector<RowVectorPtr>& results,
    const RowTypePtr& resultType,
    const std::string& duckDbSql,
    DuckDbQueryRunner& duckDbQueryRunner,
    const std::vector<uint32_t>& sortingKeys) {
  std::vector<OrderedPartition> actualPartitions;
  std::vector<OrderedPartition> expectedPartitions;

  for (const auto& vector : results) {
    auto rows = materialize(vector);
    for (const auto& row : rows) {
      auto keys = getColumns(row, sortingKeys);
      if (actualPartitions.empty() || actualPartitions.back().first != keys) {
        actualPartitions.emplace_back(keys, MaterializedRowMultiset());
      }
      actualPartitions.back().second.insert(row);
    }
  }

  auto expectedRows = duckDbQueryRunner.executeOrdered(duckDbSql, resultType);
  for (const auto& row : expectedRows) {
    auto keys = getColumns(row, sortingKeys);
    if (expectedPartitions.empty() || expectedPartitions.back().first != keys) {
      expectedPartitions.emplace_back(keys, MaterializedRowMultiset());
    }
    expectedPartitions.back().second.insert(row);
  }

  if (not compareOrderedPartitionsVectors(
          expectedPartitions, actualPartitions, resultType)) {
    auto actualPartIter = actualPartitions.begin();
    auto expectedPartIter = expectedPartitions.begin();
    while (expectedPartIter != expectedPartitions.end() &&
           actualPartIter != actualPartitions.end()) {
      if (not compareOrderedPartitions(
              *expectedPartIter, *actualPartIter, resultType)) {
        break;
      }
      ++expectedPartIter;
      ++actualPartIter;
    }
    std::ostringstream oss;
    if (expectedPartIter == expectedPartitions.end()) {
      oss << "Got extra rows: keys: "
          << toString(actualPartIter->first, resultType) << std::endl;
    } else if (actualPartIter == actualPartitions.end()) {
      oss << "Missing rows: keys: "
          << toString(expectedPartIter->first, resultType) << std::endl;
    } else {
      if (actualPartIter->first != expectedPartIter->first) {
        oss << "Expected keys: "
            << toString(expectedPartIter->first, resultType)
            << ", actual: " << toString(actualPartIter->first, resultType)
            << std::endl;
      } else {
        oss << "Keys: " << toString(expectedPartIter->first, resultType) << " ";
        oss << generateUserFriendlyDiff(
            expectedPartIter->second, actualPartIter->second, resultType);
      }
      ADD_FAILURE() << oss.str() << kDuckDbTimestampWarning
                    << "\nDuckDB query: " << duckDbSql;
    }
  }
}

tsan_atomic<int32_t>& testingAbortPct() {
  static tsan_atomic<int32_t> abortPct = 0;
  return abortPct;
}

tsan_atomic<int32_t>& testingAbortCounter() {
  static tsan_atomic<int32_t> counter = 0;
  return counter;
}

std::function<void(Task*)>& testingAbortHook() {
  static std::function<void(Task*)> hook = nullptr;
  return hook;
}

TestScopedAbortInjection::TestScopedAbortInjection(
    int32_t abortPct,
    int32_t maxInjections,
    std::function<void(Task*)> hook) {
  testingAbortPct() = abortPct;
  testingAbortCounter() = maxInjections;
  testingAbortHook() = hook;
}

TestScopedAbortInjection::~TestScopedAbortInjection() {
  testingAbortPct() = 0;
  testingAbortCounter() = 0;
  testingAbortHook() = nullptr;
}

bool testingMaybeTriggerAbort(exec::Task* task) {
  if (testingAbortPct() <= 0 || testingAbortCounter() <= 0) {
    return false;
  }

  if ((folly::Random::rand32() % 100) < testingAbortPct()) {
    if (testingAbortCounter()-- > 0) {
      if (testingAbortHook() != nullptr) {
        testingAbortHook()(task);
      }
      task->requestAbort();
      return true;
    }
  }

  return false;
}

std::pair<std::unique_ptr<TaskCursor>, std::vector<RowVectorPtr>> readCursor(
    const CursorParameters& params,
    std::function<void(TaskCursor*)> addSplits,
    uint64_t maxWaitMicros) {
  auto cursor = TaskCursor::create(params);
  // 'result' borrows memory from cursor so the life cycle must be shorter.
  std::vector<RowVectorPtr> result;
  auto* task = cursor->task().get();
  while (!cursor->noMoreSplits()) {
    addSplits(cursor.get());
    while (cursor->moveNext()) {
      auto vector = cursor->current();
      vector->loadedVector();
      result.push_back(std::move(vector));
      testingMaybeTriggerAbort(task);
    }
  }

  if (!waitForTaskCompletion(task, maxWaitMicros)) {
    // NOTE: there is async memory arbitration might fail the task after all the
    // results have been consumed and before the task finishes. So we might run
    // into the failed task state in some rare case such as exposed by
    // concurrent memory arbitration test.
    if (task->state() != TaskState::kFinished &&
        task->state() != TaskState::kRunning) {
      waitForTaskDriversToFinish(task, maxWaitMicros);
      std::rethrow_exception(task->error());
    } else {
      VELOX_FAIL(
          "Failed to wait for task to complete after {}, task: {}",
          succinctMicros(maxWaitMicros),
          task->toString());
    }
  }
  return {std::move(cursor), std::move(result)};
}

bool waitForTaskFinish(
    exec::Task* task,
    TaskState expectedState,
    uint64_t maxWaitMicros) {
  // Wait for task to transition to finished state.
  if (!waitForTaskStateChange(task, expectedState, maxWaitMicros)) {
    return false;
  }
  return waitForTaskDriversToFinish(task, maxWaitMicros);
}

bool waitForTaskCompletion(exec::Task* task, uint64_t maxWaitMicros) {
  return waitForTaskFinish(task, TaskState::kFinished, maxWaitMicros);
}

bool waitForTaskFailure(exec::Task* task, uint64_t maxWaitMicros) {
  return waitForTaskFinish(task, TaskState::kFailed, maxWaitMicros);
}

bool waitForTaskAborted(exec::Task* task, uint64_t maxWaitMicros) {
  return waitForTaskFinish(task, TaskState::kAborted, maxWaitMicros);
}

bool waitForTaskCancelled(exec::Task* task, uint64_t maxWaitMicros) {
  return waitForTaskFinish(task, TaskState::kCanceled, maxWaitMicros);
}

bool waitForTaskStateChange(
    exec::Task* task,
    TaskState state,
    uint64_t maxWaitMicros) {
  // Wait for task to transition to finished state.
  if (task->state() != state) {
    auto& executor = folly::QueuedImmediateExecutor::instance();
    auto future = task->taskCompletionFuture()
                      .within(std::chrono::microseconds(maxWaitMicros))
                      .via(&executor);
    future.wait();
  }

  return task->state() == state;
}

void waitForAllTasksToBeDeleted(uint64_t maxWaitUs) {
  uint64_t waitUs = 0;
  while (Task::numRunningTasks() != 0) {
    constexpr uint64_t kWaitInternalUs = 1'000;
    std::this_thread::sleep_for(std::chrono::microseconds(kWaitInternalUs));
    waitUs += kWaitInternalUs;
    if (waitUs >= maxWaitUs) {
      break;
    }
  }
  std::vector<std::shared_ptr<Task>> pendingTasks = Task::getRunningTasks();
  if (pendingTasks.empty()) {
    return;
  }
  std::vector<std::string> pendingTaskStats;
  pendingTaskStats.reserve(pendingTasks.size());
  for (const auto& task : pendingTasks) {
    pendingTaskStats.push_back(task->toString());
  }
  VELOX_FAIL(
      "{} pending tasks\n{}",
      pendingTasks.size(),
      folly::join("\n", pendingTaskStats));
}

std::shared_ptr<Task> assertQuery(
    const core::PlanNodePtr& plan,
    std::function<void(exec::TaskCursor*)> addSplits,
    const std::string& duckDbSql,
    DuckDbQueryRunner& duckDbQueryRunner,
    std::optional<std::vector<uint32_t>> sortingKeys) {
  CursorParameters params;
  params.planNode = plan;
  return assertQuery(
      params, addSplits, duckDbSql, duckDbQueryRunner, sortingKeys);
}

std::shared_ptr<Task> assertQuery(
    const CursorParameters& params,
    std::function<void(TaskCursor*)> addSplits,
    const std::string& duckDbSql,
    DuckDbQueryRunner& duckDbQueryRunner,
    std::optional<std::vector<uint32_t>> sortingKeys) {
  auto [cursor, actualResults] = readCursor(params, addSplits);

  if (sortingKeys) {
    assertResultsOrdered(
        actualResults,
        params.planNode->outputType(),
        duckDbSql,
        duckDbQueryRunner,
        sortingKeys.value());
  } else {
    assertResults(
        actualResults,
        params.planNode->outputType(),
        duckDbSql,
        duckDbQueryRunner);
  }
  auto task = cursor->task();

  EXPECT_TRUE(waitForTaskCompletion(task.get())) << task->taskId();
  return task;
}

std::shared_ptr<Task> assertQuery(
    const core::PlanNodePtr& plan,
    const std::vector<RowVectorPtr>& expectedResults) {
  CursorParameters params;
  params.planNode = plan;
  return assertQuery(params, expectedResults);
}

std::shared_ptr<Task> assertQuery(
    const CursorParameters& params,
    const std::vector<RowVectorPtr>& expectedResults) {
  auto result = readCursor(params);

  assertEqualResults(expectedResults, result.second);
  return result.first->task();
}

variant readSingleValue(const core::PlanNodePtr& plan, int32_t maxDrivers) {
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = maxDrivers;
  auto result = readCursor(params);

  EXPECT_EQ(1, result.second.size());
  EXPECT_EQ(1, result.second[0]->size());
  EXPECT_EQ(
      *plan->outputType()->childAt(0), *(result.second[0]->type()->childAt(0)));
  return materialize(result.second[0])[0][0];
}

void printResults(const RowVectorPtr& result, std::ostream& out) {
  auto materializedRows = materialize(result);
  const auto& type = result->type();
  for (const auto& row : materializedRows) {
    out << toString(row, type) << std::endl;
  }
}

std::unordered_map<std::string, OperatorStats> toOperatorStats(
    const TaskStats& taskStats) {
  std::unordered_map<std::string, OperatorStats> opStatsMap;

  for (const auto& pipelineStats : taskStats.pipelineStats) {
    for (const auto& opStats : pipelineStats.operatorStats) {
      const auto& opType = opStats.operatorType;
      auto it = opStatsMap.find(opType);
      if (it != opStatsMap.end()) {
        it->second.add(opStats);
      } else {
        opStatsMap.emplace(opType, opStats);
      }
    }
  }
  return opStatsMap;
}

} // namespace facebook::velox::exec::test

template <>
struct fmt::formatter<::duckdb::LogicalTypeId> : formatter<int> {
  auto format(::duckdb::LogicalTypeId s, format_context& ctx) const {
    return formatter<int>::format(static_cast<int>(s), ctx);
  }
};

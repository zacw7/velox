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
#include "velox/vector/DecodedVector.h"
#include "velox/buffer/Buffer.h"
#include "velox/common/base/BitUtil.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/LazyVector.h"

namespace facebook::velox {

uint64_t DecodedVector::constantNullMask_{0};

namespace {

std::vector<vector_size_t> makeConsecutiveIndices(size_t size) {
  std::vector<vector_size_t> consecutiveIndices(size);
  for (vector_size_t i = 0; i < consecutiveIndices.size(); ++i) {
    consecutiveIndices[i] = i;
  }
  return consecutiveIndices;
}

const VectorPtr& getLoadedVector(const VectorPtr& vector) {
  return BaseVector::loadedVectorShared(vector);
}

const BaseVector* getLoadedVector(const BaseVector* vector) {
  return vector->loadedVector();
}

const VectorPtr& getValueVector(const VectorPtr& vector) {
  return vector->valueVector();
}

const BaseVector* getValueVector(const BaseVector* vector) {
  return vector->valueVector().get();
}

} // namespace

const std::vector<vector_size_t>& DecodedVector::consecutiveIndices() {
  static std::vector<vector_size_t> consecutiveIndices =
      makeConsecutiveIndices(10'000);
  return consecutiveIndices;
}

const std::vector<vector_size_t>& DecodedVector::zeroIndices() {
  static std::vector<vector_size_t> indices(10'000);
  return indices;
}

template <typename T>
VectorPtr DecodedVector::decodeImpl(
    const T& vector,
    const SelectivityVector* rows,
    bool loadLazy) {
  reset(end(vector->size(), rows));
  partialRowsDecoded_ = rows != nullptr;
  loadLazy_ = loadLazy;
  const bool isTopLevelLazyAndLoaded = vector->isLazy() &&
      vector->template asUnchecked<LazyVector>()->isLoaded();
  if (isTopLevelLazyAndLoaded || (loadLazy_ && isLazyNotLoaded(*vector))) {
    return decodeImpl(getLoadedVector(vector), rows, loadLazy);
  }

  VectorPtr sharedBase;
  const auto encoding = vector->encoding();
  switch (encoding) {
    case VectorEncoding::Simple::FLAT:
    case VectorEncoding::Simple::BIASED:
    case VectorEncoding::Simple::ROW:
    case VectorEncoding::Simple::ARRAY:
    case VectorEncoding::Simple::FLAT_MAP:
    case VectorEncoding::Simple::MAP:
    case VectorEncoding::Simple::LAZY:
      isIdentityMapping_ = true;
      setBaseData(vector, rows, sharedBase);
      break;
    case VectorEncoding::Simple::CONSTANT: {
      isConstantMapping_ = true;
      if (isLazyNotLoaded(*vector)) {
        if constexpr (std::is_same_v<T, VectorPtr>) {
          sharedBase = vector->valueVector();
        }
        baseVector_ = vector->valueVector().get();
        constantIndex_ = vector->wrapInfo()->template as<vector_size_t>()[0];
        mayHaveNulls_ = true;
      } else {
        setBaseData(vector, rows, sharedBase);
      }
      break;
    }
    case VectorEncoding::Simple::DICTIONARY:
    case VectorEncoding::Simple::SEQUENCE: {
      combineWrappers(vector, rows, sharedBase);
      break;
    }
    default:
      VELOX_FAIL(
          "Unsupported vector encoding: {}",
          VectorEncoding::mapSimpleToName(encoding));
  }
  return sharedBase;
}

DecodedVector::DecodedVector(
    const BaseVector& vector,
    const SelectivityVector& rows,
    bool loadLazy) {
  decodeImpl(&vector, &rows, loadLazy);
}

DecodedVector::DecodedVector(const BaseVector& vector, bool loadLazy) {
  decodeImpl(&vector, nullptr, loadLazy);
}

void DecodedVector::decode(
    const BaseVector& vector,
    const SelectivityVector& rows,
    bool loadLazy) {
  decodeImpl(&vector, &rows, loadLazy);
}

void DecodedVector::decode(const BaseVector& vector, bool loadLazy) {
  decodeImpl(&vector, nullptr, loadLazy);
}

VectorPtr DecodedVector::decodeAndGetBase(
    const VectorPtr& vector,
    bool loadLazy) {
  auto sharedBase = decodeImpl(vector, nullptr, loadLazy);
  VELOX_CHECK(sharedBase.get() == baseVector_);
  return sharedBase;
}

void DecodedVector::makeIndices(
    const BaseVector& vector,
    const SelectivityVector* rows,
    int32_t numLevels) {
  if (rows) {
    VELOX_CHECK_LE(rows->end(), vector.size());
  }

  reset(end(vector.size(), rows));
  VectorPtr sharedPtr;
  combineWrappers(&vector, rows, sharedPtr, numLevels);
}

void DecodedVector::reset(vector_size_t size) {
  if (!indicesNotCopied()) {
    // Init with default value to avoid invalid indices for unselected rows)
    std::fill(copiedIndices_.begin(), copiedIndices_.end(), 0);
  }
  size_ = size;
  indices_ = nullptr;
  data_ = nullptr;
  nulls_ = nullptr;
  allNulls_.reset();
  baseVector_ = nullptr;
  mayHaveNulls_ = false;
  hasExtraNulls_ = false;
  isConstantMapping_ = false;
  isIdentityMapping_ = false;
  constantIndex_ = 0;
}

void DecodedVector::copyNulls(vector_size_t size) {
  auto numWords = bits::nwords(size);
  copiedNulls_.resize(numWords > 0 ? numWords : 1);
  if (nulls_) {
    std::copy(nulls_, nulls_ + numWords, copiedNulls_.data());
  } else {
    std::fill(copiedNulls_.begin(), copiedNulls_.end(), bits::kNotNull64);
  }
  nulls_ = copiedNulls_.data();
}

template <typename T>
void DecodedVector::combineWrappers(
    const T& vector,
    const SelectivityVector* rows,
    VectorPtr& sharedBase,
    int numLevels) {
  auto topEncoding = vector->encoding();
  T values;
  if (topEncoding == VectorEncoding::Simple::DICTIONARY) {
    indices_ = vector->wrapInfo()->template as<vector_size_t>();
    values = getValueVector(vector);
    nulls_ = vector->rawNulls();
    if (nulls_) {
      hasExtraNulls_ = true;
      mayHaveNulls_ = true;
    }
  } else {
    VELOX_FAIL(
        "Unsupported wrapper encoding: {}",
        VectorEncoding::mapSimpleToName(topEncoding));
  }
  int32_t levelCounter = 0;
  for (;;) {
    if (numLevels != -1 && ++levelCounter == numLevels) {
      if constexpr (std::is_same_v<T, VectorPtr>) {
        // We get the shared base vector only in case numLevels == -1.
        VELOX_UNREACHABLE();
      } else {
        baseVector_ = values;
      }
      return;
    }

    auto encoding = values->encoding();
    if (isLazy(encoding) &&
        (loadLazy_ || values->template asUnchecked<LazyVector>()->isLoaded())) {
      values = getLoadedVector(values);
      encoding = values->encoding();
    }

    switch (encoding) {
      case VectorEncoding::Simple::LAZY:
      case VectorEncoding::Simple::CONSTANT:
      case VectorEncoding::Simple::FLAT:
      case VectorEncoding::Simple::BIASED:
      case VectorEncoding::Simple::ROW:
      case VectorEncoding::Simple::ARRAY:
      case VectorEncoding::Simple::FLAT_MAP:
      case VectorEncoding::Simple::MAP:
        setBaseData(values, rows, sharedBase);
        return;
      case VectorEncoding::Simple::DICTIONARY:
        applyDictionaryWrapper(*values, rows);
        values = getValueVector(values);
        break;
      default:
        VELOX_CHECK(false, "Unsupported vector encoding");
    }
  }
}

void DecodedVector::applyDictionaryWrapper(
    const BaseVector& dictionaryVector,
    const SelectivityVector* rows) {
  if (size_ == 0 || (rows && !rows->hasSelections())) {
    // No further processing is needed.
    return;
  }

  auto newIndices = dictionaryVector.wrapInfo()->as<vector_size_t>();
  auto newNulls = dictionaryVector.rawNulls();
  if (newNulls) {
    hasExtraNulls_ = true;
    mayHaveNulls_ = true;
    // if we have both nulls for parent and the wrapped vectors, and nulls
    // buffer is not copied, make a copy because we may need to
    // change it when iterating through wrapped vector
    if (!nulls_ || nullsNotCopied()) {
      copyNulls(end(rows));
    }
  }
  auto copiedNulls = copiedNulls_.data();
  auto currentIndices = indices_;
  if (indicesNotCopied()) {
    copiedIndices_.resize(size_);
    indices_ = copiedIndices_.data();
  }

  applyToRows(rows, [&](vector_size_t row) {
    if (!nulls_ || !bits::isBitNull(nulls_, row)) {
      auto wrappedIndex = currentIndices[row];
      if (newNulls && bits::isBitNull(newNulls, wrappedIndex)) {
        bits::setNull(copiedNulls, row);
      } else {
        copiedIndices_[row] = newIndices[wrappedIndex];
      }
    }
  });
}

void DecodedVector::fillInIndices() const {
  if (isConstantMapping_) {
    if (size_ > zeroIndices().size() || constantIndex_ != 0) {
      copiedIndices_.resize(size_);
      std::fill(copiedIndices_.begin(), copiedIndices_.end(), constantIndex_);
      indices_ = copiedIndices_.data();
    } else {
      indices_ = zeroIndices().data();
    }
    return;
  }
  if (isIdentityMapping_) {
    if (size_ > consecutiveIndices().size()) {
      copiedIndices_.resize(size_);
      std::iota(copiedIndices_.begin(), copiedIndices_.end(), 0);
      indices_ = &copiedIndices_[0];
    } else {
      indices_ = consecutiveIndices().data();
    }
    return;
  }
  VELOX_FAIL(
      "DecodedVector::indices_ must be set for non-constant non-consecutive mapping.");
}

void DecodedVector::makeIndicesMutable() {
  if (indicesNotCopied()) {
    copiedIndices_.resize(size_ > 0 ? size_ : 1);
    memcpy(
        &copiedIndices_[0],
        indices_,
        copiedIndices_.size() * sizeof(copiedIndices_[0]));
    indices_ = &copiedIndices_[0];
  }
}

void DecodedVector::setFlatNulls(
    const BaseVector& vector,
    const SelectivityVector* rows) {
  if (hasExtraNulls_) {
    if (nullsNotCopied()) {
      copyNulls(end(rows));
    }
    auto leafNulls = vector.rawNulls();
    auto copiedNulls = &copiedNulls_[0];
    applyToRows(rows, [&](vector_size_t row) {
      if (!bits::isBitNull(nulls_, row) &&
          (leafNulls && bits::isBitNull(leafNulls, indices_[row]))) {
        bits::setNull(copiedNulls, row);
      }
    });
    nulls_ = &copiedNulls_[0];
  } else {
    nulls_ = vector.rawNulls();
    mayHaveNulls_ = nulls_ != nullptr;
  }
}

template <typename T>
void DecodedVector::setBaseData(
    const T& vector,
    const SelectivityVector* rows,
    VectorPtr& sharedBase) {
  auto encoding = vector->encoding();
  if constexpr (std::is_same_v<T, VectorPtr>) {
    sharedBase = vector;
    baseVector_ = vector.get();
  } else {
    baseVector_ = vector;
  }
  switch (encoding) {
    case VectorEncoding::Simple::LAZY:
      break;
    case VectorEncoding::Simple::FLAT:
      // values() may be nullptr if 'vector' is all nulls.
      data_ =
          vector->values() ? vector->values()->template as<void>() : nullptr;
      setFlatNulls(*vector, rows);
      break;
    case VectorEncoding::Simple::ROW:
    case VectorEncoding::Simple::ARRAY:
    case VectorEncoding::Simple::FLAT_MAP:
    case VectorEncoding::Simple::MAP:
      setFlatNulls(*vector, rows);
      break;
    case VectorEncoding::Simple::CONSTANT:
      setBaseDataForConstant(vector, rows, sharedBase);
      break;
    default:
      VELOX_UNREACHABLE();
  }
}

template <typename T>
void DecodedVector::setBaseDataForConstant(
    const T& vector,
    const SelectivityVector* rows,
    VectorPtr& sharedBase) {
  if (!vector->isScalar()) {
    if constexpr (std::is_same_v<T, VectorPtr>) {
      sharedBase = BaseVector::wrappedVectorShared(vector);
      baseVector_ = sharedBase.get();
    } else {
      baseVector_ = vector->wrappedVector();
    }
    constantIndex_ = vector->wrappedIndex(0);
  }
  if (!hasExtraNulls_ || vector->isNullAt(0)) {
    // A mapping over a constant is constant except if the
    // mapping adds nulls and the constant is not null.
    isConstantMapping_ = true;
    hasExtraNulls_ = false;
    indices_ = nullptr;
    nulls_ = vector->isNullAt(0) ? &constantNullMask_ : nullptr;
  } else {
    makeIndicesMutable();

    applyToRows(rows, [this](vector_size_t row) {
      copiedIndices_[row] = constantIndex_;
    });
    setFlatNulls(*vector, rows);
  }
  data_ = vector->valuesAsVoid();
  if (!nulls_) {
    nulls_ = vector->isNullAt(0) ? &constantNullMask_ : nullptr;
  }
  mayHaveNulls_ = hasExtraNulls_ || nulls_;
}

namespace {

/// Copies 'size' entries from 'indices' into a newly allocated buffer.
BufferPtr copyIndicesBuffer(
    const vector_size_t* indices,
    vector_size_t size,
    memory::MemoryPool* pool) {
  BufferPtr copy = AlignedBuffer::allocate<vector_size_t>(size, pool);
  memcpy(
      copy->asMutable<vector_size_t>(),
      indices,
      BaseVector::byteSize<vector_size_t>(size));
  return copy;
}

/// Copies 'size' bits from 'nulls' into a newly allocated buffer. Returns
/// nullptr if 'nulls' is null.
BufferPtr copyNullsBuffer(
    const uint64_t* nulls,
    vector_size_t size,
    memory::MemoryPool* pool) {
  if (!nulls) {
    return nullptr;
  }

  BufferPtr copy = AlignedBuffer::allocate<bool>(size, pool);
  memcpy(copy->asMutable<uint64_t>(), nulls, BaseVector::byteSize<bool>(size));
  return copy;
}
} // namespace

DecodedVector::DictionaryWrapping DecodedVector::dictionaryWrapping(
    memory::MemoryPool& pool,
    vector_size_t size) const {
  VELOX_CHECK_LE(size, size_);

  // Make a copy of the indices and nulls buffers.
  BufferPtr indices = copyIndicesBuffer(this->indices(), size, &pool);
  // Only copy nulls if we have nulls coming from one of the wrappers, don't
  // do it if nulls are missing or from the base vector.
  // TODO: remove the check for hasExtraNulls_ after #3553 is merged.
  BufferPtr nulls =
      hasExtraNulls_ ? copyNullsBuffer(nulls_, size, &pool) : nullptr;
  return {std::move(indices), std::move(nulls)};
}

VectorPtr DecodedVector::wrap(
    VectorPtr data,
    memory::MemoryPool& pool,
    vector_size_t size) {
  if (isConstantMapping_) {
    if (isNullAt(0)) {
      return BaseVector::createNullConstant(data->type(), size, data->pool());
    } else if (data->isConstantEncoding() && size == data->size()) {
      // Return `data` as is if it is constant encoded and the vector size
      // matches exactly with the selection size. Otherwise, the constant vector
      // will need to be resized to match it.
      return data;
    }
    return BaseVector::wrapInConstant(size, constantIndex_, data);
  }

  auto wrapping = dictionaryWrapping(pool, size);
  return BaseVector::wrapInDictionary(
      std::move(wrapping.nulls),
      std::move(wrapping.indices),
      size,
      std::move(data));
}

const uint64_t* DecodedVector::nulls(const SelectivityVector* rows) {
  if (allNulls_.has_value()) {
    return allNulls_.value();
  }

  if (hasExtraNulls_) {
    allNulls_ = nulls_;
  } else if (!nulls_ || size_ == 0) {
    allNulls_ = nullptr;
  } else {
    if (isIdentityMapping_) {
      allNulls_ = nulls_;
    } else if (isConstantMapping_) {
      copiedNulls_.resize(0);
      copiedNulls_.resize(bits::nwords(size_), bits::kNull64);
      allNulls_ = copiedNulls_.data();
    } else {
      // Copy base nulls.
      copiedNulls_.resize(bits::nwords(size_));
      auto* rawCopiedNulls = copiedNulls_.data();
      VELOX_CHECK(
          partialRowsDecoded_ == (rows != nullptr),
          "DecodedVector::nulls() must be called with the same rows as decode()");
      if (rows != nullptr) {
        // Partial consistency check: The end may be less than the decode time
        // end but not greater.
        VELOX_CHECK_LE(rows->end(), size_);
      }
      VELOX_DEBUG_ONLY const auto baseSize = baseVector_->size();
      applyToRows(rows, [&](auto i) {
        VELOX_DCHECK_LT(indices_[i], baseSize);
        bits::setNull(rawCopiedNulls, i, bits::isBitNull(nulls_, indices_[i]));
      });
      allNulls_ = copiedNulls_.data();
    }
  }

  return allNulls_.value();
}

template <typename Func>
void DecodedVector::applyToRows(const SelectivityVector* rows, Func&& func)
    const {
  if (rows) {
    rows->applyToSelected([&](vector_size_t row) { func(row); });
  } else {
    for (auto i = 0; i < size_; i++) {
      func(i);
    }
  }
}

std::string DecodedVector::toString(vector_size_t idx) const {
  if (isNullAt(idx)) {
    return "null";
  }

  return baseVector_->toString(index(idx));
}
} // namespace facebook::velox

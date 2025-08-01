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

#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kDoubleMax = std::numeric_limits<double>::max();
constexpr double kDoubleMin = std::numeric_limits<double>::min();
constexpr int64_t kBigIntMax = std::numeric_limits<int64_t>::max();
constexpr int64_t kBigIntMin = std::numeric_limits<int64_t>::min();

MATCHER(IsNan, "is NaN") {
  return arg && std::isnan(*arg);
}

MATCHER(IsInf, "is Infinity") {
  return arg && std::isinf(*arg);
}

auto roundToPrecision(double num, int precision) {
  double factor = pow(10.0, precision);
  return round(num * factor) / factor;
};

class ProbabilityTest : public functions::test::FunctionBaseTest {
 protected:
  template <typename ValueType>
  auto poissonCDF(
      const std::optional<double>& lambda,
      const std::optional<ValueType>& value) {
    return evaluateOnce<double>("poisson_cdf(c0, c1)", lambda, value);
  }

  template <typename ValueType>
  auto binomialCDF(
      std::optional<ValueType> numberOfTrials,
      std::optional<double> successProbability,
      std::optional<ValueType> value) {
    return evaluateOnce<double>(
        "binomial_cdf(c0, c1, c2)", numberOfTrials, successProbability, value);
  }

  template <typename ValueType>
  void poissonCDFTests() {
    EXPECT_EQ(0.91608205796869657, poissonCDF<ValueType>(3, 5));
    EXPECT_EQ(0, poissonCDF<ValueType>(kDoubleMax, 3));
    EXPECT_EQ(
        1, poissonCDF<ValueType>(3, std::numeric_limits<ValueType>::max()));
    EXPECT_EQ(1, poissonCDF<ValueType>(kDoubleMin, 3));
    EXPECT_EQ(std::nullopt, poissonCDF<ValueType>(std::nullopt, 1));
    EXPECT_EQ(std::nullopt, poissonCDF<ValueType>(1, std::nullopt));
    EXPECT_EQ(std::nullopt, poissonCDF<ValueType>(std::nullopt, std::nullopt));
    VELOX_ASSERT_THROW(
        poissonCDF<ValueType>(kNan, 3), "lambda must be greater than 0");
    VELOX_ASSERT_THROW(
        poissonCDF<ValueType>(-3, 5), "lambda must be greater than 0");
    VELOX_ASSERT_THROW(
        poissonCDF<ValueType>(3, std::numeric_limits<ValueType>::min()),
        "value must be a non-negative integer");
    VELOX_ASSERT_THROW(
        poissonCDF<ValueType>(3, -10), "value must be a non-negative integer");
    EXPECT_THROW(poissonCDF<ValueType>(kInf, 3), VeloxUserError);
  }

  template <typename ValueType>
  void binomialCDFTests() {
    EXPECT_EQ(binomialCDF<ValueType>(5, 0.5, 5), 1.0);
    EXPECT_EQ(binomialCDF<ValueType>(5, 0.5, 0), 0.03125);
    EXPECT_EQ(binomialCDF<ValueType>(3, 0.5, 1), 0.5);
    EXPECT_EQ(binomialCDF<ValueType>(20, 1.0, 0), 0.0);
    EXPECT_EQ(binomialCDF<ValueType>(20, 0.3, 6), 0.60800981220092398);
    EXPECT_EQ(binomialCDF<ValueType>(200, 0.3, 60), 0.5348091761606989);
    EXPECT_EQ(
        binomialCDF<ValueType>(std::numeric_limits<ValueType>::max(), 0.5, 2),
        0.0);
    EXPECT_EQ(
        binomialCDF<ValueType>(
            std::numeric_limits<ValueType>::max(),
            0.5,
            std::numeric_limits<ValueType>::max()),
        0.0);
    EXPECT_EQ(
        binomialCDF<ValueType>(10, 0.5, std::numeric_limits<ValueType>::max()),
        1.0);
    EXPECT_EQ(
        binomialCDF<ValueType>(10, 0.1, std::numeric_limits<ValueType>::min()),
        0.0);
    EXPECT_EQ(binomialCDF<ValueType>(10, 0.1, -2), 0.0);
    EXPECT_EQ(binomialCDF<ValueType>(25, 0.5, -100), 0.0);
    EXPECT_EQ(binomialCDF<ValueType>(2, 0.1, 3), 1.0);

    // Invalid inputs
    VELOX_ASSERT_THROW(
        binomialCDF<ValueType>(5, -0.5, 3),
        "successProbability must be in the interval [0, 1]");
    VELOX_ASSERT_THROW(
        binomialCDF<ValueType>(5, 2, 3),
        "successProbability must be in the interval [0, 1]");
    VELOX_ASSERT_THROW(
        binomialCDF<ValueType>(5, std::numeric_limits<ValueType>::max(), 3),
        "successProbability must be in the interval [0, 1]");
    VELOX_ASSERT_THROW(
        binomialCDF<ValueType>(5, kNan, 3),
        "successProbability must be in the interval [0, 1]");
    VELOX_ASSERT_THROW(
        binomialCDF<ValueType>(-1, 0.5, 2),
        "numberOfTrials must be greater than 0");
    VELOX_ASSERT_THROW(
        binomialCDF<ValueType>(std::numeric_limits<ValueType>::min(), 0.5, 1),
        "numberOfTrials must be greater than 0");
    VELOX_ASSERT_THROW(
        binomialCDF<ValueType>(-2, 2, -1),
        "successProbability must be in the interval [0, 1]");
    VELOX_ASSERT_THROW(
        binomialCDF<ValueType>(-2, 0.5, -1),
        "numberOfTrials must be greater than 0");
  }
};

TEST_F(ProbabilityTest, betaCDF) {
  const auto betaCDF = [&](std::optional<double> a,
                           std::optional<double> b,
                           std::optional<double> value) {
    return evaluateOnce<double>("beta_cdf(c0, c1, c2)", a, b, value);
  };

  EXPECT_EQ(0.09888000000000001, betaCDF(3, 4, 0.2));
  EXPECT_EQ(0.0, betaCDF(3, 3.6, 0.0));
  EXPECT_EQ(1.0, betaCDF(3, 3.6, 1.0));
  EXPECT_EQ(0.21764809997679951, betaCDF(3, 3.6, 0.3));
  EXPECT_EQ(0.9972502881611551, betaCDF(3, 3.6, 0.9));
  EXPECT_EQ(0.0, betaCDF(kInf, 3, 0.2));
  EXPECT_EQ(0.0, betaCDF(3, kInf, 0.2));
  EXPECT_EQ(0.0, betaCDF(kInf, kInf, 0.2));
  EXPECT_EQ(0.0, betaCDF(kDoubleMax, kDoubleMax, 0.3));
  EXPECT_EQ(0.5, betaCDF(kDoubleMin, kDoubleMin, 0.3));

  VELOX_ASSERT_THROW(
      betaCDF(3, 3, kInf), "value must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(betaCDF(0, 3, 0.5), "a must be > 0");
  VELOX_ASSERT_THROW(betaCDF(3, 0, 0.5), "b must be > 0");
  VELOX_ASSERT_THROW(
      betaCDF(3, 5, -0.1), "value must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      betaCDF(3, 5, 1.1), "value must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(betaCDF(kNan, 3, 0.5), "a must be > 0");
  VELOX_ASSERT_THROW(betaCDF(3, kNan, 0.5), "b must be > 0");
  VELOX_ASSERT_THROW(
      betaCDF(3, 3, kNan), "value must be in the interval [0, 1]");
}

TEST_F(ProbabilityTest, normalCDF) {
  const auto normal_cdf = [&](std::optional<double> mean,
                              std::optional<double> sd,
                              std::optional<double> value) {
    return evaluateOnce<double>("normal_cdf(c0, c1, c2)", mean, sd, value);
  };

  EXPECT_EQ(0.97500210485177963, normal_cdf(0, 1, 1.96));
  EXPECT_EQ(0.5, normal_cdf(10, 9, 10));
  EXPECT_EQ(0.0013498980316301035, normal_cdf(-1.5, 2.1, -7.8));
  EXPECT_EQ(1.0, normal_cdf(0, 1, kInf));
  EXPECT_EQ(0.0, normal_cdf(0, 1, -kInf));
  EXPECT_EQ(0.0, normal_cdf(kInf, 1, 0));
  EXPECT_EQ(1.0, normal_cdf(-kInf, 1, 0));
  EXPECT_EQ(0.5, normal_cdf(0, kInf, 0));
  EXPECT_THAT(normal_cdf(kNan, 1, 0), IsNan());
  EXPECT_THAT(normal_cdf(0, 1, kNan), IsNan());
  EXPECT_EQ(normal_cdf(0, 1, kDoubleMax), 1);
  EXPECT_EQ(normal_cdf(0, kDoubleMax, 0), 0.5);
  EXPECT_EQ(0.0, normal_cdf(kDoubleMax, 1, 0));
  EXPECT_EQ(0.5, normal_cdf(0, 1, kDoubleMin));
  EXPECT_EQ(0.5, normal_cdf(kDoubleMin, 1, 0));
  EXPECT_EQ(0, normal_cdf(1, kDoubleMin, 0));
  EXPECT_THAT(normal_cdf(kDoubleMax, kDoubleMax, kInf), IsNan());
  EXPECT_EQ(0.5, normal_cdf(kDoubleMax, kDoubleMax, kDoubleMax));
  EXPECT_EQ(0.5, normal_cdf(kDoubleMin, kDoubleMin, kDoubleMin));
  EXPECT_EQ(0.5, normal_cdf(kDoubleMax, 1, kDoubleMax));
  EXPECT_EQ(0.5, normal_cdf(10, kDoubleMax, kDoubleMax));
  EXPECT_EQ(0.5, normal_cdf(kDoubleMax, kDoubleMax, 1.96));
  EXPECT_EQ(std::nullopt, normal_cdf(std::nullopt, 1, 1.96));
  EXPECT_EQ(std::nullopt, normal_cdf(1, 1, std::nullopt));
  EXPECT_EQ(std::nullopt, normal_cdf(std::nullopt, 1, std::nullopt));
  EXPECT_EQ(std::nullopt, normal_cdf(std::nullopt, std::nullopt, std::nullopt));

  VELOX_ASSERT_THROW(normal_cdf(0, 0, 0.1985), "standardDeviation must be > 0");
  VELOX_ASSERT_THROW(
      normal_cdf(0, kNan, 0.1985), "standardDeviation must be > 0");
}

TEST_F(ProbabilityTest, cauchyCDF) {
  const auto cauchyCDF = [&](std::optional<double> median,
                             std::optional<double> scale,
                             std::optional<double> value) {
    return evaluateOnce<double>("cauchy_cdf(c0, c1, c2)", median, scale, value);
  };

  EXPECT_EQ(0.5, cauchyCDF(0.0, 1.0, 0.0));
  EXPECT_EQ(0.75, cauchyCDF(0.0, 1.0, 1.0));
  EXPECT_EQ(0.25, cauchyCDF(5.0, 2.0, 3.0));
  EXPECT_EQ(1.0, cauchyCDF(5.0, 2.0, kInf));
  EXPECT_EQ(0.5, cauchyCDF(5.0, kInf, 3.0));
  EXPECT_EQ(0.0, cauchyCDF(kInf, 2.0, 3.0));
  EXPECT_EQ(1.0, cauchyCDF(5.0, 2.0, kDoubleMax));
  EXPECT_EQ(0.5, cauchyCDF(5.0, kDoubleMax, 3.0));
  EXPECT_EQ(0.0, cauchyCDF(kDoubleMax, 1.0, 1.0));
  EXPECT_EQ(0.25, cauchyCDF(1.0, 1.0, kDoubleMin));
  EXPECT_EQ(0.5, cauchyCDF(5.0, kDoubleMin, 5.0));
  EXPECT_EQ(0.75, cauchyCDF(kDoubleMin, 1.0, 1.0));
  EXPECT_EQ(0.64758361765043326, cauchyCDF(2.5, 1.0, 3.0));
  EXPECT_THAT(cauchyCDF(kNan, 1.0, 1.0), IsNan());
  EXPECT_THAT(cauchyCDF(1.0, 1.0, kNan), IsNan());
  EXPECT_THAT(cauchyCDF(kInf, 1.0, kNan), IsNan());
  VELOX_ASSERT_THROW(cauchyCDF(1.0, kNan, 1.0), "scale must be greater than 0");
  VELOX_ASSERT_THROW(cauchyCDF(0, -1, 0), "scale must be greater than 0");
}

TEST_F(ProbabilityTest, invBetaCDF) {
  const auto invBetaCDF = [&](std::optional<double> a,
                              std::optional<double> b,
                              std::optional<double> p) {
    return evaluateOnce<double>("inverse_beta_cdf(c0, c1, c2)", a, b, p);
  };

  EXPECT_EQ(0.0, invBetaCDF(3, 3.6, 0.0));
  EXPECT_EQ(1.0, invBetaCDF(3, 3.6, 1.0));
  EXPECT_EQ(0.34696754854406159, invBetaCDF(3, 3.6, 0.3));
  EXPECT_EQ(0.76002724631002683, invBetaCDF(3, 3.6, 0.95));

  EXPECT_EQ(std::nullopt, invBetaCDF(std::nullopt, 3.6, 0.95));
  EXPECT_EQ(std::nullopt, invBetaCDF(3.6, std::nullopt, 0.95));
  EXPECT_EQ(std::nullopt, invBetaCDF(3.6, 3.6, std::nullopt));

  // Boost libraries currently throw an assert. Created the expected values via
  // Matlab. Presto currently throws an exception from Apache Math for these
  // values
  // EXPECT_EQ(0.5, invBetaCDF(kDoubleMax, kDoubleMax, 0.3));
  // EXPECT_EQ(0.0, invBetaCDF(kDoubleMin, kDoubleMin, 0.3));

  VELOX_ASSERT_THROW(invBetaCDF(kInf, 3, 0.2), "a must be > 0");
  VELOX_ASSERT_THROW(invBetaCDF(kNan, 3, 0.5), "a must be > 0");
  VELOX_ASSERT_THROW(invBetaCDF(0, 3, 0.5), "a must be > 0");

  VELOX_ASSERT_THROW(invBetaCDF(3, kInf, 0.2), "b must be > 0");
  VELOX_ASSERT_THROW(invBetaCDF(3, kNan, 0.5), "b must be > 0");
  VELOX_ASSERT_THROW(invBetaCDF(3, 0, 0.5), "b must be > 0");

  VELOX_ASSERT_THROW(
      invBetaCDF(3, 3.6, kInf), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invBetaCDF(3, 3.6, kNan), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invBetaCDF(3, 5, -0.1), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(invBetaCDF(3, 5, 1.1), "p must be in the interval [0, 1]");
}

TEST_F(ProbabilityTest, chiSquaredCDF) {
  const auto chiSquaredCDF = [&](std::optional<double> df,
                                 std::optional<double> value) {
    return evaluateOnce<double>("chi_squared_cdf(c0, c1)", df, value);
  };

  EXPECT_EQ(chiSquaredCDF(3, 0.0), 0.0);
  EXPECT_EQ(chiSquaredCDF(3, 1.0), 0.1987480430987992);
  EXPECT_EQ(chiSquaredCDF(3, 2.5), 0.52470891665697938);
  EXPECT_EQ(chiSquaredCDF(3, 4), 0.73853587005088939);
  // Invalid inputs
  VELOX_ASSERT_THROW(chiSquaredCDF(-3, 0.3), "df must be greater than 0");
  VELOX_ASSERT_THROW(chiSquaredCDF(3, -10), "value must non-negative");
}

TEST_F(ProbabilityTest, fCDF) {
  const auto fCDF = [&](std::optional<double> df1,
                        std::optional<double> df2,
                        std::optional<double> value) {
    return evaluateOnce<double>("f_cdf(c0, c1, c2)", df1, df2, value);
  };

  EXPECT_EQ(fCDF(2.0, 5.0, 0.0), 0.0);
  EXPECT_EQ(fCDF(2.0, 5.0, 0.7988), 0.50001145221750731);
  EXPECT_EQ(fCDF(2.0, 5.0, 3.7797), 0.89999935988961155);

  EXPECT_EQ(fCDF(kDoubleMax, 5.0, 3.7797), 1);
  EXPECT_EQ(fCDF(1, kDoubleMax, 97.1), 1);
  EXPECT_EQ(fCDF(82.6, 901.10, kDoubleMax), 1);
  EXPECT_EQ(fCDF(12.12, 4.2015, kDoubleMin), 0);
  EXPECT_EQ(fCDF(0.4422, kDoubleMin, 0.697), 7.9148959162596482e-306);
  EXPECT_EQ(fCDF(kDoubleMin, 50.620, 4), 1);
  EXPECT_EQ(fCDF(kBigIntMax, 5.0, 3.7797), 0.93256230095450132);
  EXPECT_EQ(fCDF(76.901, kBigIntMax, 77.97), 1);
  EXPECT_EQ(fCDF(2.0, 5.0, kBigIntMax), 1);

  EXPECT_EQ(fCDF(2.0, 5.0, std::nullopt), std::nullopt);
  EXPECT_EQ(fCDF(2.0, std::nullopt, 3.7797), std::nullopt);
  EXPECT_EQ(fCDF(std::nullopt, 5.0, 3.7797), std::nullopt);

  // Test invalid inputs for df1.
  VELOX_ASSERT_THROW(fCDF(0, 3, 0.5), "numerator df must be greater than 0");
  VELOX_ASSERT_THROW(
      fCDF(kBigIntMin, 5.0, 3.7797), "numerator df must be greater than 0");

  // Test invalid inputs for df2.
  VELOX_ASSERT_THROW(fCDF(3, 0, 0.5), "denominator df must be greater than 0");
  VELOX_ASSERT_THROW(
      fCDF(2.0, kBigIntMin, 3.7797), "denominator df must be greater than 0");

  // Test invalid inputs for value.
  VELOX_ASSERT_THROW(fCDF(3, 5, -0.1), "value must non-negative");
  VELOX_ASSERT_THROW(fCDF(2.0, 5.0, kBigIntMin), "value must non-negative");

  // Test a combination of invalid inputs.
  VELOX_ASSERT_THROW(fCDF(-1.2, 0, -0.1), "value must non-negative");
  VELOX_ASSERT_THROW(fCDF(1, -kInf, -0.1), "value must non-negative");
}

TEST_F(ProbabilityTest, laplaceCDF) {
  const auto laplaceCDF = [&](std::optional<double> location,
                              std::optional<double> scale,
                              std::optional<double> x) {
    return evaluateOnce<double>("laplace_cdf(c0, c1, c2)", location, scale, x);
  };

  EXPECT_DOUBLE_EQ(0.5, laplaceCDF(0.0, 1.0, 0.0).value());
  EXPECT_DOUBLE_EQ(0.5, laplaceCDF(5.0, 2.0, 5.0).value());
  EXPECT_DOUBLE_EQ(0.0, laplaceCDF(5.0, 2.0, -kInf).value());
  EXPECT_THAT(laplaceCDF(kNan, 1.0, 0.5), IsNan());
  EXPECT_THAT(laplaceCDF(1.0, 1.0, kNan), IsNan());
  EXPECT_THAT(laplaceCDF(kInf, 1.0, kNan), IsNan());
  EXPECT_EQ(std::nullopt, laplaceCDF(std::nullopt, 1.0, 0.5));
  EXPECT_EQ(std::nullopt, laplaceCDF(1.0, std::nullopt, 0.5));
  EXPECT_EQ(std::nullopt, laplaceCDF(1.0, 1.0, std::nullopt));
  EXPECT_EQ(0, laplaceCDF(kDoubleMax, 1.0, 0.5));
  EXPECT_EQ(0.5, laplaceCDF(1.0, kDoubleMax, 0.5));
  EXPECT_EQ(1, laplaceCDF(1.0, 1.0, kDoubleMax));
  EXPECT_NEAR(
      0.69673467014368329, laplaceCDF(kDoubleMin, 1.0, 0.5).value(), 1e-15);
  EXPECT_EQ(0, laplaceCDF(1.0, kDoubleMin, 0.5));
  EXPECT_NEAR(
      0.18393972058572117, laplaceCDF(1.0, 1.0, kDoubleMin).value(), 1e-15);
  VELOX_ASSERT_THROW(laplaceCDF(1.0, 0.0, 0.5), "scale must be greater than 0");
  VELOX_ASSERT_THROW(
      laplaceCDF(1.0, -1.0, 0.5), "scale must be greater than 0");
}

TEST_F(ProbabilityTest, gammaCDF) {
  const auto gammaCDF = [&](std::optional<double> shape,
                            std::optional<double> scale,
                            std::optional<double> value) {
    return evaluateOnce<double>("gamma_cdf(c0, c1, c2)", shape, scale, value);
  };

  EXPECT_DOUBLE_EQ(0.96675918913720599, gammaCDF(0.5, 3.0, 6.8).value());
  EXPECT_DOUBLE_EQ(0.50636537728827200, gammaCDF(1.5, 2.0, 2.4).value());
  EXPECT_DOUBLE_EQ(0.55950671493478754, gammaCDF(5.0, 2.0, 10.0).value());
  EXPECT_DOUBLE_EQ(0.01751372384616767, gammaCDF(6.5, 3.5, 8.1).value());
  EXPECT_DOUBLE_EQ(1.0, gammaCDF(5.0, 2.0, 100.0).value());
  EXPECT_DOUBLE_EQ(0.0, gammaCDF(5.0, 2.0, 0.0).value());
  EXPECT_DOUBLE_EQ(0.15085496391539036, gammaCDF(2.5, 1.0, 1.0).value());
  EXPECT_DOUBLE_EQ(1.0, gammaCDF(2.0, kInf, kInf).value());
  EXPECT_DOUBLE_EQ(0.0, gammaCDF(kInf, 3.0, 6.0).value());
  EXPECT_DOUBLE_EQ(0.0, gammaCDF(2.0, kInf, 6.0).value());
  EXPECT_DOUBLE_EQ(1.0, gammaCDF(2.0, 3.0, kInf).value());
  EXPECT_DOUBLE_EQ(0.0, gammaCDF(kDoubleMax, 3.0, 6.0).value());
  EXPECT_DOUBLE_EQ(0.0, gammaCDF(2.0, kDoubleMax, 6.0).value());
  EXPECT_DOUBLE_EQ(1.0, gammaCDF(2.0, 3.0, kDoubleMax).value());
  EXPECT_DOUBLE_EQ(1.0, gammaCDF(kDoubleMin, 3.0, 6.0).value());
  EXPECT_DOUBLE_EQ(1.0, gammaCDF(2.0, kDoubleMin, 6.0).value());
  EXPECT_DOUBLE_EQ(0.0, gammaCDF(2.0, 3.0, kDoubleMin).value());

  EXPECT_EQ(std::nullopt, gammaCDF(std::nullopt, 3.0, 6.0));
  EXPECT_EQ(std::nullopt, gammaCDF(2.0, std::nullopt, 6.0));
  EXPECT_EQ(std::nullopt, gammaCDF(2.0, 3.0, std::nullopt));

  // invalid inputs
  VELOX_ASSERT_THROW(gammaCDF(-1.0, 3.0, 6.0), "shape must be greater than 0");
  VELOX_ASSERT_THROW(gammaCDF(2.0, -1.0, 6.0), "scale must be greater than 0");
  VELOX_ASSERT_THROW(
      gammaCDF(2.0, 3.0, -1.0), "value must be greater than, or equal to, 0");
  VELOX_ASSERT_THROW(gammaCDF(kNan, 3.0, 6.0), "shape must be greater than 0");
  VELOX_ASSERT_THROW(gammaCDF(2.0, kNan, 6.0), "scale must be greater than 0");
  VELOX_ASSERT_THROW(
      gammaCDF(2.0, 3.0, kNan), "value must be greater than, or equal to, 0");
}

TEST_F(ProbabilityTest, poissonCDF) {
  poissonCDFTests<int32_t>();
}

TEST_F(ProbabilityTest, binomialCDF) {
  binomialCDFTests<int32_t>();
  binomialCDFTests<int64_t>();
}

TEST_F(ProbabilityTest, weibullCDF) {
  const auto weibullCDF = [&](std::optional<double> a,
                              std::optional<double> b,
                              std::optional<double> value) {
    return evaluateOnce<double>("weibull_cdf(c0, c1, c2)", a, b, value);
  };

  EXPECT_EQ(weibullCDF(1.0, 1.0, 0.0), 0.0);
  EXPECT_EQ(weibullCDF(1.0, 1.0, 40.0), 1.0);
  EXPECT_EQ(weibullCDF(1.0, 0.6, 3.0), 0.99326205300091452);
  EXPECT_EQ(weibullCDF(1.0, 0.9, 2.0), 0.89163197677810413);

  EXPECT_EQ(weibullCDF(std::nullopt, 1.0, 0.3), std::nullopt);
  EXPECT_EQ(weibullCDF(1.0, std::nullopt, 0.2), std::nullopt);
  EXPECT_EQ(weibullCDF(1.0, 0.4, std::nullopt), std::nullopt);

  EXPECT_EQ(weibullCDF(kDoubleMin, 1.0, 2.0), 0.63212055882855767);
  EXPECT_EQ(weibullCDF(kDoubleMax, 1.0, 3.0), 1.0);
  EXPECT_EQ(weibullCDF(1.0, kDoubleMin, 2.0), 1.0);
  EXPECT_EQ(weibullCDF(1.0, kDoubleMax, 3.0), 1.668805393880401e-308);
  EXPECT_EQ(weibullCDF(kInf, 1.0, 3.0), 1.0);
  EXPECT_EQ(weibullCDF(1.0, kInf, 20.0), 0.0);
  EXPECT_EQ(weibullCDF(kDoubleMin, kDoubleMin, 1.0), 0.63212055882855767);
  EXPECT_EQ(weibullCDF(kDoubleMax, kDoubleMax, 4.0), 0.0);
  EXPECT_EQ(weibullCDF(kDoubleMax, kDoubleMin, kInf), 1.0);
  EXPECT_EQ(weibullCDF(kInf, kInf, 10.0), 0.0);
  EXPECT_EQ(weibullCDF(1.0, 1.0, kInf), 1.0);
  EXPECT_EQ(weibullCDF(99999999999999, 999999999999999, kInf), 1.0);
  EXPECT_EQ(weibullCDF(kInf, 1.0, 40.0), 1.0);
  EXPECT_EQ(weibullCDF(1.0, kInf, 10.0), 0.0);
  EXPECT_THAT(weibullCDF(1.0, 0.5, kNan), IsNan());
  EXPECT_THAT(weibullCDF(99999999999999.0, 999999999999999.0, kNan), IsNan());

  VELOX_ASSERT_THROW(
      weibullCDF(kNan, kNan, kDoubleMin), "a must be greater than 0");
  VELOX_ASSERT_THROW(weibullCDF(0, 3, 0.5), "a must be greater than 0");
  VELOX_ASSERT_THROW(weibullCDF(3, 0, 0.5), "b must be greater than 0");
  VELOX_ASSERT_THROW(weibullCDF(kNan, 3.0, 0.5), "a must be greater than 0");
  VELOX_ASSERT_THROW(weibullCDF(3.0, kNan, 0.5), "b must be greater than 0");
  VELOX_ASSERT_THROW(weibullCDF(-1.0, 1.0, 30.0), "a must be greater than 0");
  VELOX_ASSERT_THROW(weibullCDF(1.0, -1.0, 40.0), "b must be greater than 0");
  VELOX_ASSERT_THROW(
      weibullCDF(kNan, kDoubleMin, kDoubleMax), "a must be greater than 0");
  VELOX_ASSERT_THROW(
      weibullCDF(kDoubleMin, kNan, kDoubleMax), "b must be greater than 0");
}

TEST_F(ProbabilityTest, inverseNormalCDF) {
  const auto inverseNormalCDF = [&](std::optional<double> mean,
                                    std::optional<double> sd,
                                    std::optional<double> p) {
    return evaluateOnce<double>("inverse_normal_cdf(c0, c1, c2)", mean, sd, p);
  };

  EXPECT_EQ(inverseNormalCDF(0, 1, 0.3), -0.52440051270804089);
  EXPECT_EQ(inverseNormalCDF(10, 9, 0.9), 21.533964089901406);
  EXPECT_EQ(inverseNormalCDF(0.5, 0.25, 0.65), 0.59633011660189195);
  EXPECT_EQ(inverseNormalCDF(0, 1, 0.00001), -4.2648907939226017);

  EXPECT_EQ(inverseNormalCDF(kDoubleMin, 0.25, 0.65), 0.096330116601891919);
  EXPECT_EQ(inverseNormalCDF(kDoubleMax, 0.25, 0.65), 1.7976931348623157e+308);
  EXPECT_EQ(inverseNormalCDF(0.5, kDoubleMin, 0.65), 0.5);
  EXPECT_THAT(inverseNormalCDF(0.5, kDoubleMax, 0.65), IsInf());
  EXPECT_THAT(inverseNormalCDF(kNan, 2, 0.1985), IsNan());

  EXPECT_EQ(inverseNormalCDF(std::nullopt, 1, 1), std::nullopt);
  EXPECT_EQ(inverseNormalCDF(1, 1, std::nullopt), std::nullopt);
  EXPECT_EQ(inverseNormalCDF(std::nullopt, 1, std::nullopt), std::nullopt);
  EXPECT_EQ(
      inverseNormalCDF(std::nullopt, std::nullopt, std::nullopt), std::nullopt);

  EXPECT_THAT(inverseNormalCDF(kInf, 1, 0.1985), IsInf());
  EXPECT_THAT(inverseNormalCDF(0, kInf, 0.1985), IsInf());
  EXPECT_THAT(inverseNormalCDF(-kInf, 1, 0.1985), IsInf());

  // Test invalid inputs.
  VELOX_ASSERT_THROW(
      inverseNormalCDF(0, -kInf, 0.1985), "standardDeviation must be > 0");
  VELOX_ASSERT_THROW(inverseNormalCDF(0, 1, kInf), "p must be 0 > p > 1");
  VELOX_ASSERT_THROW(inverseNormalCDF(0, 1, -kInf), "p must be 0 > p > 1");

  VELOX_ASSERT_THROW(
      inverseNormalCDF(0, kNan, 0.1985), "standardDeviation must be > 0");
  VELOX_ASSERT_THROW(inverseNormalCDF(0, 1, kNan), "p must be 0 > p > 1");
  VELOX_ASSERT_THROW(inverseNormalCDF(kNan, kNan, kNan), "p must be 0 > p > 1");

  VELOX_ASSERT_THROW(inverseNormalCDF(0, 1, kDoubleMax), "p must be 0 > p > 1");
  VELOX_ASSERT_THROW(
      inverseNormalCDF(0, 1, kDoubleMin),
      "Error in function boost::math::erf_inv<double>(double, double): Overflow Error");

  VELOX_ASSERT_THROW(
      inverseNormalCDF(0, 0, 0.1985), "standardDeviation must be > 0");
  VELOX_ASSERT_THROW(inverseNormalCDF(0, 1, 1.00001), "p must be 0 > p > 1");
}

TEST_F(ProbabilityTest, inverseWeibullCDF) {
  const auto inverseWeibullCDF = [&](std::optional<double> a,
                                     std::optional<double> b,
                                     std::optional<double> p) {
    return evaluateOnce<double>("inverse_weibull_cdf(c0, c1, c2)", a, b, p);
  };

  const auto roundToTwoDecimals = [](double value) {
    return round(value * 100) / 100.0;
  };

  EXPECT_EQ(inverseWeibullCDF(1.0, 1.0, 0.), 0.0);
  EXPECT_EQ(
      roundToTwoDecimals(inverseWeibullCDF(1.0, 1.0, 0.632).value()), 1.00);
  EXPECT_EQ(
      roundToTwoDecimals(inverseWeibullCDF(1.0, 0.6, 0.91).value()), 1.44);

  VELOX_ASSERT_THROW(
      inverseWeibullCDF(1.0, 1.0, kNan), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      inverseWeibullCDF(999999999.9, 999999999.0, kInf),
      "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      inverseWeibullCDF(3, 5, -0.1), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      inverseWeibullCDF(3, 5, 1.1), "p must be in the interval [0, 1]");

  EXPECT_EQ(inverseWeibullCDF(kDoubleMin, 1.0, 0.3), 0.0);
  EXPECT_EQ(inverseWeibullCDF(kDoubleMax, 1.0, 0.4), 1.0);
  EXPECT_EQ(inverseWeibullCDF(kInf, 999999999.9, 0.4), 9.999999999E8);
  EXPECT_EQ(inverseWeibullCDF(kInf, 1.0, 0.1), 1.0);
  EXPECT_EQ(inverseWeibullCDF(kDoubleMin, kDoubleMin, 0.9), kInf);
  EXPECT_EQ(inverseWeibullCDF(kDoubleMax, kDoubleMax, 0.8), kDoubleMax);
  VELOX_ASSERT_THROW(
      inverseWeibullCDF(kNan, 1.0, 0.1), "a must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseWeibullCDF(-1.0, 1.0, 0.1), "a must be greater than 0");
  VELOX_ASSERT_THROW(inverseWeibullCDF(0, 3, 0.5), "a must be greater than 0");

  EXPECT_EQ(inverseWeibullCDF(1.0, kDoubleMin, 0.5), 1.5423036715619055e-308);
  EXPECT_EQ(inverseWeibullCDF(1.0, kDoubleMax, 0.7), kInf);
  EXPECT_THAT(inverseWeibullCDF(1.0, kInf, 0.2), IsInf());
  VELOX_ASSERT_THROW(
      inverseWeibullCDF(1.0, kNan, 0.4), "b must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseWeibullCDF(1.0, -1.0, 0.4), "b must be greater than 0");
  VELOX_ASSERT_THROW(inverseWeibullCDF(3, 0, 0.5), "b must be greater than 0");
}

TEST_F(ProbabilityTest, inverseCauchyCDF) {
  const auto invCauchyCDF = [&](std::optional<double> median,
                                std::optional<double> scale,
                                std::optional<double> p) {
    return evaluateOnce<double>(
        "inverse_cauchy_cdf(c0, c1, c2)", median, scale, p);
  };

  EXPECT_EQ(invCauchyCDF(0.0, 1.0, 0.5), 0.0);
  EXPECT_EQ(invCauchyCDF(2.5, 1.0, 0.64758361765043326), 3.0);

  EXPECT_EQ(invCauchyCDF(5.0, 2.0, 1.0), kInf);
  VELOX_ASSERT_THROW(
      invCauchyCDF(1.0, 1.0, kNan), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invCauchyCDF(1.0, 1.0, 9), "p must be in the interval [0, 1]");

  EXPECT_EQ(invCauchyCDF(5.0, kInf, 0.5), 5.0);
  VELOX_ASSERT_THROW(invCauchyCDF(0, -1, 0.5), "scale must be greater than 0");

  EXPECT_THAT(invCauchyCDF(kNan, 1.0, 0.5), IsNan());
  EXPECT_EQ(invCauchyCDF(kInf, 2.0, 0.5), kInf);
  EXPECT_EQ(invCauchyCDF(kDoubleMax, 2.0, 0.5), kDoubleMax);
  EXPECT_EQ(invCauchyCDF(kDoubleMin, 2.0, 0.5), kDoubleMin);
}

TEST_F(ProbabilityTest, inverseLaplaceCDF) {
  const auto inverseLaplaceCDF = [&](std::optional<double> location,
                                     std::optional<double> scale,
                                     std::optional<double> p) {
    return evaluateOnce<double>(
        "inverse_laplace_cdf(c0, c1, c2)", location, scale, p);
  };

  EXPECT_EQ(inverseLaplaceCDF(0.0, 1.0, 0.5), 0.0);
  EXPECT_EQ(inverseLaplaceCDF(5.0, 2.0, 0.5), 5.0);

  VELOX_ASSERT_THROW(
      inverseLaplaceCDF(1.0, 1.0, kNan), "p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      inverseLaplaceCDF(1.0, 1.0, 2.0), "p must be in the interval [0, 1]");

  EXPECT_EQ(inverseLaplaceCDF(10.0, kDoubleMax, 0.999999999999), kInf);
  EXPECT_EQ(inverseLaplaceCDF(10.0, kDoubleMin, 0.000000000001), 10.0);
  VELOX_ASSERT_THROW(
      inverseLaplaceCDF(1.0, kNan, 0.5), "scale must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseLaplaceCDF(1.0, -1.0, 0.5), "scale must be greater than 0");

  EXPECT_THAT(inverseLaplaceCDF(kInf, 1.0, 0.5), IsNan());
  EXPECT_THAT(inverseLaplaceCDF(kNan, 1.0, 0.5), IsNan());
  EXPECT_THAT(inverseLaplaceCDF(kDoubleMax, 1.0, 0.5), kDoubleMax);
  EXPECT_THAT(inverseLaplaceCDF(kDoubleMin, 1.0, 0.5), kDoubleMin);
}

TEST_F(ProbabilityTest, invGammaCDF) {
  const auto invGammaCDF = [&](std::optional<double> shape,
                               std::optional<double> scale,
                               std::optional<double> p) {
    return evaluateOnce<double>(
        "inverse_gamma_cdf(c0, c1, c2)", shape, scale, p);
  };

  EXPECT_EQ(0, invGammaCDF(3, 3.6, 0.0));
  EXPECT_EQ(33.624, roundToPrecision(invGammaCDF(3, 4, 0.99).value(), 3));
  EXPECT_EQ(10.696, roundToPrecision(invGammaCDF(3, 4, 0.50).value(), 3));
  EXPECT_EQ(
      9999.333,
      roundToPrecision(invGammaCDF(10000.0 / 2, 2.0, 0.50).value(), 3));
  EXPECT_EQ(std::numeric_limits<double>::infinity(), invGammaCDF(1, 1, 1));

  // This is an example to illustrate the precision differences
  // Java: exactly 0.0,
  // C++: 5.0926835901984765e-184
  EXPECT_EQ(
      0.0,
      roundToPrecision(
          ((invGammaCDF(
                0.005626026709040224, 0.6631619541440159, 0.09358172053411777))
               .value()),
          3));

  EXPECT_EQ(std::nullopt, invGammaCDF(std::nullopt, 1, 1));
  EXPECT_EQ(std::nullopt, invGammaCDF(1, std::nullopt, 1));
  EXPECT_EQ(std::nullopt, invGammaCDF(1, 1, std::nullopt));

  VELOX_ASSERT_THROW(
      invGammaCDF(0, 3, 0.5),
      "inverseGammaCdf Function: shape must be greater than 0");
  VELOX_ASSERT_THROW(
      invGammaCDF(3, 0, 0.5),
      "inverseGammaCdf Function: scale must be greater than 0");
  VELOX_ASSERT_THROW(
      invGammaCDF(3, 5, -0.1),
      "inverseGammaCdf Function: p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invGammaCDF(3, 5, 1.1),
      "inverseGammaCdf Function: p must be in the interval [0, 1]");

  VELOX_ASSERT_THROW(
      invGammaCDF(kInf, 1, 0.5),
      "inverseGammaCdf Function: shape must be greater than 0");
  VELOX_ASSERT_THROW(
      invGammaCDF(kNan, 1, 0.5),
      "inverseGammaCdf Function: shape must be greater than 0");
  VELOX_ASSERT_THROW(
      invGammaCDF(1, kInf, 0.5),
      "inverseGammaCdf Function: scale must be greater than 0");
  VELOX_ASSERT_THROW(
      invGammaCDF(1, kNan, 0.5),
      "inverseGammaCdf Function: scale must be greater than 0");
  VELOX_ASSERT_THROW(
      invGammaCDF(1, 0.5, kInf),
      "inverseGammaCdf Function: p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invGammaCDF(1, 0.5, kNan),
      "inverseGammaCdf Function: p must be in the interval [0, 1]");
}

TEST_F(ProbabilityTest, invBinomialCDF) {
  const auto invBinomialCDF = [&](std::optional<int32_t> numberOfTrials,
                                  std::optional<double> successProbability,
                                  std::optional<double> p) {
    return evaluateOnce<int32_t>(
        "inverse_binomial_cdf(c0, c1, c2)",
        numberOfTrials,
        successProbability,
        p);
  };

  EXPECT_EQ(0, invBinomialCDF(20, 0.5, 0.0));
  EXPECT_EQ(10, invBinomialCDF(20, 0.5, 0.5));
  EXPECT_EQ(20, invBinomialCDF(20, 0.5, 1.0));
  EXPECT_EQ(INT32_MAX, invBinomialCDF(INT32_MAX, 0.5, 1));
  EXPECT_EQ(611204, invBinomialCDF(1223340, 0.5, 0.2));

  EXPECT_EQ(std::nullopt, invBinomialCDF(std::nullopt, 1, 1));
  EXPECT_EQ(std::nullopt, invBinomialCDF(1, std::nullopt, 1));
  EXPECT_EQ(std::nullopt, invBinomialCDF(1, 1, std::nullopt));

  VELOX_ASSERT_THROW(
      invBinomialCDF(5, -0.5, 0.3),
      "inverseBinomialCdf Function: successProbability must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invBinomialCDF(5, 1.5, 0.3),
      "inverseBinomialCdf Function: successProbability must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invBinomialCDF(5, 0.5, -3.0),
      "inverseBinomialCdf Function: p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invBinomialCDF(5, 0.5, 3.0),
      "inverseBinomialCdf Function: p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invBinomialCDF(-5, 0.5, 0.3),
      "inverseBinomialCdf Function: numberOfTrials must be greater than 0");

  VELOX_ASSERT_THROW(
      invBinomialCDF(1, kInf, 0.5),
      "inverseBinomialCdf Function: successProbability must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invBinomialCDF(1, kNan, 0.5),
      "inverseBinomialCdf Function: successProbability must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invBinomialCDF(1, 0.5, kInf),
      "inverseBinomialCdf Function: p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      invBinomialCDF(1, 0.5, kNan),
      "inverseBinomialCdf Function: p must be in the interval [0, 1]");
}

TEST_F(ProbabilityTest, invPoissonCDF) {
  const auto invPoissonCDF = [&](std::optional<double> lambda,
                                 std::optional<double> p) {
    return evaluateOnce<int32_t>("inverse_poisson_cdf(c0, c1)", lambda, p);
  };

  EXPECT_EQ(0, invPoissonCDF(3, 0));
  // EXPECT_EQ(2, invPoissonCDF(3, 0.3)); // 1.499999... round to floor to 1
  EXPECT_EQ(6, invPoissonCDF(3, 0.95));
  EXPECT_EQ(17, invPoissonCDF(3, 0.99999999));
  EXPECT_EQ(
      std::numeric_limits<int32_t>::max(),
      invPoissonCDF(1.8819579427461317E18, 0.659094));

  EXPECT_EQ(std::nullopt, invPoissonCDF(std::nullopt, 0.5));
  EXPECT_EQ(std::nullopt, invPoissonCDF(0.5, std::nullopt));

  VELOX_ASSERT_THROW(
      invPoissonCDF(-3, 0.3),
      "inversePoissonCdf Function: lambda must be greater than 0");
  VELOX_ASSERT_THROW(
      invPoissonCDF(kInf, 0),
      "inversePoissonCdf Function: lambda must be greater than 0");
  VELOX_ASSERT_THROW(
      invPoissonCDF(kNan, 0),
      "inversePoissonCdf Function: lambda must be greater than 0");
  VELOX_ASSERT_THROW(
      invPoissonCDF(0, 0),
      "inversePoissonCdf Function: lambda must be greater than 0");

  VELOX_ASSERT_THROW(
      invPoissonCDF(0.5, kInf),
      "inversePoissonCdf Function: p must be in the interval [0, 1)");
  VELOX_ASSERT_THROW(
      invPoissonCDF(0.5, kNan),
      "inversePoissonCdf Function: p must be in the interval [0, 1)");
  VELOX_ASSERT_THROW(
      invPoissonCDF(3, 1.1),
      "inversePoissonCdf Function: p must be in the interval [0, 1)");
  VELOX_ASSERT_THROW(
      invPoissonCDF(3, 1),
      "inversePoissonCdf Function: p must be in the interval [0, 1)");
  VELOX_ASSERT_THROW(
      invPoissonCDF(3, -0.1),
      "inversePoissonCdf Function: p must be in the interval [0, 1)");
}

TEST_F(ProbabilityTest, inverseFCDF) {
  const auto inverseFCDF = [&](std::optional<double> df1,
                               std::optional<double> df2,
                               std::optional<double> p) {
    return evaluateOnce<double>("inverse_f_cdf(c0, c1, c2)", df1, df2, p);
  };

  EXPECT_EQ(0, inverseFCDF(2.0, 5.0, 0.0));
  EXPECT_EQ(0.7988, roundToPrecision((inverseFCDF(2.0, 5.0, 0.5)).value(), 4));
  EXPECT_EQ(3.7797, roundToPrecision((inverseFCDF(2.0, 5.0, 0.9)).value(), 4));

  VELOX_ASSERT_THROW(
      inverseFCDF(0, 3, 0.5),
      "inverseFCdf Function: numerator df must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseFCDF(3, 0, 0.5),
      "inverseFCdf Function: denominator df must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseFCDF(3, 5, -0.1),
      "inverseFCdf Function: p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      inverseFCDF(3, 5, 1.1),
      "inverseFCdf Function: p must be in the interval [0, 1]");

  VELOX_ASSERT_THROW(
      inverseFCDF(kNan, 1, 0.5),
      "inverseFCdf Function: numerator df must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseFCDF(kInf, 1, 0.5),
      "inverseFCdf Function: numerator df must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseFCDF(1, kNan, 0.5),
      "inverseFCdf Function: denominator df must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseFCDF(1, kInf, 0.5),
      "inverseFCdf Function: denominator df must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseFCDF(1, 1, kNan),
      "inverseFCdf Function: p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      inverseFCDF(1, 1, kInf),
      "inverseFCdf Function: p must be in the interval [0, 1]");
}

TEST_F(ProbabilityTest, inverseChiSquaredCdf) {
  const auto inverseChiSquaredCdf = [&](std::optional<double> df,
                                        std::optional<double> p) {
    return evaluateOnce<double>("inverse_chi_squared_cdf(c0, c1)", df, p);
  };

  EXPECT_EQ(0, inverseChiSquaredCdf(3, 0.0));
  EXPECT_EQ(
      11.3449, roundToPrecision(inverseChiSquaredCdf(3, 0.99).value(), 4));
  EXPECT_EQ(1.42, roundToPrecision(inverseChiSquaredCdf(3, 0.3).value(), 2));
  EXPECT_EQ(7.81, roundToPrecision(inverseChiSquaredCdf(3, 0.95).value(), 2));

  VELOX_ASSERT_THROW(
      inverseChiSquaredCdf(-3, 0.3),
      "inverseChiSquaredCdf Function: df must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseChiSquaredCdf(3, -0.1),
      "inverseChiSquaredCdf Function: p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      inverseChiSquaredCdf(3, 1.1),
      "inverseChiSquaredCdf Function: p must be in the interval [0, 1]");

  VELOX_ASSERT_THROW(
      inverseChiSquaredCdf(kNan, 0.5),
      "inverseChiSquaredCdf Function: df must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseChiSquaredCdf(kInf, 0.5),
      "inverseChiSquaredCdf Function: df must be greater than 0");
  VELOX_ASSERT_THROW(
      inverseChiSquaredCdf(1, kNan),
      "inverseChiSquaredCdf Function: p must be in the interval [0, 1]");
  VELOX_ASSERT_THROW(
      inverseChiSquaredCdf(1, kInf),
      "inverseChiSquaredCdf Function: p must be in the interval [0, 1]");
}

} // namespace
} // namespace facebook::velox

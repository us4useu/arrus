// Tests for the type-descriptor family: DType, DeviceRef, Shape, NdArrayType.
#include <gtest/gtest.h>

#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/framework/DType.h"
#include "arrus/core/api/framework/DeviceRef.h"
#include "arrus/core/api/framework/NdArrayType.h"
#include "arrus/core/api/framework/Shape.h"

namespace {

using ::arrus::IllegalArgumentException;
using ::arrus::framework::DType;
using ::arrus::framework::DeviceRef;
using ::arrus::framework::NdArrayType;
using ::arrus::framework::Shape;
using ::arrus::framework::dtypeName;
using ::arrus::framework::dtypeSize;
using ::arrus::framework::isComplex;
using ::arrus::framework::isFloating;

// --- DType ----------------------------------------------------------------

TEST(DTypeTest, SizesMatchExpectedBytes) {
    EXPECT_EQ(dtypeSize(DType::BOOL),       sizeof(bool));
    EXPECT_EQ(dtypeSize(DType::INT8),       1u);
    EXPECT_EQ(dtypeSize(DType::UINT8),      1u);
    EXPECT_EQ(dtypeSize(DType::INT16),      2u);
    EXPECT_EQ(dtypeSize(DType::UINT16),     2u);
    EXPECT_EQ(dtypeSize(DType::INT32),      4u);
    EXPECT_EQ(dtypeSize(DType::UINT32),     4u);
    EXPECT_EQ(dtypeSize(DType::FLOAT32),    4u);
    EXPECT_EQ(dtypeSize(DType::INT64),      8u);
    EXPECT_EQ(dtypeSize(DType::UINT64),     8u);
    EXPECT_EQ(dtypeSize(DType::FLOAT64),    8u);
    EXPECT_EQ(dtypeSize(DType::COMPLEX64),  8u);
    EXPECT_EQ(dtypeSize(DType::COMPLEX128), 16u);
}

TEST(DTypeTest, UnknownDtypeHasNoDefinedSize) {
    EXPECT_THROW(dtypeSize(DType::UNKNOWN), IllegalArgumentException);
}

TEST(DTypeTest, NamesAreDistinct) {
    std::unordered_set<std::string> names;
    for (auto d : {DType::UNKNOWN, DType::BOOL, DType::INT8, DType::UINT8,
                   DType::INT16, DType::UINT16, DType::INT32, DType::UINT32,
                   DType::INT64, DType::UINT64, DType::FLOAT32, DType::FLOAT64,
                   DType::COMPLEX64, DType::COMPLEX128}) {
        names.insert(dtypeName(d));
    }
    EXPECT_EQ(names.size(), 14u);
}

TEST(DTypeTest, IsComplexAndIsFloatingClassifyCorrectly) {
    EXPECT_FALSE(isComplex(DType::FLOAT32));
    EXPECT_TRUE(isComplex(DType::COMPLEX64));
    EXPECT_TRUE(isComplex(DType::COMPLEX128));

    EXPECT_TRUE(isFloating(DType::FLOAT32));
    EXPECT_TRUE(isFloating(DType::FLOAT64));
    EXPECT_TRUE(isFloating(DType::COMPLEX64));
    EXPECT_TRUE(isFloating(DType::COMPLEX128));

    EXPECT_FALSE(isFloating(DType::INT16));
    EXPECT_FALSE(isFloating(DType::BOOL));
}

// Enum values are the on-disk serialization codes; we lock the numeric
// mapping so future readers cannot silently break wire compatibility by
// reordering the enum.
TEST(DTypeTest, WireCodesAreStable) {
    EXPECT_EQ(static_cast<uint32_t>(DType::UNKNOWN),    0u);
    EXPECT_EQ(static_cast<uint32_t>(DType::BOOL),       1u);
    EXPECT_EQ(static_cast<uint32_t>(DType::INT8),       2u);
    EXPECT_EQ(static_cast<uint32_t>(DType::UINT8),      3u);
    EXPECT_EQ(static_cast<uint32_t>(DType::INT16),      4u);
    EXPECT_EQ(static_cast<uint32_t>(DType::UINT16),     5u);
    EXPECT_EQ(static_cast<uint32_t>(DType::INT32),      6u);
    EXPECT_EQ(static_cast<uint32_t>(DType::UINT32),     7u);
    EXPECT_EQ(static_cast<uint32_t>(DType::INT64),      8u);
    EXPECT_EQ(static_cast<uint32_t>(DType::UINT64),     9u);
    EXPECT_EQ(static_cast<uint32_t>(DType::FLOAT32),    10u);
    EXPECT_EQ(static_cast<uint32_t>(DType::FLOAT64),    11u);
    EXPECT_EQ(static_cast<uint32_t>(DType::COMPLEX64),  12u);
    EXPECT_EQ(static_cast<uint32_t>(DType::COMPLEX128), 13u);
}

// --- DeviceRef ------------------------------------------------------------

TEST(DeviceRefTest, DefaultIsAny) {
    DeviceRef d;
    EXPECT_TRUE(d.isAny());
    EXPECT_EQ(d.value(), "ANY");
}

TEST(DeviceRefTest, ExplicitConstructorSetsValue) {
    DeviceRef gpu{"GPU:0"};
    EXPECT_FALSE(gpu.isAny());
    EXPECT_EQ(gpu.value(), "GPU:0");
}

TEST(DeviceRefTest, EqualityAndInequality) {
    EXPECT_EQ(DeviceRef{"CPU"}, DeviceRef{"CPU"});
    EXPECT_NE(DeviceRef{"CPU"}, DeviceRef{"GPU:0"});
    EXPECT_EQ(DeviceRef{}, DeviceRef{"ANY"});
}

TEST(DeviceRefTest, StreamedRepresentationIsLiteralValue) {
    std::ostringstream os;
    os << DeviceRef{"Us4R:0/OEM:2"};
    EXPECT_EQ(os.str(), "Us4R:0/OEM:2");
}

TEST(DeviceRefTest, HashIsUsableInUnorderedContainers) {
    std::unordered_map<DeviceRef, int> counts;
    counts[DeviceRef{"GPU:0"}] = 1;
    counts[DeviceRef{"GPU:0"}] += 2;  // overwrites/increments existing entry
    counts[DeviceRef{"CPU"}] = 5;
    EXPECT_EQ(counts.size(), 2u);
    EXPECT_EQ(counts[DeviceRef{"GPU:0"}], 3);
    EXPECT_EQ(counts[DeviceRef{"CPU"}], 5);
}

// --- Shape ----------------------------------------------------------------

TEST(ShapeTest, DefaultConstructedIsScalar) {
    Shape s;
    EXPECT_EQ(s.rank(), 0u);
    EXPECT_TRUE(s.isScalar());
    EXPECT_TRUE(s.isFullyKnown());
    EXPECT_EQ(s.numElements(), 1u);  // scalar has one element
}

TEST(ShapeTest, InitializerListConstructor) {
    Shape s{8u, 128u, 32u};
    EXPECT_EQ(s.rank(), 3u);
    EXPECT_FALSE(s.isScalar());
    EXPECT_TRUE(s.isFullyKnown());
    EXPECT_EQ(s[0].value(), 8u);
    EXPECT_EQ(s[1].value(), 128u);
    EXPECT_EQ(s[2].value(), 32u);
}

TEST(ShapeTest, UnknownDimsAreNotFullyKnown) {
    Shape s{8u, std::nullopt, 32u};
    EXPECT_EQ(s.rank(), 3u);
    EXPECT_FALSE(s.isFullyKnown());
    EXPECT_TRUE(s[0].has_value());
    EXPECT_FALSE(s[1].has_value());
    EXPECT_TRUE(s[2].has_value());
}

TEST(ShapeTest, NumElementsThrowsOnUnknownDim) {
    Shape s{8u, std::nullopt, 32u};
    EXPECT_THROW(s.numElements(), IllegalArgumentException);
}

TEST(ShapeTest, NumElementsProduct) {
    Shape s{4u, 5u, 6u};
    EXPECT_EQ(s.numElements(), 4u * 5u * 6u);
}

TEST(ShapeTest, EqualityRespectsOrder) {
    EXPECT_EQ(Shape({2u, 3u}), Shape({2u, 3u}));
    EXPECT_NE(Shape({2u, 3u}), Shape({3u, 2u}));
    EXPECT_NE(Shape({2u, 3u}), Shape({2u, 3u, 1u}));
}

TEST(ShapeTest, StreamedRepresentationUsesQuestionMarkForUnknown) {
    std::ostringstream os;
    os << Shape({8u, std::nullopt, 32u});
    EXPECT_EQ(os.str(), "(8, ?, 32)");
}

TEST(ShapeTest, StreamedScalarIsEmptyParens) {
    std::ostringstream os;
    os << Shape{};
    EXPECT_EQ(os.str(), "()");
}

// --- NdArrayType ---------------------------------------------------------

TEST(NdArrayTypeTest, DefaultConstructedIsUnknownScalarOnAny) {
    NdArrayType t;
    EXPECT_EQ(t.dtype(), DType::UNKNOWN);
    EXPECT_TRUE(t.shape().isScalar());
    EXPECT_TRUE(t.device().isAny());
    // A scalar shape is fully known even for UNKNOWN dtype.
    EXPECT_TRUE(t.isFullyKnown());
}

TEST(NdArrayTypeTest, ExplicitConstructionCarriesAllThreeFields) {
    NdArrayType t{Shape{8u, 128u}, DType::FLOAT32, DeviceRef{"GPU:0"}};
    EXPECT_EQ(t.shape(), Shape({8u, 128u}));
    EXPECT_EQ(t.dtype(), DType::FLOAT32);
    EXPECT_EQ(t.device(), DeviceRef{"GPU:0"});
}

TEST(NdArrayTypeTest, DefaultDeviceIsAny) {
    NdArrayType t{Shape{3u}, DType::FLOAT32};
    EXPECT_TRUE(t.device().isAny());
}

TEST(NdArrayTypeTest, EqualityRequiresAllThreeFieldsMatch) {
    NdArrayType a{Shape{8u}, DType::FLOAT32, DeviceRef{"GPU:0"}};
    NdArrayType b{Shape{8u}, DType::FLOAT32, DeviceRef{"GPU:0"}};
    NdArrayType c{Shape{8u}, DType::FLOAT32, DeviceRef{"CPU"}};
    NdArrayType d{Shape{8u}, DType::FLOAT64, DeviceRef{"GPU:0"}};
    NdArrayType e{Shape{9u}, DType::FLOAT32, DeviceRef{"GPU:0"}};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
    EXPECT_NE(a, e);
}

TEST(NdArrayTypeTest, IsFullyKnownFollowsShape) {
    NdArrayType known{Shape{8u, 128u}, DType::FLOAT32, DeviceRef{"GPU:0"}};
    NdArrayType partial{Shape{8u, std::nullopt}, DType::FLOAT32, DeviceRef{"GPU:0"}};

    EXPECT_TRUE(known.isFullyKnown());
    EXPECT_FALSE(partial.isFullyKnown());
    // "ANY" placement does not affect known-ness.
    NdArrayType any_placed{Shape{8u}, DType::FLOAT32};
    EXPECT_TRUE(any_placed.isFullyKnown());
}

TEST(NdArrayTypeTest, StreamedRepresentation) {
    NdArrayType t{Shape{8u, std::nullopt}, DType::COMPLEX64, DeviceRef{"GPU:0"}};
    std::ostringstream os;
    os << t;
    EXPECT_EQ(os.str(), "complex64(8, ?)@GPU:0");
}

TEST(NdArrayTypeTest, HashDistinguishesTypes) {
    std::hash<NdArrayType> h;
    NdArrayType a{Shape{8u}, DType::FLOAT32, DeviceRef{"GPU:0"}};
    NdArrayType b{Shape{8u}, DType::FLOAT32, DeviceRef{"GPU:0"}};
    NdArrayType c{Shape{8u}, DType::FLOAT32, DeviceRef{"CPU"}};

    EXPECT_EQ(h(a), h(b));
    // Different device -> different hash with very high probability;
    // if this ever collides it is not a correctness issue, but flag it.
    EXPECT_NE(h(a), h(c));
}

TEST(NdArrayTypeTest, HashUsableInUnorderedSet) {
    std::unordered_set<NdArrayType> types;
    types.insert(NdArrayType{Shape{8u}, DType::FLOAT32, DeviceRef{"GPU:0"}});
    types.insert(NdArrayType{Shape{8u}, DType::FLOAT32, DeviceRef{"GPU:0"}});
    types.insert(NdArrayType{Shape{16u}, DType::FLOAT32, DeviceRef{"GPU:0"}});
    EXPECT_EQ(types.size(), 2u);
}

} // namespace

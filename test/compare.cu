#include <gtest/gtest.h>
#include <gcuda/detail/internal/utility/compare.h>

TEST(Compare, EqualDimension0)
{
    {
        const char lhs = (char)250;
        const char rhs = (char)250;
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const short lhs = 17;
        const short rhs = 17;
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const int lhs = -7007;
        const int rhs = -7007;
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float lhs = -4.1245f;
        const float rhs = -4.1245f;
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const double lhs = 0.00002312;
        const double rhs = 0.00002312;
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const int lhs = -7007;
        const int rhs = -7003;
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float lhs = -4.1245f;
        const float rhs = 4.1245f;
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const double lhs = 0.00002311;
        const double rhs = 0.00002312;
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
}
TEST(Compare, EqualDimension1)
{
    {
        const char1 lhs = { (char)250 };
        const char1 rhs = { (char)250 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const short1 lhs = { 17 };
        const short1 rhs = { 17 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const ushort1 lhs = { 765 };
        const ushort1 rhs = { 765 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float1 lhs = { 1.765f };
        const float1 rhs = { 1.765f };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const double1 lhs = { 0.0000002311 };
        const double1 rhs = { 0.0000002311 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const ushort1 lhs = { 765 };
        const ushort1 rhs = { 1765 };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float1 lhs = { 1.765f };
        const float1 rhs = { 1.7649f };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
}
TEST(Compare, EqualDimension2)
{
    {
        const uchar2 lhs = { (unsigned char)250, (unsigned char)16 };
        const uchar2 rhs = { (unsigned char)250, (unsigned char)16};
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const ushort2 lhs = { 17, 1000 };
        const ushort2 rhs = { 17, 1000 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const uint2 lhs = { 765, 5656431 };
        const uint2 rhs = { 765, 5656431 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float2 lhs = { 1.765f, -0.001f };
        const float2 rhs = { 1.765f, -0.001f };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const double2 lhs = { 0.0000002311, 7651.123 };
        const double2 rhs = { 0.0000002311, 7651.123 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const uint2 lhs = { 765, 5656431 };
        const uint2 rhs = { 765, 5656435 };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float2 lhs = { 1.765f, -0.001f };
        const float2 rhs = { 1.766f, -0.001f };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const double2 lhs = { 0.0000002311, 7651.123 };
        const double2 rhs = { 0.0000002111, 7651.123 };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
}
TEST(Compare, EqualDimension3)
{
    {
        const uchar3 lhs = { (unsigned char)250, (unsigned char)16, 0 };
        const uchar3 rhs = { (unsigned char)250, (unsigned char)16, 0 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const short3 lhs = { 17, 1000, -123 };
        const short3 rhs = { 17, 1000, -123 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const int3 lhs = { 765, 5656431, -123 };
        const int3 rhs = { 765, 5656431, -123 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float3 lhs = { 1.765f, -0.001f, 0.0f };
        const float3 rhs = { 1.765f, -0.001f, 0.0f };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const double3 lhs = { 0.0000002311, 7651.123, 0.0123 };
        const double3 rhs = { 0.0000002311, 7651.123, 0.0123 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const int3 lhs = { 765, 5656431, -123 };
        const int3 rhs = { 765, 5656430, -123 };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float3 lhs = { 1.765f, -0.001f, 0.0f };
        const float3 rhs = { 1.765f, -0.011f, 0.0f };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const double3 lhs = { 0.0000002311, 7651.123, 0.0123 };
        const double3 rhs = { 0.0000002311, 7651.123, 0.012301 };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
}
TEST(Compare, EqualDimension4)
{
    {
        const uchar4 lhs = { (unsigned char)250, (unsigned char)16, 0, 1 };
        const uchar4 rhs = { (unsigned char)250, (unsigned char)16, 0, 1 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const short4 lhs = { 17, 1000, 777, -123 };
        const short4 rhs = { 17, 1000, 777, -123 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const int4 lhs = { -765, 5656431, -12324, 0xffffffff};
        const int4 rhs = { -765, 5656431, -12324, 0xffffffff };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float4 lhs = { 1.765f, -0.001f, 0.0f, 3.1415f };
        const float4 rhs = { 1.765f, -0.001f, 0.0f, 3.1415f };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const double4 lhs = { 0.0000002311, 7651.123, 0.0123, 1234.0001 };
        const double4 rhs = { 0.0000002311, 7651.123, 0.0123, 1234.0001 };
        ASSERT_TRUE(gcuda::detail::compareEq(lhs, rhs));
    }

    {
        const int4 lhs = { -765, 5656431, -12324, 0xefffffff};
        const int4 rhs = { -765, 5656431, -12324, 0xffffffff };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const float4 lhs = { 1.765f, -0.001f, 0.1f, 3.1415f };
        const float4 rhs = { 1.765f, -0.001f, 0.0f, 3.1415f };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
    {
        const double4 lhs = { 0.0000002311, 7651.123, 0.0123, 1234.0002 };
        const double4 rhs = { 0.0000002313, 7651.123, 0.0123, 1234.0001 };
        ASSERT_FALSE(gcuda::detail::compareEq(lhs, rhs));
    }
}
TEST(Compare, NearDimension0)
{
    {
        const double abs_error = 0.00003;
        const float lhs = -4.12451f;
        const float rhs = -4.12453f;
        ASSERT_TRUE(gcuda::detail::compareNear(lhs, rhs, abs_error));
    }
    {
        const double abs_error = 0.000001;
        const double lhs = 0.00002311;
        const double rhs = 0.00002312;
        ASSERT_TRUE(gcuda::detail::compareNear(lhs, rhs, abs_error));
    }
    {
        const double abs_error = 0.0001;
        const float lhs = -4.1245f;
        const float rhs = 4.1246f;
        ASSERT_FALSE(gcuda::detail::compareNear(lhs, rhs, abs_error));
    }
    {
        const double abs_error = 0.0000000001;
        const double lhs = 0.000002311;
        const double rhs = 0.000002313;
        ASSERT_FALSE(gcuda::detail::compareNear(lhs, rhs, abs_error));
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

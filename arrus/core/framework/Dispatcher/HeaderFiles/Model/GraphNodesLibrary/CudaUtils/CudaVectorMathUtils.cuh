#ifndef __CUDA_VECTOR_MATH_UTILS__
#define __CUDA_VECTOR_MATH_UTILS__

#include <cuda_runtime.h>

inline __host__ __device__ float2 make_float2(float s) {
    return make_float2(s, s);
}

inline __host__ __device__ float2 operator-(float2 &a) {
    return make_float2(-a.x, -a.y);
}

inline __host__ __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(float2 &a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ void operator-=(float2 &a, float2 b) {
    a.x -= b.x;
    a.y -= b.y;
}

inline __host__ __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ float2 operator*(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}

inline __host__ __device__ float2 operator*(float s, float2 a) {
    return make_float2(a.x * s, a.y * s);
}

inline __host__ __device__ void operator*=(float2 &a, float s) {
    a.x *= s;
    a.y *= s;
}

inline __host__ __device__ float2 operator/(float2 a, float2 b) {
    return make_float2(a.x / b.x, a.y / b.y);
}

inline __host__ __device__ float2 operator/(float2 a, float s) {
    float inv = 1.0f / s;
    return a * inv;
}

inline __host__ __device__ float2 operator/(float s, float2 a) {
    float inv = 1.0f / s;
    return a * inv;
}

inline __host__ __device__ void operator/=(float2 &a, float s) {
    float inv = 1.0f / s;
    a *= inv;
}

template<typename T>
__device__ __forceinline__ T makeZeroValue() {
    return T(0);
}

template<>
__device__ __forceinline__ float2 makeZeroValue<float2>() {
    return make_float2(0.0f);
}

#endif

#include <cstdio>

typedef double dtype;

inline unsigned divup(unsigned x, unsigned div) {
    return (x + div-1) / div;
}


__global__ void padHalfWithZeros(double* dst, double* src,
                                 int dstW, int dstH,
                                 int srcW, int srcH, int depth)
{
    const int x = blockDim.x*blockIdx.x + threadIdx.x;
    const int y = blockDim.y*blockIdx.y + threadIdx.y;
    const int z = blockDim.z*blockIdx.z + threadIdx.z;
    const int dstIdx = z + y*depth + x*depth*dstH;

    const int leftBorderX = (dstW-srcW)/2;
    const int rightBorderX = leftBorderX+srcW;

    const int leftBorderY = (dstH-srcH)/2;
    const int rightBorderY = leftBorderY+srcH;

    // TODO(pjarosik) grid-stride loop
    if(x < dstW && y < dstH && z < depth) {
        if (x >= leftBorderX && x < rightBorderX &&
            y >= leftBorderY && y < rightBorderY) {

            const int srcIdx = z+(y-leftBorderY)*depth+(x-leftBorderX)*depth*srcH;
            dst[dstIdx] = src[srcIdx];
        }
        else {
            dst[dstIdx] = 0.0;
        }
    }
}

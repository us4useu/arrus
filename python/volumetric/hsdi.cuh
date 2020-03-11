#include <cstdio>
#include <math.h>
#include <cufft.h>

typedef double dtype;
typedef cufftDoubleComplex complexType;

inline unsigned divup(unsigned x, unsigned div) {
    return (x + div-1) / div;
}


__global__ void padHalfWithZeros(dtype* dst, dtype* src,
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

// :param dkz: a change in a single dkz axis
__global__ void interpWeight(complexType* dst, complexType* src,
                             int dstW, int dstH, int dstD,
                             int srcW, int srcH, int srcD,
                             dtype speedOfSound,
                             dtype dkx, dtype dky, dtype dkz,
                             dtype df)
{
    const int x = blockDim.x*blockIdx.x + threadIdx.x;
    const int y = blockDim.y*blockIdx.y + threadIdx.y;
    const int z = blockDim.z*blockIdx.z + threadIdx.z;
    const int halfW = srcW/2;
    const int halfH = srcH/2;
    const int dstIdx = z + y*dstD + x*dstD*dstH;

    if(x < dstW && y < dstH && z < dstD) {
        if(z == 0) {
            dst[dstIdx] = {0.0, 0.0};
        }
        else {
            // TODO(pjarosik) consider moving to shared memory
            dtype kx = dkx*(-halfW+(x+halfW)%srcW);
            dtype ky = dky*(-halfH+(y+halfH)%srcH);
            dtype kz = dkz*z;

            dtype sample = speedOfSound/(4*M_PI)*((kz*kz+kx*kx+ky*ky)/kz);
            
            // Sample at df should be equal one (just cast frequencies to
            // index numbers).
            sample = sample/df;
            dtype w = speedOfSound/(4*M_PI)*((kz*kz-kx*kx-ky*ky)/kz);

            // linear interp.
            int z1 = int(sample);
            if(z1 >= srcD) {
                dst[dstIdx] = {0.0, 0.0};
                return;
            }
            int z2 = z1+1;
            if(z2 >= srcD) {
                dst[dstIdx] = src[z1+y*srcD+x*srcD*srcH];
                return;
            }
            dtype alpha = sample-z1;
            complexType y1 = src[z1+y*srcD+x*srcD*srcH];
            complexType y2 = src[z2+y*srcD+x*srcD*srcH];
            dst[dstIdx] = {w*((1-alpha)*y1.x+alpha*y2.x),
                           w*((1-alpha)*y1.y+alpha*y2.y)};
        }
    }
}

#include <fstream>
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <stdio.h>

// TODO replace int64_t with size_t

// Computes position of given component, relative to the first component of
// given entity.
static __device__ __forceinline__
float get_relative_position(const int componentNo,
                            const int componentsCount,
                            const float entityWidth) {
    return (float) componentNo / (float) (componentsCount - 1) * entityWidth;
}

static __device__ __forceinline__ float sq(const float x) {
    return x * x;
}

static __device__ __forceinline__
float euclidean_distance(const float x1,
                         const float y1,
                         const float x2,
                         const float y2) {
    return sqrtf(sq(x1 - x2) + sq(y1 - y2));
}

static __device__ __forceinline__
float time_lapse(const float distance,
                 const float speedOfSound) {
    return distance / speedOfSound;
}

template<class T>
static __device__ __forceinline__
float
interpolate_1d(const T *input, const float approxSampleNo) {
    int sampleNoFloor = floorf(approxSampleNo);
    float ratio = approxSampleNo - sampleNoFloor;
    return (1.0f - ratio) * input[sampleNoFloor]
           + ratio * input[sampleNoFloor + 1];
}

template<typename T>
__forceinline__ __device__ float
delay_and_sum_and_interpolate(
        const T *input,
        const int64_t receiversCount,
        const int64_t samplesCount,
        const float speedOfSound,
        const float receiverWidth,
        const float samplingFrequency,
        const float startDepth,
        const float transmitDistance,
        const float outputX,
        const float outputY
) {
    float result = float(0);
    for(int r = 0; r < receiversCount; ++r) {
        float receiverX = get_relative_position(r, receiversCount,
                                                receiverWidth);
        float receiveDistance = euclidean_distance(outputX, outputY, receiverX,
                                                   0);
        float totalDistance = transmitDistance + receiveDistance;
        // approximate sample number
        float approxSampleNo =
                time_lapse(totalDistance - startDepth, speedOfSound)
                * samplingFrequency;

        float recvImpact = float(0);
        auto sampleNo = (int) approxSampleNo;
        if((sampleNo < samplesCount) && (sampleNo >= 0)) {
            const int64_t inputOffset = samplesCount * r;
            recvImpact = interpolate_1d(input + inputOffset, approxSampleNo);
        }
        result += recvImpact;
    }
    return result;
}

// Coordinate system:
// OX(C) --> rec_1, rec_2, ....
// OY(S)     s1_1   s2_1
// |t1       s1_2   ...
// |t2       s1_3
// .t3       ...
template<typename T>
__global__ void sta_with_focusing_gpu(
        const T *input,
        const int64_t receiversCount,
        const int64_t samplesCount,
        const float speedOfSound,
        // rx subaperture size [m]
        const float receiverWidth,
        const float samplingFrequency,
        const float areaHeight,
        const float startDepth,
        // rozmiar wyjsciowy w liczbie pikseli/probek
        const int64_t outputHeight,
        const int64_t outputWidth,
        float *output) {
    // Here is computed one pixel output[y][x].
    // printf("Calling\n");
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // (ph)ysical (x,y) coordinates of pixel computed by this thread
    float posX = receiverWidth / (float) 2;
    float posY = (float) y / (float) (outputHeight - 1) * areaHeight
                 + startDepth;

    float transmitDistance = posY;
    // below implies, that x < eventsCount
    // event to get based on x
    const int64_t inputOffset = receiversCount * samplesCount * x;

    float result = delay_and_sum_and_interpolate(
            input + inputOffset,
            receiversCount,
            samplesCount,
            speedOfSound,
            receiverWidth,
            samplingFrequency,
            startDepth,
            transmitDistance,
            posX,
            posY);
    // blockDim.x*gridDim.x == outputWidth, see STA implementation
    output[x + y * blockDim.x * gridDim.x] = result;
}

int main() {
    const size_t N_SAMPLES = 8192;
    const size_t N_RX_CHANNELS = 64;
    const size_t N_EVENTS = 192;

    // parameters:
    const int64_t outputWidth = N_EVENTS;
    const int64_t outputHeight = N_SAMPLES;
    const float speedOfSound = 1490;
    const float pitch = 0.30e-3;
    const float receiverWidth = N_RX_CHANNELS * pitch;
    const float samplingFrequency = 65e6;
    const float areaHeight = N_SAMPLES/samplingFrequency*speedOfSound/2;
    const float startDepth = 5e-3; // TODO precompute


    const size_t DATA_SIZE = N_EVENTS*N_RX_CHANNELS*N_SAMPLES;
    int16_t* inBuffer = new int16_t[DATA_SIZE];
    int16_t* gpuInBuffer;
    cudaMalloc(&gpuInBuffer, DATA_SIZE*sizeof(int16_t));

    float* outputBuffer = new float[N_EVENTS*N_SAMPLES];
    float* gpuOutputBuffer;
    cudaMalloc(&gpuOutputBuffer, N_EVENTS*N_SAMPLES*sizeof(float));

    std::ifstream ifile("/home/pjarosik/data/rf2.bin", std::ios::binary);
    ifile.read((char*) inBuffer, DATA_SIZE*sizeof(int16_t));

    cudaMemcpy(inBuffer, gpuInBuffer, DATA_SIZE*sizeof(int16_t),
               cudaMemcpyHostToDevice);

    // TODO use CUDA runtime heuristic
    dim3 blockDim(std::min<int64_t>(8, outputWidth),
                  std::min<int64_t>(32, outputHeight));
    dim3 gridDim(outputWidth / blockDim.x, outputHeight / blockDim.y);

    std::cout << "Running kernel" << std::endl;

    float totalTime = 0;

    size_t nIt = 1;

    for(int i = 0; i < nIt; ++i) {

        auto start = std::chrono::steady_clock::now();

        sta_with_focusing_gpu<int16_t> <<<gridDim, blockDim>>> (
           gpuInBuffer, N_RX_CHANNELS, N_SAMPLES, speedOfSound, receiverWidth,
           samplingFrequency, areaHeight, startDepth, outputHeight,
           outputWidth, gpuOutputBuffer);
        cudaDeviceSynchronize();
        cudaMemcpy(gpuOutputBuffer, outputBuffer, N_EVENTS*N_SAMPLES*sizeof(float),
                   cudaMemcpyDeviceToHost);
        auto end = std::chrono::steady_clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    std::cout << "Total time: " << totalTime << std::endl;
    std::cout << "Average time: " << totalTime/(float)nIt << std::endl;

    std::cout << "Done running the kernel" << std::endl;

    cudaMemcpy(gpuOutputBuffer, outputBuffer, N_EVENTS*N_SAMPLES*sizeof(int16_t),
               cudaMemcpyDeviceToHost);

}



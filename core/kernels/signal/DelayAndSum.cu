#include <fstream>
#include <cstdint>

// Computes position of given component, relative to the first component of
// given entity.
static __device__ __forceinline__
float get_relative_position(const int componentNo,
                            const int componentsCount,
                            const float entityWidth) {
    return (float) componentNo / (float) (componentsCount - 1) * entityWidth;
}

static __device__ __forceinline__
float time_lapse(const float distance,
                 const float speedOfSound) {
    return distance / speedOfSound;
}

template<class T>
static __device__ __forceinline__
T
interpolate_1d(const T *input, const float approxSampleNo) {
    int sampleNoFloor = floorf(approxSampleNo);
    float ratio = approxSampleNo - sampleNoFloor;
    return (1.0f - ratio) * input[sampleNoFloor]
           + ratio * input[sampleNoFloor + 1];
}

template<typename T>
__forceinline__ __device__ T
delay_and_sum_and_interpolate(
        const T *input,
        const int64 receiversCount,
        const int64 samplesCount,
        const float speedOfSound,
        const float receiverWidth,
        const float samplingFrequency,
        const float startDepth,
        const float transmitDistance,
        const float outputX,
        const float outputY
) {
    T result = T(0);
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

        T recvImpact = T(0);
        auto sampleNo = (int) approxSampleNo;
        if((sampleNo < samplesCount) && (sampleNo >= 0)) {
            const int64 inputOffset = samplesCount * r;
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
        const int64 receiversCount,
        const int64 samplesCount,
        const float speedOfSound,
        // rx subaperture size [m]
        const float receiverWidth,
        const float samplingFrequency,
        const float areaHeight,
        const float startDepth,
        // rozmiar wyjsciowy w liczbie pikseli/probek
        const int64 outputHeight,
        const int64 outputWidth,
        T *output) {
    // Here is computed one pixel output[y][x].
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // (ph)ysical (x,y) coordinates of pixel computed by this thread
    float posX = receiverWidth / (float) 2;
    float posY = (float) y / (float) (outputHeight - 1) * areaHeight
                 + startDepth;

    float transmitDistance = posY;
    // below implies, that x < eventsCount
    // event to get based on x
    const int64 inputOffset = receiversCount * samplesCount * x;

    T result = delay_and_sum_and_interpolate(
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

    const size_t N_SAMPLES = 4096;
    const size_t N_RX_CHANNELS = 64;
    const size_t N_EVENTS = 192;

    const size_t DATA_SIZE = N_EVENTS*N_RX_CHANNELS*N_SAMPLES;
    int16_t buffer = new int16_t[DATA_SIZE];
    std::ifstream ifile("/home/pjarosik/data/data_rf.bin", std::ios::binary);
    ifile.read((char*) buffer, DATA_SIZE*sizeof(int16_t));

    // parameters:
    const int64 outputWidth = N_EVENTS;
    const int64 outputHeight = N_SAMPLES;
    const float speedOfSound = 1490;
    const float pitch = 0.30e-3;
    const float receiverWidth = N_RX_CHANNELS * pitch;
    const float samplingFrequency = 65e6;
    const float areaHeight = N_SAMPLES/samplingFrequency*speedOfSound/2;
    const float startDepth =

    // TODO use CUDA runtime heuristic
    dim3 blockDim(std::min<int64>(8, outputWidth),
                  std::min<int64>(32, outputHeight));
    dim3 gridDim(outputWidth / blockDim.x, outputHeight / blockDim.y);

    sta_with_focusing_gpu<T> <<< gridDim, blockDim, 0>>> (
            input, N_RX_CHANNELS, N_SAMPLES, speedOfSound, receiverWidth,
                    samplingFrequency, areaHeight, startDepth, outputHeight,
                    outputWidth, output
    );

}

template<typename T>
void STA<GPUDevice, T>::operator()(const GPUDevice &d,
                                   const T *input,
                                   const int64 receiversCount,
                                   const int64 samplesCount,
                                   const float speedOfSound,
                                   const float receiverWidth,
                                   const float samplingFrequency,
                                   const float startDepth,
                                   const float areaHeight,
                                   const int64 outputHeight,
                                   const int64 outputWidth,
                                   T *output) {

}


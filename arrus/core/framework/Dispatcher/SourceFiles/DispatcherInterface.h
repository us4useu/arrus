#pragma once

#ifdef WIN32
#ifdef DISPATCHERDLL_EXPORTS
#define DISPATCHERDLL_API __declspec(dllexport) 
#else
#define DISPATCHERDLL_API __declspec(dllimport) 
#endif
#elif linux
#define DISPATCHERDLL_API
#else
#error Platform not supported.
#endif

#include <string>
#include <vector>
#include <cuda_runtime.h>

class DispatcherInterface {
public:
    virtual void startAsync() = 0;

    virtual void startOnce() = 0;

    virtual void stop() = 0;

    virtual void kill() = 0;

    /** Builds execution graph using JSON string. */
    virtual bool build(const std::string &json, const bool cascadeMode = false) = 0;

    virtual void buildAsync(const std::string &json, const bool cascadeMode = false) = 0;

    /** Binds callback function to particular node in execution graph. */
    virtual void registerCallback(const int id,
                                  void(*callback)(void *data, int iterationId, int graphNodeId, int dimX, int dimY,
                                                  int dimZ, int dataType)) = 0;

    virtual void
    setInputData(short *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                 const unsigned int dimZ = 1) = 0;

    virtual void
    setInputData(int *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                 const unsigned int dimZ = 1) = 0;

    virtual void
    setInputData(float *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                 const unsigned int dimZ = 1) = 0;

    virtual void
    setInputData(double *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                 const unsigned int dimZ = 1) = 0;

    virtual void
    setInputData(float2 *dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY = 1,
                 const unsigned int dimZ = 1) = 0;

    virtual void
    updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName, const bool parameterValue) = 0;

    virtual void
    updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName, const int parameterValue) = 0;

    virtual void
    updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName, const float parameterValue) = 0;

    virtual void updateGraphNodeParameterAsync(const int nodeId, const std::string &parameterName,
                                               const std::string &parameterValue) = 0;

    virtual std::vector <std::string> getAvailableDevicesNames() = 0;

    virtual void setDevice(const std::string &deviceName) = 0;
};

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

EXTERN_C DISPATCHERDLL_API DispatcherInterface
*

getDispatcherInstance();

EXTERN_C DISPATCHERDLL_API void releaseDispatcherInstance(DispatcherInterface * dispatcher);

// C Interface
EXTERN_C
{
DISPATCHERDLL_API void startAsync(DispatcherInterface * dispatcher);
DISPATCHERDLL_API void startOnce(DispatcherInterface * dispatcher);
DISPATCHERDLL_API void stop(DispatcherInterface * dispatcher);
DISPATCHERDLL_API void kill(DispatcherInterface * dispatcher);

DISPATCHERDLL_API bool build(DispatcherInterface * dispatcher,
const char *json,
const bool cascadeMode = false
);
DISPATCHERDLL_API void buildAsync(DispatcherInterface * dispatcher,
const char *json,
const bool cascadeMode = false
);

DISPATCHERDLL_API void registerCallback(DispatcherInterface * dispatcher,
const int id,

void (*callback)(void *data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ, int dataType)

);

DISPATCHERDLL_API void setInputDataShort(DispatcherInterface * dispatcher, short * dataPtr,
const int numberOfBatches,
const unsigned int dimX,
const unsigned int dimY,
const unsigned int dimZ
);
DISPATCHERDLL_API void setInputDataInt(DispatcherInterface * dispatcher, int * dataPtr,
const int numberOfBatches,
const unsigned int dimX,
const unsigned int dimY,
const unsigned int dimZ
);
DISPATCHERDLL_API void setInputDataFloat(DispatcherInterface * dispatcher, float * dataPtr,
const int numberOfBatches,
const unsigned int dimX,
const unsigned int dimY,
const unsigned int dimZ
);
DISPATCHERDLL_API void setInputDataDouble(DispatcherInterface * dispatcher, double * dataPtr,
const int numberOfBatches,
const unsigned int dimX,
const unsigned int dimY,
const unsigned int dimZ
);
DISPATCHERDLL_API void setInputDataFloat2(DispatcherInterface * dispatcher, float2 * dataPtr,
const int numberOfBatches,
const unsigned int dimX,
const unsigned int dimY,
const unsigned int dimZ
);

DISPATCHERDLL_API void updateGraphNodeParameterAsyncBool(DispatcherInterface * dispatcher,
const int nodeId,
const char *parameterName,
const bool parameterValue
);
DISPATCHERDLL_API void updateGraphNodeParameterAsyncInt(DispatcherInterface * dispatcher,
const int nodeId,
const char *parameterName,
const int parameterValue
);
DISPATCHERDLL_API void updateGraphNodeParameterAsyncFloat(DispatcherInterface * dispatcher,
const int nodeId,
const char *parameterName,
const float parameterValue
);
DISPATCHERDLL_API void updateGraphNodeParameterAsyncChar(DispatcherInterface * dispatcher,
const int nodeId,
const char *parameterName,
const char *parameterValue
);

DISPATCHERDLL_API int getAvailableDevicesNames(DispatcherInterface * dispatcher, char * **listOfNames);
DISPATCHERDLL_API void setDevice(DispatcherInterface * dispatcher,
const char *deviceName
);
}

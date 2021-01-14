#pragma once

#include <unordered_map>
#include <boost/any.hpp>
#include <boost/variant.hpp>
#include <cuda_runtime.h>

class PluginInterface {
public:
    virtual void setNodeVariables(const std::unordered_map <std::string, boost::any> nodeVariables) = 0;

    virtual void
    setInputDataPointerProperties(const std::unordered_map <std::string, boost::any> pointerProperties) = 0;

    virtual void
    setInputDataPointer(const boost::variant<short *, int *, float *, double *, float2 *> ptr, const int dimX,
                        const int dimY, const int dimZ) = 0;

    virtual const std::tuple<boost::variant<short *, int *, float *, double *, float2 *>, int, int, int>
    getOutputDataPointer() = 0;

    virtual const std::unordered_map <std::string, boost::any> getOutputDataPointerProperties() = 0;

    virtual void process(cudaStream_t &defaultStream) = 0;
};
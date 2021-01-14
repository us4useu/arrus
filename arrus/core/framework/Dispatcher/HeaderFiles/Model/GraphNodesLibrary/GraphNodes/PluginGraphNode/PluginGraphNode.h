#pragma once

#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include "Model/GraphNodesLibrary/GraphNodesFactory.h"
#include "Model/GraphNodesLibrary/GraphNodes/PluginGraphNode/PluginInterface.h"
#include <Windows.h>

class PluginGraphNode : public GraphNode {
private:
    static GraphNodesFactoryRegister <PluginGraphNode> graphNodesFactoryRegister;
    std::unique_ptr <PluginInterface> plugin;
    HINSTANCE dllModule;

    void loadPlugin();

    void setPluginInputDataPointerProperties();

    void setPluginNodeVariables();

    void getOutputDataPointerPropertiesFromPlugin();

public:
    PluginGraphNode();

    ~PluginGraphNode();

    void process(cudaStream_t &defaultStream);
};


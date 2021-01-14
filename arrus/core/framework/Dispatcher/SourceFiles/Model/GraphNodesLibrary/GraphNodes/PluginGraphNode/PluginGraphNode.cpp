#include "Model/GraphNodesLibrary/GraphNodes/PluginGraphNode/PluginGraphNode.h"
#include "Utils/DispatcherLogger.h"

typedef PluginInterface *(__stdcall *f_funci)();

GraphNodesFactoryRegister <PluginGraphNode> PluginGraphNode::graphNodesFactoryRegister("plugin");

PluginGraphNode::PluginGraphNode() {
    this->plugin = nullptr;
}


PluginGraphNode::~PluginGraphNode() {
    FreeLibrary(dllModule);
}

void PluginGraphNode::loadPlugin() {
    if(this->plugin == nullptr) {
        std::string dllPath = this->getNodeVariable("dllFile").getValue<std::string>();
        this->dllModule = LoadLibrary(dllPath.c_str());

        if(!dllModule)
            DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Could not load the dynamic library: ") + dllPath);

        f_funci funci = (f_funci) GetProcAddress(dllModule, "getPluginInstance");
        if(!funci)
            DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Could not locate function: getPluginInstance"));

        this->plugin = std::unique_ptr<PluginInterface>(funci());
    }
}

void PluginGraphNode::setPluginInputDataPointerProperties() {
    std::unordered_map <propertyName, VariableAnyValue> ptrProperties = this->inputData.getPtrProperties();
    std::unordered_map <propertyName, boost::any> rawPtrProperties;
    for(std::pair <propertyName, VariableAnyValue> currPair : ptrProperties)
        rawPtrProperties[currPair.first] = currPair.second.getExpandedAnyValue();
    this->plugin->setInputDataPointerProperties(rawPtrProperties);
}

void PluginGraphNode::getOutputDataPointerPropertiesFromPlugin() {
    for(std::pair <propertyName, boost::any> currPair : this->plugin->getOutputDataPointerProperties())
        this->outputData.setPtrProperty(currPair.first, VariableAnyValue(currPair.second));
}

void PluginGraphNode::setPluginNodeVariables() {
    std::unordered_map <variableName, boost::any> rawNodeVariables;
    for(std::pair <variableName, VariableAnyValue> currPair : this->nodeVariables)
        rawNodeVariables[currPair.first] = currPair.second.getExpandedAnyValue();
    this->plugin->setNodeVariables(rawNodeVariables);
}

void PluginGraphNode::process(cudaStream_t &defaultStream) {
    this->loadPlugin();
    this->plugin->setInputDataPointer(this->inputData.getRawPtr(), this->inputData.getDims().x,
                                      this->inputData.getDims().y, this->inputData.getDims().z);
    this->setPluginInputDataPointerProperties();
    this->setPluginNodeVariables();

    this->plugin->process(defaultStream);

    std::tuple<boost::variant<short *, int *, float *, double *, float2 *>, int, int, int> outputDataPointer = this->plugin->getOutputDataPointer();
    this->outputData = DataPtr(std::get<0>(outputDataPointer),
                               Dims(std::get<1>(outputDataPointer), std::get<2>(outputDataPointer),
                                    std::get<3>(outputDataPointer)));
    this->getOutputDataPointerPropertiesFromPlugin();
}
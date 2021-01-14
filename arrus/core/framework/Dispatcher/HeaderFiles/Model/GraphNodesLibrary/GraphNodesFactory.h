#pragma once

#include <unordered_map>
#include <boost/function.hpp>
#include <memory>
#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"

typedef std::string nodeNameType;
typedef std::unordered_map <nodeNameType, boost::function0<std::shared_ptr < GraphNode>>>
nodesMapType;

class GraphNodesFactory {
protected:
    static std::shared_ptr <nodesMapType> nodesConstructors;

public:
    static std::shared_ptr <GraphNode> createGraphNode(const nodeNameType &nodeName);
};

template<typename T>
std::shared_ptr <GraphNode> createT() { return std::shared_ptr<T>(new T); }

template<typename T>
class GraphNodesFactoryRegister : public GraphNodesFactory {
public:
    GraphNodesFactoryRegister(const std::string &nodeName) {
        nodesConstructors->insert(std::make_pair(nodeName, &createT<T>));
    }
};


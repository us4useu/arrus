#pragma once

#include <unordered_map>
#include <memory>
#include "Model/GraphNodesLibrary/GraphNodes/GraphNode.h"
#include <boost/thread/mutex.hpp>

typedef int nodeId;
typedef std::unordered_map <nodeId, std::shared_ptr<GraphNode>> graphNodesMapType;

typedef std::string parameterName;
typedef VariableAnyValue parameterValue;
typedef std::pair <nodeId, parameterName> pendingGraphNodesUpdatesMapKey;

struct UpdatesMapKeysHashFunction {
    size_t operator()(const pendingGraphNodesUpdatesMapKey &key) const {
        size_t h1 = std::hash<nodeId>()(key.first);
        size_t h2 = std::hash<parameterName>()(key.second);
        return h1 ^ (h2 << 1);
    };
};

struct UpdatesMapKeysEqual {
    bool
    operator()(const pendingGraphNodesUpdatesMapKey &firstKey, const pendingGraphNodesUpdatesMapKey &secondKey) const {
        return (firstKey.first == secondKey.first) && (firstKey.second.compare(secondKey.second) == 0);
    }
};

typedef std::unordered_map <pendingGraphNodesUpdatesMapKey, parameterValue, UpdatesMapKeysHashFunction, UpdatesMapKeysEqual> pendingGraphNodesUpdatesMap;

#define STARTING_NODE_ID 0

class GraphNodesLibrary {
private:
    mutable boost::mutex mutex;
    pendingGraphNodesUpdatesMap pendingUpdates;
    graphNodesMapType graphNodes;
public:
    GraphNodesLibrary();

    ~GraphNodesLibrary();

    void registerNode(const nodeId id, const std::shared_ptr <GraphNode> node);

    std::shared_ptr <GraphNode> getNode(const nodeId id);

    void connectNodes(const nodeId srcId, const nodeId dstId);

    void updateGraphNodeParameter(const nodeId id, const parameterName &name, const parameterValue &value);

    void applyGraphNodesUpdates();
};


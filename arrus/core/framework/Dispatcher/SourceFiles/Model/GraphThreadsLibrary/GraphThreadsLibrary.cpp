#include "Model/GraphThreadsLibrary/GraphThreadsLibrary.h"
#include <cuda_runtime.h>

GraphThreadsLibrary::GraphThreadsLibrary()
{

}

GraphThreadsLibrary::GraphThreadsLibrary(const int cudaDeviceCount)
{	
	for (int i = 0; i < cudaDeviceCount; ++i)
	{
		this->graphThreads.push_back(std::unique_ptr<GraphThread>(new GraphThread(i, this)));
		this->graphThreads[i]->start();
	}
}

GraphThreadsLibrary::~GraphThreadsLibrary()
{
}

void GraphThreadsLibrary::updateGraphNodesLibrary(const int id)
{
	this->graphNodesLibraries[id]->applyGraphNodesUpdates();
}

void GraphThreadsLibrary::suspendThread()
{
	boost::recursive_mutex::scoped_lock lock(this->activeThreadsMutex);

	this->activeThreadsNumber -= 1;

	if (this->activeThreadsNumber == 0)
	{
		this->updateGraphNodesLibrary(0);
		this->startThread(this->graphNodesLibraries[0]->getNode(STARTING_NODE_ID), 0);
	}
}

void GraphThreadsLibrary::startThread(const std::shared_ptr<GraphNode> startingNode, const int threadId)
{
	boost::recursive_mutex::scoped_lock lock(this->activeThreadsMutex);

	this->activeThreadsNumber += 1;
	this->graphThreads[threadId]->startFromNode(startingNode);
}

void GraphThreadsLibrary::plainStart()
{	
	this->activeThreadsNumber = 0;
	this->startThread(this->graphNodesLibraries[0]->getNode(STARTING_NODE_ID), 0);
}

void GraphThreadsLibrary::startCascade()
{
	for (int i = 0; i < this->graphThreads.size(); ++i)
		this->graphThreads[i]->continousWorkFromNode(this->graphNodesLibraries[i]->getNode(STARTING_NODE_ID));
}

void GraphThreadsLibrary::start()
{
	if (this->graphNodesLibraries.size() != 1)
		this->startCascade();
	else
		this->plainStart();
}

void GraphThreadsLibrary::stop()
{
	for (int i = 0; i < this->graphThreads.size(); ++i)
		this->graphThreads[i]->join();
}

void GraphThreadsLibrary::setGraphNodesLibraries(const std::vector<std::shared_ptr<GraphNodesLibrary>>& graphNodesLibraries)
{
	this->graphNodesLibraries = graphNodesLibraries;
}

void GraphThreadsLibrary::kill()
{
	for (int i = 0; i < this->graphThreads.size(); ++i)
		this->graphThreads[i]->kill();
}
#include "Model/GraphThreadsLibrary/GraphThread.h"
#include "Model/GraphThreadsLibrary/GraphThreadsLibrary.h"
#include <boost/chrono.hpp>

GraphThread::GraphThread(const int id, GraphThreadsLibrary* graphThreadsLibrary)
{
	this->isContinousWork = false;
	this->myMutex.lock();
	this->myThreadId = id;
	this->graphThreadsLibrary = graphThreadsLibrary;
	CUDA_ASSERT( cudaSetDevice(this->myThreadId) );
	CUDA_ASSERT( cudaStreamCreate(&this->defaultStream) );
}


GraphThread::~GraphThread()
{
	CUDA_ASSERT( cudaStreamDestroy(this->defaultStream) );
}

void GraphThread::iterateOverGraph()
{
	// iteration without branches
	std::shared_ptr<GraphNode> currentNode = this->startingNode;
	DataPtr data;
	while (currentNode != nullptr)
	{
		currentNode->setInputData(data);
		currentNode->process(this->defaultStream);
		if (currentNode->getSuccessors().empty())
			currentNode = nullptr;
		else
		{
			data = currentNode->getOutputData();
			currentNode = currentNode->getSuccessors().front();
		}
	}
}

void GraphThread::process()
{	
	CUDA_ASSERT(cudaSetDevice(this->myThreadId));

	while (this->isWorking)
	{
		this->myMutex.lock();

		this->iterateOverGraph();

		// end of path
		if (this->isContinousWork)
		{
			this->graphThreadsLibrary->updateGraphNodesLibrary(this->myThreadId);
			this->myMutex.unlock();
		}
		else
			this->graphThreadsLibrary->suspendThread();
	}
}

void GraphThread::start()
{
	this->isWorking = true;
	this->myThread = boost::thread(&GraphThread::process, this);
}

void GraphThread::join()
{
	this->isWorking = false;
	this->startingNode = nullptr;
	this->myMutex.unlock();
	boost::chrono::duration<int, boost::milli> tryJoinTime(1000);
	if (!this->myThread.try_join_for(tryJoinTime))
		this->myThread.interrupt();
}

void GraphThread::startFromNode(const std::shared_ptr<GraphNode> node)
{
	this->isContinousWork = false;
	this->startingNode = node;
	this->myMutex.unlock();
}

void GraphThread::continousWorkFromNode(const std::shared_ptr<GraphNode> node)
{
	this->startingNode = node;
	this->isContinousWork = true;
	this->myMutex.unlock();
}

void GraphThread::kill()
{
	this->myThread.interrupt();
}

#include "Dispatcher.h"
#include "DispatcherInterface.h"

DispatcherInterface* getDispatcherInstance()
{
	return new Dispatcher();
}

void releaseDispatcherInstance(DispatcherInterface* dispatcher)
{
	delete dispatcher;
}

void startAsync(DispatcherInterface* dispatcher)
{
	dispatcher->startAsync();
}

void startOnce(DispatcherInterface* dispatcher)
{
	dispatcher->startOnce();
}

void stop(DispatcherInterface* dispatcher)
{
	dispatcher->stop();
}

void kill(DispatcherInterface* dispatcher)
{
	dispatcher->kill();
}

bool build(DispatcherInterface* dispatcher, const char* json, const bool cascadeMode)
{
	return dispatcher->build(std::string(json), cascadeMode);
}

void buildAsync(DispatcherInterface* dispatcher, const char* json, const bool cascadeMode)
{
	dispatcher->buildAsync(std::string(json), cascadeMode);
}

void registerCallback(DispatcherInterface* dispatcher, const int id, void(*callback)(void* data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ, int dataType))
{
	dispatcher->registerCallback(id, callback);
}

void setInputDataShort(DispatcherInterface* dispatcher, short* dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY, const unsigned int dimZ)
{
	dispatcher->setInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void setInputDataInt(DispatcherInterface* dispatcher, int* dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY, const unsigned int dimZ)
{
	dispatcher->setInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void setInputDataFloat(DispatcherInterface* dispatcher, float* dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY, const unsigned int dimZ)
{
	dispatcher->setInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void setInputDataDouble(DispatcherInterface* dispatcher, double* dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY, const unsigned int dimZ)
{
	dispatcher->setInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void setInputDataFloat2(DispatcherInterface* dispatcher, float2* dataPtr, const int numberOfBatches, const unsigned int dimX, const unsigned int dimY, const unsigned int dimZ)
{
	dispatcher->setInputData(dataPtr, numberOfBatches, dimX, dimY, dimZ);
}

void updateGraphNodeParameterAsyncBool(DispatcherInterface* dispatcher, const int nodeId, const char* parameterName, const bool parameterValue)
{
	dispatcher->updateGraphNodeParameterAsync(nodeId, std::string(parameterName), parameterValue);
}

void updateGraphNodeParameterAsyncInt(DispatcherInterface* dispatcher, const int nodeId, const char* parameterName, const int parameterValue)
{
	dispatcher->updateGraphNodeParameterAsync(nodeId, std::string(parameterName), parameterValue);
}

void updateGraphNodeParameterAsyncFloat(DispatcherInterface* dispatcher, const int nodeId, const char* parameterName, const float parameterValue)
{
	dispatcher->updateGraphNodeParameterAsync(nodeId, std::string(parameterName), parameterValue);
}

void updateGraphNodeParameterAsyncChar(DispatcherInterface* dispatcher, const int nodeId, const char* parameterName, const char* parameterValue)
{
	dispatcher->updateGraphNodeParameterAsync(nodeId, std::string(parameterName), std::string(parameterValue));
}

int getAvailableDevicesNames(DispatcherInterface* dispatcher, char*** listOfNames)
{
	std::vector<std::string> names = dispatcher->getAvailableDevicesNames();
	int namesNumber = (int)names.size();

	char** cNames = new char*[namesNumber];
	for (int i = 0; i < namesNumber; ++i)
	{
		cNames[i] = new char[names[i].length() + 1];
		memcpy(cNames[i], names[i].c_str(), sizeof(char) * (names[i].length() + 1));
	}
	*listOfNames = cNames;

	return namesNumber;
}

void setDevice(DispatcherInterface* dispatcher, const char* deviceName)
{
	dispatcher->setDevice(std::string(deviceName));
}
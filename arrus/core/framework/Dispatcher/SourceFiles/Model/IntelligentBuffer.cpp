#include "Model/IntelligentBuffer.h"

IntelligentBuffer::IntelligentBuffer()
{
	this->currReadElementIndex = 0;
	this->currWriteElementIndex = 0;
	this->writerExitFlag = false;
}

IntelligentBuffer::~IntelligentBuffer()
{
	CUDA_ASSERT(cudaDeviceSynchronize());
	for (int i = 0; i < this->dataBuffer.size(); ++i)
		if (this->dataBuffer[i].ptr.getAllocatedDataSize() != 0)
			CUDA_ASSERT(cudaHostUnregister(this->dataBuffer[i].ptr.getVoidPtr()));
}

void IntelligentBuffer::configure(int bufferSize)
{
	this->dataBuffer = std::vector<IntelligentBufferElement>(bufferSize);

	for (int i = 0; i < bufferSize; ++i)
        {
                DataPtr&& ptr = DataPtr();
                this->dataBuffer[i] = IntelligentBufferElement(ptr, ElementState::EMPTY);
        }
}

void IntelligentBuffer::unlockReader() 
{
	this->readCondition.notify_one();
}

void IntelligentBuffer::unlockWriter()
{
	this->writeCondition.notify_one();
}

void IntelligentBuffer::unlockWriterToExit()
{
	this->writerExitFlag = true;
	this->writeCondition.notify_one();
}

std::pair<elementIndex, DataPtr> IntelligentBuffer::getData()
{
	std::pair<elementIndex, DataPtr> data;
	{
		boost::mutex::scoped_lock lock(mutex);

		while (this->dataBuffer[this->currReadElementIndex].state != ElementState::FULL) // while - to guard against spurious wakeups
			this->readCondition.wait(lock);

		this->dataBuffer[this->currReadElementIndex].state = ElementState::BUSY;
		data = std::make_pair(this->currReadElementIndex, this->dataBuffer[this->currReadElementIndex].ptr);
		this->currReadElementIndex = (this->currReadElementIndex + 1) % this->dataBuffer.size();
	}
	return data;
}

void IntelligentBuffer::setDataEmpty(unsigned int index)
{
	{
		boost::mutex::scoped_lock lock(mutex);
		this->dataBuffer[index].state = ElementState::EMPTY;
		this->dataBuffer[index].freeFunc(this->dataBuffer[index].ptr);
	}
	this->unlockWriter();
}

void IntelligentBuffer::setData(DataPtr &ptr, std::function<void(DataPtr& ptr)> freeFunc)
{
	{
		boost::mutex::scoped_lock lock(mutex);
		while (this->dataBuffer[this->currWriteElementIndex].state != ElementState::EMPTY) // while - to guard against spurious wakeups
		{
			if (this->writerExitFlag)
				return;
			this->writeCondition.wait(lock);
		}

		if (this->dataBuffer[this->currWriteElementIndex].ptr.getAllocatedDataSize() == 0)
			CUDA_ASSERT(cudaHostRegister(ptr.getVoidPtr(), ptr.getDataSize(), cudaHostRegisterPortable));

		this->dataBuffer[this->currWriteElementIndex] = IntelligentBufferElement(ptr, ElementState::FULL, freeFunc);
		this->currWriteElementIndex = (this->currWriteElementIndex + 1) % this->dataBuffer.size();
	}
	this->unlockReader();
}

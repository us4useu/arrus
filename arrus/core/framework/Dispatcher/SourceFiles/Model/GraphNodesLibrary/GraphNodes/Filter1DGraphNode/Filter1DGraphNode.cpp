#include "Model/GraphNodesLibrary/GraphNodes/Filter1DGraphNode/Filter1DGraphNode.h"
#include "Utils/DispatcherLogger.h"

#include <fstream>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/classification.hpp>

#include <boost/numeric/ublas/matrix.hpp>

extern __device__ __constant__ float deviceFeedforwardCoefficients[];
extern __device__ __constant__ float deviceFeedbackCoefficients[];
extern __device__ __constant__ float deviceFeedbackCoefficientsMatrix[];

GraphNodesFactoryRegister<Filter1DGraphNode> Filter1DGraphNode::graphNodesFactoryRegister("filter1D");

Filter1DGraphNode::Filter1DGraphNode()
{
	this->feedforwardCoefficientsFileName = std::string("");
	this->feedbackCoefficientsFileName = std::string("");
	this->feedbackCoefficients.push_back(1.0f);

	this->setNodeVariable("feedforwardCoefficients", VariableAnyValue(std::string("")));
	this->setNodeVariable("feedbackCoefficients", VariableAnyValue(std::string("")));
}

Filter1DGraphNode::~Filter1DGraphNode()
{
	this->releaseGPUMemory(&this->outputData);
}

void Filter1DGraphNode::readFilterCoefficientsFromTxtFile(const std::string& pathToFile, std::vector<float>& filterCoefficients)
{
	std::ifstream ifs(pathToFile);
	if (!ifs.good())
	{
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Cannot open file with filter coefficients. File path: ") + pathToFile);
	}

	std::stringstream fileBuffer;
	fileBuffer << ifs.rdbuf();
	ifs.close();

	std::string fileData = fileBuffer.str();
	boost::replace_all(fileData, ",", " ");
	boost::replace_all(fileData, "\n", " ");
	boost::replace_all(fileData, ";", " ");

	std::vector<std::string> tokens;
	boost::split(tokens, fileData, boost::is_any_of(" "), boost::token_compress_on);

	BOOST_FOREACH(std::string t, tokens)
		filterCoefficients.push_back(std::stof(t));
}

void Filter1DGraphNode::readFilterCoefficientsFromBinaryFile(const std::string& pathToFile, std::vector<float>& filterCoefficients)
{
	std::ifstream ifs(pathToFile, std::ios::binary | std::ios::ate);
	if (!ifs.good()) 
	{
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Cannot open file with filter coefficients. File path: ") + pathToFile);
	}
	std::ifstream::pos_type pos = ifs.tellg();

	filterCoefficients.resize(pos / sizeof(float));
	ifs.seekg(0, std::ios::beg);
	ifs.read((char*)&filterCoefficients[0], pos);
}

void Filter1DGraphNode::readFilterCoefficientsFromFile(const std::string& pathToFile, std::vector<float>& filterCoefficients)
{
	std::size_t found = pathToFile.find(".txt");
	if (found != std::string::npos)
		this->readFilterCoefficientsFromTxtFile(pathToFile, filterCoefficients);
	else
	{
		found = pathToFile.find(".bin");
		if (found != std::string::npos)
			this->readFilterCoefficientsFromBinaryFile(pathToFile, filterCoefficients);
		else
			DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("File with filter coefficients has wrong extension. File path: ") + pathToFile);
	}
}

std::vector<float> Filter1DGraphNode::getNormalizedCoefficients(const std::vector<float>& inputCoeff, const float normalizeFactor)
{
	std::vector<float> normalizedCoeff(inputCoeff.size());
	std::transform(inputCoeff.begin(), inputCoeff.end(), normalizedCoeff.begin(), [normalizeFactor](const float a)->float { return a / normalizeFactor; });
	return normalizedCoeff;
}

void Filter1DGraphNode::buildFeedbackCoefficientsMatrix(const std::vector<float>& normFeedbackCoeff, cudaStream_t& defaultStream)
{
	int feedbackCoefficientsMatrixSize = (int)this->feedbackCoefficients.size() - 1;
	boost::numeric::ublas::matrix<float> feedbackCoefficientsMatrix = boost::numeric::ublas::zero_matrix<float>(feedbackCoefficientsMatrixSize, feedbackCoefficientsMatrixSize);
	for (int i = 0; i < feedbackCoefficientsMatrixSize - 1; ++i)
		feedbackCoefficientsMatrix(i + 1, i) = 1.0f;
	for (int i = 0; i < feedbackCoefficientsMatrixSize; ++i)
		feedbackCoefficientsMatrix(i, feedbackCoefficientsMatrixSize - 1) = -normFeedbackCoeff[feedbackCoefficientsMatrixSize - i];

	boost::numeric::ublas::matrix<float> initFeedbackCoefficientsMatrix(feedbackCoefficientsMatrix);
	// feedbackCoefficientsMatrix must be raised to the IIR_DATA_BLOCK_COUNT power
	for (int p = 0; p < IIR_DATA_BLOCK_COUNT - 1; ++p)
		feedbackCoefficientsMatrix = boost::numeric::ublas::prod(feedbackCoefficientsMatrix, initFeedbackCoefficientsMatrix);

	this->feedbackCoefficientsMatrixAsVector.clear();
	this->feedbackCoefficientsMatrixAsVector.resize(feedbackCoefficientsMatrixSize * feedbackCoefficientsMatrixSize);
	for (int i = 0; i < feedbackCoefficientsMatrixSize; ++i)
		for (int j = 0; j < feedbackCoefficientsMatrixSize; ++j)
			this->feedbackCoefficientsMatrixAsVector[i * feedbackCoefficientsMatrixSize + j] = feedbackCoefficientsMatrix(i, j);
}

void Filter1DGraphNode::loadFilterCoefficients(cudaStream_t& defaultStream)
{
	std::string pathToFileWithFeedforwardCoefficients = this->getNodeVariable("feedforwardCoefficients").getValue<std::string>();
	std::string pathToFileWithFeedbackCoefficients = this->getNodeVariable("feedbackCoefficients").getValue<std::string>();

	if (this->feedbackCoefficientsFileName.compare(pathToFileWithFeedbackCoefficients) != 0)
	{
		this->feedbackCoefficientsFileName = pathToFileWithFeedbackCoefficients;
		this->feedbackCoefficients.clear();
		this->readFilterCoefficientsFromFile(pathToFileWithFeedbackCoefficients, this->feedbackCoefficients);

		std::vector<float> normFeedbackCoeff = this->getNormalizedCoefficients(this->feedbackCoefficients, this->feedbackCoefficients[0]);
		this->buildFeedbackCoefficientsMatrix(normFeedbackCoeff, defaultStream);
	}

	if (this->feedforwardCoefficientsFileName.compare(pathToFileWithFeedforwardCoefficients) != 0)
	{
		this->feedforwardCoefficientsFileName = pathToFileWithFeedforwardCoefficients;
		this->feedforwardCoefficients.clear();
		this->readFilterCoefficientsFromFile(pathToFileWithFeedforwardCoefficients, this->feedforwardCoefficients);
	}
	
	// different instances of Filter1DGraphNode override their coefficients, so as a result they must be send to gpu every time
	std::vector<float> normFeedbackCoeff = this->getNormalizedCoefficients(this->feedbackCoefficients, this->feedbackCoefficients[0]);
	std::vector<float> normFeedforwardCoeff = this->getNormalizedCoefficients(this->feedforwardCoefficients, this->feedbackCoefficients[0]);
	CUDA_ASSERT(cudaMemcpyToSymbolAsync(deviceFeedbackCoefficients, &normFeedbackCoeff[0], sizeof(float) * normFeedbackCoeff.size(), 0, cudaMemcpyHostToDevice, defaultStream));
	if (!normFeedforwardCoeff.empty())
		CUDA_ASSERT(cudaMemcpyToSymbolAsync(deviceFeedforwardCoefficients, &normFeedforwardCoeff[0], sizeof(float) * normFeedforwardCoeff.size(), 0, cudaMemcpyHostToDevice, defaultStream));
	if (!this->feedbackCoefficientsMatrixAsVector.empty())
	{
		CUDA_ASSERT(cudaMemcpyToSymbolAsync(deviceFeedbackCoefficientsMatrix, &this->feedbackCoefficientsMatrixAsVector[0],
			sizeof(float) * this->feedbackCoefficientsMatrixAsVector.size(), 0, cudaMemcpyHostToDevice, defaultStream));
	}
}

void Filter1DGraphNode::process(cudaStream_t& defaultStream)
{
	bool iqData = this->inputData.getPtrProperty("iq").getValue<bool>();

	if (iqData)
		this->allocGPUMemory<float2>(&this->outputData, this->inputData.getDims());
	else
		this->allocGPUMemory<float>(&this->outputData, this->inputData.getDims());

	int batchLength = this->inputData.getDims().x;
	this->loadFilterCoefficients(defaultStream);

	std::string pathToFileWithFeedbackCoefficients = this->getNodeVariable("feedbackCoefficients").getValue<std::string>();
	bool isIIRFilter = (pathToFileWithFeedbackCoefficients.compare("") != 0);
	
	if (!isIIRFilter)
	{
		if (iqData)
			cudaFilter1DGraphNode.firIq(this->inputData.getPtr<float2*>(), this->outputData.getPtr<float2*>(), defaultStream,
									    (int)this->feedforwardCoefficients.size(), this->inputData.getDims().flatten(), batchLength);
		else
			cudaFilter1DGraphNode.fir(this->inputData.getPtr<float*>(), this->outputData.getPtr<float*>(), defaultStream,
									  (int)this->feedforwardCoefficients.size(), this->inputData.getDims().flatten(), batchLength);
	}
	else
	{
		if (iqData)
			cudaFilter1DGraphNode.iirIq(this->inputData.getPtr<float2*>(), this->outputData.getPtr<float2*>(), defaultStream,
									    (int)this->feedforwardCoefficients.size(), (int)this->feedbackCoefficients.size() - 1,
									    this->inputData.getDims().flatten(), batchLength);
		else
			cudaFilter1DGraphNode.iir(this->inputData.getPtr<float*>(), this->outputData.getPtr<float*>(), defaultStream,
									  (int)this->feedforwardCoefficients.size(), (int)this->feedbackCoefficients.size() - 1, 
									  this->inputData.getDims().flatten(), batchLength);
	}
	
	this->outputData.copyExtraData(this->inputData);
}

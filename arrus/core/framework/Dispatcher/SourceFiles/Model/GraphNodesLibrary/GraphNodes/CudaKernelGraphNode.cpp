#include "Model/GraphNodesLibrary/GraphNodes/CudaKernelGraphNode.h"
#include "Utils/DispatcherLogger.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>

GraphNodesFactoryRegister<CudaKernelGraphNode> CudaKernelGraphNode::graphNodesFactoryRegister("cudaKernel");

CudaKernelGraphNode::CudaKernelGraphNode()
{
	this->module = 0;
	this->isUserKernelCompiled = false;
	this->setNodeVariable("kernelFile", VariableAnyValue(std::string("test.cu")));
	this->setNodeVariable("sharedMemBytes", VariableAnyValue(0));
	this->setNodeVariable("grid", VariableAnyValue(std::vector<VariableAnyValue>()));
	this->setNodeVariable("block", VariableAnyValue(std::vector<VariableAnyValue>()));
	this->setNodeVariable("args", VariableAnyValue(std::vector<VariableAnyValue>()));
	std::vector<VariableAnyValue> compileOptions;
	compileOptions.push_back(VariableAnyValue(std::string("--gpu-architecture=compute_20")));
	this->setNodeVariable("compileOptions", VariableAnyValue(compileOptions));
	this->setNodeVariable("outputDims", VariableAnyValue(std::vector<VariableAnyValue>()));
	this->setNodeVariable("inputDataType", VariableAnyValue(std::string("float")));
	this->setNodeVariable("outputDataType", VariableAnyValue(std::string("float")));
}

CudaKernelGraphNode::~CudaKernelGraphNode()
{
	this->releaseGPUMemory(&this->outputData);
	if (this->module)
		CU_ASSERT(cuModuleUnload(this->module));
}

std::string CudaKernelGraphNode::loadCudaKernelFromFile(const std::string& kernelFilePath)
{
	std::ifstream kernelFile(kernelFilePath);
	if (kernelFile.is_open())
	{
		std::stringstream kernelFileBuffer;
		kernelFileBuffer << kernelFile.rdbuf();
		kernelFile.close();
		return kernelFileBuffer.str();
	}
	else
	{
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Cannot open file with user cuda kernel. File path: ") + kernelFilePath);
		return "";
	}
}

void CudaKernelGraphNode::checkCudaKernelCompilationLog(const nvrtcProgram& prog)
{
	size_t logSize;
	NVRTC_ASSERT(nvrtcGetProgramLogSize(prog, &logSize));
	if (logSize > 1)
	{
		std::vector<char> log(logSize);
		NVRTC_ASSERT(nvrtcGetProgramLog(prog, &log[0]));
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("User cuda kernel compilation log: ") + &log[0]);
	}
}

std::vector<char> CudaKernelGraphNode::loadPTX(const nvrtcProgram& prog)
{
	size_t ptxSize;
	NVRTC_ASSERT(nvrtcGetPTXSize(prog, &ptxSize));
	std::vector<char> ptx(ptxSize);
	NVRTC_ASSERT(nvrtcGetPTX(prog, &ptx[0]));
	return ptx;
}

void CudaKernelGraphNode::loadUserGlobalFunction(const std::string& cudaKernel)
{
	std::smatch match;
	const std::regex regex("__global__ void (\\w)+\\(");
	std::regex_search(cudaKernel, match, regex);
	std::string matchResult = match[0];
	std::string globalFunctionName = matchResult.substr(16, matchResult.size() - 17);
	CU_ASSERT(cuModuleGetFunction(&this->kernel, this->module, globalFunctionName.c_str()));
}

void CudaKernelGraphNode::compileUserCudaKernel()
{
	std::string kernelFilePath = this->getNodeVariable("kernelFile").getValue<std::string>();
	std::string cudaKernel = this->loadCudaKernelFromFile(kernelFilePath);

	nvrtcProgram prog;
	NVRTC_ASSERT(nvrtcCreateProgram(&prog, cudaKernel.c_str(), kernelFilePath.c_str(), 0, NULL, NULL));

	std::vector<VariableAnyValue> compileOptions = this->getNodeVariable("compileOptions").getValue<std::vector<VariableAnyValue>>();
	std::vector<const char*> opts;
	for (int i = 0; i < compileOptions.size(); ++i)
		opts.push_back(boost::any_cast<std::string>(compileOptions[i].getAnyValuePtr())->c_str());
	NVRTC_ASSERT(nvrtcCompileProgram(prog, (int)compileOptions.size(), &opts[0]));

	this->checkCudaKernelCompilationLog(prog);
	std::vector<char> ptx = this->loadPTX(prog);
	CU_ASSERT(cuModuleLoadDataEx(&this->module, &ptx[0], 0, 0, 0));
	NVRTC_ASSERT(nvrtcDestroyProgram(&prog));

	this->loadUserGlobalFunction(cudaKernel);

	this->isUserKernelCompiled = true;
}

std::vector<void*> CudaKernelGraphNode::getPointersToCudaKernelArguments(std::vector<VariableAnyValue>& inputArgs)
{
	std::vector<void*> args(inputArgs.size());
	
	for (int i = 0; i < inputArgs.size(); ++i)
	{
		if (inputArgs[i].getAnyValue().type() == boost::typeindex::type_id<bool>())
			args[i] = boost::any_cast<bool>(inputArgs[i].getAnyValuePtr());
		else if (inputArgs[i].getAnyValue().type() == boost::typeindex::type_id<int>())
			args[i] = boost::any_cast<int>(inputArgs[i].getAnyValuePtr());
		else if (inputArgs[i].getAnyValue().type() == boost::typeindex::type_id<float>())
			args[i] = boost::any_cast<float>(inputArgs[i].getAnyValuePtr());
	}
	return args;
}

void CudaKernelGraphNode::allocOutputData()
{
	std::vector<VariableAnyValue> outputDims = this->getNodeVariable("outputDims").getValue<std::vector<VariableAnyValue>>();
	outputDims.resize(3, VariableAnyValue(1));
	Dims dims(outputDims[0].getValue<int>(), outputDims[1].getValue<int>(), outputDims[2].getValue<int>());

	std::string outputDataType = this->getNodeVariable("outputDataType").getValue<std::string>();
	if (outputDataType.compare("short") == 0)
		this->allocGPUMemory<short>(&this->outputData, dims);
	else if (outputDataType.compare("int") == 0)
		this->allocGPUMemory<int>(&this->outputData, dims);
	else if (outputDataType.compare("float") == 0)
		this->allocGPUMemory<float>(&this->outputData, dims);
	else if (outputDataType.compare("double") == 0)
		this->allocGPUMemory<double>(&this->outputData, dims);
	else if (outputDataType.compare("float2") == 0)
		this->allocGPUMemory<float2>(&this->outputData, dims);
	else
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Unsupported data type for user's cuda kernel output data pointer."));
}

CUdeviceptr CudaKernelGraphNode::getInputDataPtr()
{
	std::string inputDataType = this->getNodeVariable("inputDataType").getValue<std::string>();
	if (inputDataType.compare("short") == 0)
		return (CUdeviceptr)this->inputData.getPtr<short*>();
	else if (inputDataType.compare("int") == 0)
		return (CUdeviceptr)this->inputData.getPtr<int*>();
	else if (inputDataType.compare("float") == 0)
		return (CUdeviceptr)this->inputData.getPtr<float*>();
	else if (inputDataType.compare("double") == 0)
		return (CUdeviceptr)this->inputData.getPtr<double*>();
	else if (inputDataType.compare("float2") == 0)
		return (CUdeviceptr)this->inputData.getPtr<float2*>();
	else
		DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Unsupported data type for user's cuda kernel input data pointer."));

	return (CUdeviceptr)nullptr;
}

void CudaKernelGraphNode::process(cudaStream_t& defaultStream)
{
	if (!this->isUserKernelCompiled)
		this->compileUserCudaKernel();

	this->allocOutputData();

	CUdeviceptr input = this->getInputDataPtr();
	CUdeviceptr output = (CUdeviceptr)this->outputData.getVoidPtr();

	std::vector<void*> args;
	args.push_back(&input);
	args.push_back(&output);
	std::vector<VariableAnyValue> inputArgs = this->getNodeVariable("args").getValue<std::vector<VariableAnyValue>>();
	std::vector<void*> params = this->getPointersToCudaKernelArguments(inputArgs);
	args.insert(args.end(), params.begin(), params.end());
	int sharedMemBytes = this->getNodeVariable("sharedMemBytes").getValue<int>();

	std::vector<VariableAnyValue> grid = this->getNodeVariable("grid").getValue<std::vector<VariableAnyValue>>();
	grid.resize(3, VariableAnyValue(1));

	std::vector<VariableAnyValue> block = this->getNodeVariable("block").getValue<std::vector<VariableAnyValue>>();
	block.resize(3, VariableAnyValue(1));

	CU_ASSERT(cuLaunchKernel(this->kernel, grid[0].getValue<int>(), grid[1].getValue<int>(), grid[2].getValue<int>(), 
							 block[0].getValue<int>(), block[1].getValue<int>(), block[2].getValue<int>(), sharedMemBytes, defaultStream, &args[0], 0));
	CUDA_ASSERT(cudaGetLastError());

	this->outputData.copyExtraData(this->inputData);
}
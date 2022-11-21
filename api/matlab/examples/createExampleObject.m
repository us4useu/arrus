addpath("/home/pjarosik/src/arrus/build/api/matlab/wrappers");
addpath("/home/pjarosik/src/arrus/api/matlab/");


modelId = arrus.devices.probe.ProbeModelId("manufacturer" ,"test", "name", "testprobe");
probeModel = arrus.devices.probe.ProbeModel("modelId", modelId, "nElements", 192, "pitch", 0.2e-3, "txFrequencyRange", [0 15e6], "voltageRange", [0 90], "curvatureRadius", 0);

arrus_mex_object_wrapper("__global", "createExampleObject", probeModel);


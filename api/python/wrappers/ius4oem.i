%include stdint.i
%include exception.i
%include windows.i

%exception {
	try {
		$action
	} catch(const std::exception &e) {
	    SWIG_exception(SWIG_RuntimeError, e.what());
	} catch(...) {
	    std::cout << "Unhandled type of exception! Check logs for more information" << std::endl;
	}
}

%module(directors="1") ius4oem

%include <std_shared_ptr.i>
%shared_ptr(IUs4OEM)

%include "carrays.i"
%array_functions(unsigned short, uint16Array);
%array_functions(double, doubleArray);


%ignore us4oem::Us4OEMException;
%ignore us4oem::afe58jd18::Register195;
%ignore us4oem::afe58jd18::Register196;
%ignore us4oem::afe58jd18::Register203;
%ignore us4oem::afe58jd18::REGISTER_ADDRESS;
// TODO(pjarosik) should not be part of the ius4oem interface!
%ignore AttachCVSeries;
%{
#include <iostream>
#include <afe58jd18Registers.h>
#include <ius4oem.h>
#include "iI2CMaster.h"
#include <thread>
#include <chrono>
#include <bitset>
#include <vector>
#include "core/api/DataAcquiredEvent.h"

static constexpr size_t NCH = IUs4OEM::NCH;
%}
%include afe58jd18Registers.h
%include ius4oem.h
%include iI2CMaster.h
%include "core/callbacks/Event.h"
%include "core/callbacks/DataAcquiredEvent.h"

%feature("director") ScheduleReceiveCallback;

%inline %{

// TODO (pjarosik) move this callback to some other place
class ScheduleReceiveCallback {
public:
    virtual void run(const arrus::DataAcquiredEvent& event) const = 0;
    virtual ~ScheduleReceiveCallback() {};
};

void ScheduleReceiveWithCallback(IUs4OEM* that, const size_t address,
                                 const size_t length,
                                 const size_t start,
                                 const size_t decimation,
                                 ScheduleReceiveCallback& callback) {
    auto fn = [&callback, address, length] () {
		// TODO(pjarosik) consider creating event outside the lambda function, to reduce interrupt handling time.
		const arrus::DataAcquiredEvent& event = arrus::DataAcquiredEvent(address, length);

		PyGILState_STATE gstate = PyGILState_Ensure();
		try {
			callback.run(event);
		} catch(const std::exception &e) {
			std::cerr << e.what() << std::endl;
		}
		PyGILState_Release(gstate);
    };
    that->ScheduleReceive(address, length, start, decimation, fn);
}

void ScheduleReceiveWithoutCallback(IUs4OEM* that, const size_t address,
                                    const size_t length,
                                    const size_t start,
                                    const size_t decimation) {
    that->ScheduleReceive(address, length, start, decimation);
}

// TODO(pjarosik) fix below in more elegant way
void TransferRXBufferToHostLocation(IUs4OEM* that, unsigned long long dstAddress, size_t length, size_t srcAddress) {
    that->TransferRXBufferToHost((unsigned char*) dstAddress, length, srcAddress+0x1'0000'0000);
}

II2CMaster* castToII2CMaster(IUs4OEM* ptr) {
    return dynamic_cast<II2CMaster*>(ptr);
}

std::shared_ptr<IUs4OEM> getUs4OEMPtr(unsigned idx) {
    return std::shared_ptr<IUs4OEM>(GetUs4OEM(idx));
}

void EnableReceiveDelayed(IUs4OEM* ptr) {
    ptr->EnableReceive();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void setTxApertureCustom(IUs4OEM* that, const unsigned short* enabled, const size_t length, const unsigned short firing) {
    std::bitset<NCH> aperture;
    for(int i=0; i < length; ++i) {
        if(enabled[i]) {
            aperture.set(i);
        }
    }
    that->SetTxAperture(aperture, firing);
}

void setActiveChannelGroupCustom(
    IUs4OEM* that,
    const unsigned short* enabled,
    const size_t length,
    const unsigned short firing)
{
    std::bitset<NCH/8> mask;
    for(int i=0; i < length; ++i) {
        if(enabled[i]) {
            mask.set(i);
        }
    }
    that->SetActiveChannelGroup(mask, firing);
}

void setRxApertureCustom(IUs4OEM* that, const unsigned short* enabled, const size_t length, const unsigned short firing)
{
    std::bitset<NCH> aperture;
    for(int i=0; i < length; ++i) {
        if(enabled[i]) {
            aperture.set(i);
        }
    }
    that->SetRxAperture(aperture, firing);
}

void setTGCSamplesCustom(IUs4OEM* that, const double* input,
    const size_t length, const unsigned short firing)
{
    std::vector<float> samples(length);
    for(int i=0; i < length; ++i) {
        samples[i] = input[i];
    }
    that->TGCSetSamples(samples, firing);
}
%}

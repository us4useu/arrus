%include stdint.i
%include exception.i
%include windows.i

%exception {
	try {
		$action
	} catch(const std::exception &e) {
	    SWIG_exception(SWIG_RuntimeError, e.what());
	}
}

%module(directors="1") iarius

%include <std_shared_ptr.i>
%shared_ptr(IArius)

%include "carrays.i"
%array_functions(unsigned short, uint16Array);

%ignore arius::AriusException;
%ignore arius::afe58jd18::Register195;
%ignore arius::afe58jd18::Register196;
%ignore arius::afe58jd18::Register203;
%ignore arius::afe58jd18::REGISTER_ADDRESS;
// TODO(pjarosik) should not be part of the iarius interface!
%ignore AttachCVSeries;
%{
#include <iostream>
#include <afe58jd18Registers.h>
#include <iarius.h>
#include "iI2CMaster.h"
#include <thread>
#include <chrono>
#include "core/api/DataAcquiredEvent.h"
#include <typeinfo>

#include <bitset>
static constexpr size_t NCH = IArius::NCH;
%}
%include afe58jd18Registers.h
%include iarius.h
%include iI2CMaster.h
%include "core/api/Event.h"
%include "core/api/DataAcquiredEvent.h"

%feature("director") ScheduleReceiveCallback;

%inline %{
// TODO (pjarosik) move this callback to some other place
class ScheduleReceiveCallback {
public:
    virtual void run(const arius::DataAcquiredEvent& event) const = 0;
    virtual ~ScheduleReceiveCallback() {};
};

// TODO(pjarosik) unify below functions

void ScheduleReceiveWithCallback(IArius* that, const size_t address, const size_t length,
                                 ScheduleReceiveCallback& callback) {
	std::cout << address << length << std::endl;
    auto fn = [&callback, address, length] () {
		// TODO(pjarosik) consider creating event outside the lambda function, to reduce interrupt handling time.
		
		const arius::DataAcquiredEvent& event = arius::DataAcquiredEvent(address, length);
		
		PyGILState_STATE gstate = PyGILState_Ensure();
		try {	
			callback.run(event);
		} catch(const std::exception &e) {
			// TODO(pjarosik) use logger here
			std::cerr << e.what() << std::endl;
		}
		PyGILState_Release(gstate);
    };
    that->ScheduleReceive(address, length, fn);
}

void ScheduleReceiveWithoutCallback(IArius* that, const size_t address, const size_t length) {
    that->ScheduleReceive(address, length);
}

// TODO(pjarosik) fix below in more elegant way
void TransferRXBufferToHostLocation(IArius* that, unsigned long long dstAddress, size_t length, size_t srcAddress) {
    that->TransferRXBufferToHost((unsigned char*) dstAddress, length, srcAddress+0x1'0000'0000);
}

II2CMaster* castToII2CMaster(IArius* ptr) {
    return dynamic_cast<II2CMaster*>(ptr);
}

std::shared_ptr<IArius> getAriusPtr(unsigned idx) {
    return std::shared_ptr<IArius>(GetArius(idx));
}

void EnableReceiveDelayed(IArius* ptr) {
    ptr->EnableReceive();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void setTxApertureCustom(IArius* that, const unsigned short* enabled, const size_t length, const unsigned short firing) {
    std::bitset<NCH> aperture;
    for(int i=0; i < length; ++i) {
        if(enabled[i]) {
            aperture.set(i);
        }
    }
    that->SetTxAperture(aperture, firing);
}

void setRxApertureCustom(IArius* that, const unsigned short* enabled, const size_t length, const unsigned short firing) {
    std::bitset<NCH> aperture;
    for(int i=0; i < length; ++i) {
        if(enabled[i]) {
            aperture.set(i);
        }
    }
    that->SetRxAperture(aperture, firing);
}
%}

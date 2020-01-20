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

%module iarius
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
%}
%include afe58jd18Registers.h
%include iarius.h
%include iI2CMaster.h

%inline %{
// TODO(pjarosik) fix below in more elegant way
void TransferRXBufferToHostLocation(IArius* that, unsigned long long dstAddress, size_t length, size_t srcAddress) {
    that->TransferRXBufferToHost((unsigned char*) dstAddress, length, srcAddress+0x1'0000'0000);
}

II2CMaster* castToII2CMaster(IArius* ptr) {
    return dynamic_cast<II2CMaster*>(ptr);
}
%}

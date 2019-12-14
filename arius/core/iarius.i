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
%}
%include afe58jd18Registers.h
%include iarius.h

%inline %{
// TODO(pjarosik) fix below in more elegant way
void TransferRXBufferToHostLocation(IArius* that, unsigned long long dstAddress, size_t length, size_t srcAddress) {
    that->TransferRXBufferToHost((unsigned char*) dstAddress, length, srcAddress);
}
%}

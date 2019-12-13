%include stdint.i
%include exception.i
%include windows.i

%exception {
	try {
		$action
	} catch(const std::exception &e) {
		std::cout << "SWIG exception: "  << e.what() << std::endl;
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

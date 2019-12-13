%include "stdint.i"
%include exception.i
%include windows.i

%exception {
	try {
		$action
	} catch(const std::exception &e) {
		std::cout << "SWIG exception: "  << e.what() << std::endl;
	}
}

%module ihv256
%{
#include <iostream>
#include "ii2CMaster.h"
#include "ihv256.h"
%}

%include ii2CMaster.h
%include ihv256.h

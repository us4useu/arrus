%include "stdint.i"
%include exception.i
%include windows.i

%exception {
	try {
		$action
	} catch(const std::exception &e) {
	    SWIG_exception(SWIG_RuntimeError, e.what());
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

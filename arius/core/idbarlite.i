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
%module idbarlite
%ignore dbarlite::DBARLite::Write;
%ignore dbarlite::DBARLite::Read;
%ignore dbarlite::DBARLite::WriteAndRead;
%ignore dbarlite::DBARLiteException;
%{
#include <iostream>
#include "iI2CMaster.h"
#include "idbarLite.h"
%}
%include iI2CMaster.h
%include idbarLite.h

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
%module dbarlite
%ignore dbarlite::DBARLite::Write;
%ignore dbarlite::DBARLite::Read;
%ignore dbarlite::DBARLite::WriteAndRead;
%ignore dbarlite::DBARLiteException;
%{
#include <iostream>
#include "ii2CMaster.h"
#include "idbarLite.h"
%}
%include ii2CMaster.h
%include idbarLite.h

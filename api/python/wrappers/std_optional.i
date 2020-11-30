%typemap(in) std::optional<float> %{
    if($input == Py_None) {
        $1 = std::optional<float>();
    }
    else {
        $1 = std::optional<float>((float)PyFloat_AsDouble($input));
    }
%}

%typemap(out) std::optional<float> %{
    if($1) {
        $result = PyFloat_FromDouble(*$1);
    }
    else {
        $result = Py_None;
        Py_INCREF(Py_None);
    }
%}

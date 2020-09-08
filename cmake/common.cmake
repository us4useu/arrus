# Prepends a value to the Path env. variable and returns new value in output_var.
# This function is os independent.
function(prepend_env_path output_var_name value)
    if(WIN32)
        set(arrus_path_sep "\;")
    elseif(UNIX)
        set(arrus_path_sep ":")
    endif()
    set(${output_var_name} ${value}${arrus_path_sep}$ENV{PATH} PARENT_SCOPE)
endfunction()
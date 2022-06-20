classdef Ptr < handle
    % A simple C++ pointer wrapper.
    % This class stores a given pointer to some memory area and a type of object
    % that is stored there.
    % Proviedes also methods to access object's methods.
    %
    % NOTE: this class does not call the C++ constructor/destructor at the time
    % of constructing/destroing an instance of this class.
    % To create an object that owns some C++ object (in a RAII sense), see arrus.UniquePtr.
    % NOTE: the value of ptr may be invalid at some time, i.e. still point to an object,
    % that no longer exists. This will be true for objects that are managed by some
    % external entity.
    properties(GetAccess = protected, SetAccess = immutable, Transient = true, Hidden = true)
        % C++ class name
        className (1, 1) string
        % uint64 value which should be interpreted as a pointer to some memory location.
        ptr (1, 1) uint64
    end

    methods
        function obj = Ptr(className, ptr)
            obj.className = className;
            obj.ptr = ptr;
        end

        function outputs = callMethod(obj, methodName, nargout, varargin)
            if isempty(obj.ptr)
                error("ARRUS:IllegalState", "Objects handle is not set.");
            end
            outputs = cell(1, nargout);
            [outputs{:}] = arrus_mex_object_wrapper(obj.className, methodName, obj.ptr, varargin{:});
        end

    end
end


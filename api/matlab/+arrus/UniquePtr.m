classdef UniquePtr < arrus.Ptr
    % A simple std::unique_ptr-like wrapper.
    % Note: this class will:
    % - allow you to create backend object by providing constructor parameters,
    % - the object pointed to by the ptr value will be deleted on destruction.
    methods
        function obj = UniquePtr(className, varargin)
            ptr = arrus_mex_object_wrapper(className, "create",  varargin{:});
            obj = obj@arrus.Ptr(className, ptr);
        end

        function delete(obj)
            if ~isempty(obj.ptr)
                arrus_mex_object_wrapper(obj.className, "remove", obj.ptr);
            end
        end
    end
end


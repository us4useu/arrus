classdef (Abstract = true) UniquePtr < arrus.Ptr
    % A simple std::unique_ptr-like wrapper.
    % Note: this class will:
    % - allow you to create backend object by providing constructor parameters,
    % - the object pointed to by the ptr value will be deleted on destruction.
    methods
        function obj = UniquePtr(className, varargin)
            obj.className = className;
            obj.ptr = arrus.arrus_mex_object_wrapper(obj.className, "create",  varargin{:});
        end

        function delete(obj)
            if ~isempty(obj.handle)
                arrus.arrus_mex_object_wrapper(obj.className, "remove", obj.handle);
            end
        end
    end
end


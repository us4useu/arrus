classdef (Abstract = true) MexObject < handle
    properties(GetAccess = private, SetAccess = immutable, Transient = true, Hidden = true)
        className
        handle
    end

    methods
        function obj = MexObject(className, varargin)
            obj.className = className;
            % Verify available mex file.
            % TODO(pjarosik) check if the version of mex file is the same as for version of this toolbox
            obj.handle = mex_object_wrapper(obj.className, "create",  varargin{:});
        end

        function delete(obj)
            if ~isempty(obj.handle)
                mex_object_wrapper(obj.className, "remove", obj.handle);
            end
        end
    end

    methods(Access = protected, Sealed = true)

        function res = callMethod(obj, methodName, varargin)
            if isempty(obj.handle)
                error("ARRUS:IllegalState", "Objects handle is not set.")
            end
            res = mex_object_wrapper(obj.className, methodName, obj.handle, varargin{:});
        end

    end

end


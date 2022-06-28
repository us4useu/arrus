classdef BufferElement < handle
    % A single element of data buffer.

    properties(GetAccess = protected, SetAccess = immutable, Transient = true, Hidden = true)
        ptr arrus.Ptr {mustBeScalarOrEmpty}
    end
    methods
        function obj = BufferElement(ptr)
            obj.ptr = arrus.Ptr("arrus.framework.BufferElement", ptr);
        end
        
        function array = eval(obj)
            % Returns and releases the last element of the buffer.
            res = obj.ptr.callMethod("eval", 1);
            array = res{1, 1};
        end
    end
end
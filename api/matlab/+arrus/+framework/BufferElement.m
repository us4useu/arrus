classdef BufferElement < handle
    % A single element of data buffer.

    properties(GetAccess = protected, SetAccess = immutable, Transient = true, Hidden = true)
        ptr (1, 1) arrus.Ptr
    end
    methods
        function obj = BufferElement(ptr)
            obj.ptr = arrus.Ptr("arrus.framework.BufferElement", ptr);
        end
        
        function array = eval(obj)
            % Returns and releases the last element of the buffer.
            array = obj.ptr.callMethod("eval");
        end
    end
end
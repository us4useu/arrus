classdef Buffer < handle
    % A data buffer.
    % Data buffer can be populated by one of the processes running in the session background.
    % In particualar, an ultrasound device can populate this buffer with some new raw channel data.
    properties(GetAccess = protected, SetAccess = immutable, Transient = true, Hidden = true)
        ptr arrus.Ptr {mustBeScalarOrEmpty}
    end
    methods
        function obj = Buffer(ptr)
            obj.ptr = arrus.UniquePtr("arrus.framework.Buffer", ptr);
        end
        
        function element = front(obj)
            % Returns and releases the last element of the buffer.
            res = obj.ptr.callMethod("front", 1); % uint64 pointer to buffer element (cell array)
            element = arrus.framework.BufferElement(res{1, 1});
        end
    end
end
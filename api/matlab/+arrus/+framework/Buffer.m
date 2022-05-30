classdef Buffer < handle
    % A data buffer.
    % Data buffer can be populated by one of the processes running in the session background.
    % In particualar, an ultrasound device can populate this buffer with some new raw channel data.
    properties(GetAccess = protected, SetAccess = immutable, Transient = true, Hidden = true)
        ptr (1, 1) arrus.Ptr
    end
    methods
        function obj = Buffer(ptr)
            obj.ptr = ptr;
        end
        
        function array = back(obj)
            % Returns and releases the last element of the buffer.
            array = obj.ptr.callMethod("front");
        end
    end
end
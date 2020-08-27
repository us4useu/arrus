classdef Session < arrus.MexObject
    methods
        function obj = Session()
            obj = obj@arrus.MexObject("Session");
        end
        
        function res = getDevice(obj, deviceId)
            res = obj.callMethod("getDevice", deviceId);
        end
    end
end
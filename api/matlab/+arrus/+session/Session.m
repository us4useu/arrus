classdef Session < arrus.MexObject
    methods
        function obj = Session(sessionSettings)
            obj = obj@arrus.MexObject("Session", sessionSettings);
        end
        
        function res = getDevice(obj, deviceId)
            res = obj.callMethod("getDevice", deviceId);
        end
    end
end
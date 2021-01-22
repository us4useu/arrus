classdef DeviceId
    % System device identifier.
    %
    % :param deviceType: (string) device type, available values: 'Us4OEM',\ 
    %   'ProbeAdapter', 'Probe', 'Us4R', 'GPU', 'CPU'
    % :param ordinal: (integer) ordinal number of the device in the system
    
    properties(GetAccess = public, SetAccess = private)
        deviceType
        ordinal
    end
    
    methods(Access = public)
        function obj = DeviceId(deviceType, ordinal)
            obj.deviceType = deviceType;
            obj.ordinal = ordinal;
        end
    end
    
end
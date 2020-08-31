classdef ProbeAdapterModelId
    % Probe adapter model id.
    % 
    % :param name: name of the model
    % :param manufacturer: name of the manufacturer
    
    properties(GetAccess = public, SetAccess = private)
        name
        manufacturer
    end
    
    methods
        function obj = ProbeAdapterModelId(name, manufacturer)
            obj.name = convertCharsToStrings(name);
            obj.manufacturer = convertCharsToStrings(manufacturer);
        end
    end
end


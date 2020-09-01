classdef ProbeAdapterModelId
    % Probe adapter model id.
    % 
    % :param name: name of the model
    % :param manufacturer: name of the manufacturer
    
    properties(GetAccess = public, SetAccess = private)
        manufacturer
        name
    end
    
    methods
        function obj = ProbeAdapterModelId(manufacturer, name)
            obj.manufacturer = convertCharsToStrings(manufacturer);
            obj.name = convertCharsToStrings(name);
        end
    end
end


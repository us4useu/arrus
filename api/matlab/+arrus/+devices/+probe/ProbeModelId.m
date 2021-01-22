classdef ProbeModelId
    % Probe model id.
    % 
    % :param manufacturer: name of the manufacturer
    % :param name: name of the model

    properties(GetAccess = public, SetAccess = private)
        manufacturer
        name
    end
    
    methods
        function obj = ProbeModelId(manufacturer, name)
            obj.manufacturer = convertCharsToStrings(manufacturer);
            obj.name = convertCharsToStrings(name);
        end
    end
end


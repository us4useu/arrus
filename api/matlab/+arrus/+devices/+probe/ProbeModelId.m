classdef ProbeModelId
    % Probe model id.
    % 
    % :param name: name of the model
    % :param manufacturer: name of the manufacturer
    
    properties(GetAccess = public, SetAccess = private)
        name
        manufacturer
    end
    
    methods
        function obj = ProbeModelId(name, manufacturer)
            obj.name = name;
            obj.manufacturer = manufacturer;
        end
    end
end


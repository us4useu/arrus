classdef Us4RSettings 
    properties(GetAccess = public, SetAccess = private)
        us4OEMSettings
    end
    
    methods(Access = public)
        function obj = Us4RSettings(us4OEMSettings)
            obj.us4OEMSettings = us4OEMSettings;
        end
    end
    
end
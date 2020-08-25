classdef SessionSettings
    properties(GetAccess = public, SetAccess = private)
        us4RSettings arrus.devices.Us4RSettings
    end
    
    methods(Access = public)
        function obj = SessionSettings(us4RSettings)
            obj.us4RSettings = us4RSettings;
        end
    end
    
end
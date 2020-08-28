classdef SessionSettings
    % Session configuration object.
    %
    % :param us4RSettings: us4Settings to apply durring session
    
    properties(GetAccess = public, SetAccess = private)
        us4RSettings arrus.devices.us4r.Us4RSettings
    end
    
    methods(Access = public)
        function obj = SessionSettings(us4RSettings)
            obj.us4RSettings = us4RSettings;
        end
    end
    
end
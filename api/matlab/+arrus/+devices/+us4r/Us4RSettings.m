classdef Us4RSettings 
    % Us4R system settings.
    % 
    % Only one of the following options to configure system may be
    % provided: us4OEMSettings or a tuple (probeSettings, 
    % probeAdapterSettings, rxSettings).
    %
    % :param us4OEMSettings: settings to apply to Us4OEMs that are \ 
    %    available in a Us4R system
    % :param probeAdapterSettings: probe adapter settings to apply
    % :param probeSettings: probe settings to apply
    % :param rxSettings: data acquisition settings to apply
    
    properties(GetAccess = public, SetAccess = private)
        us4OEMSettings 
        probeAdapterSettings
        probeSettings
        rxSettings
    end
    
    methods(Access = public)
        function obj = Us4RSettings(varargin)
            % Us4Settings constructor
            % 
            % Constructor can take one parameter (us4OEMSettings) or 
            % three parameters: (probeAdapterSettings, probeSettings, 
            % rxSettings).
            if nargin == 1
                obj.us4OEMSettings = varargin{1};
            elseif nargin == 3
                obj.probeAdapterSettings = varargin{1};
                obj.probeSettings = varargin{2};
                obj.rxSettings = varargin{3};
            else
                error("ARRUS:IllegalArgument", ...
                    "Constructor should take 1 or 3 parameters.");
            end
        end
    end
    
end
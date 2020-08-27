classdef RxSettings
    % Us4R data acquisition settings.
    %
    % :param dtgcAttenuation: (scalar, optional), DTGC attenuation value, \
    %   when set to empty array, DTGC will be off [dB]
    % :param pgaGain: (scalar) Programable Gain Amplifier gain value [dB]
    % :param lnaGain: (scalar) Low-noise Amplifier gain value [dB]
    % :param tgcSamples: (vector, optional) TGC curve to apply, when set \
    %   to empty array, TGC will be off [dB]
    % :param lpfCutoff: Low-pass filter cutoff value [Hz]
    % :param active termination: active termination value
    
    properties(GetAccess = public, SetAccess = private)
        dtgcAttenuation
        pgaGain (1, 1)
        lnaGain (1, 1)
        tgcSamples
        lpfCutoff
        activeTermination
    end
    
    methods(Access = public)
        
        function obj = RxSettings(varargin)
            % Rx settings constructor.
            % 
            % Values can be provided in the order of the class properties
            % or by providing a list of 'param1Name', 'param1Value', 
            % 'param2Name', 'param2Value', ...
            mc = metaclass(obj);
            nParams = size(mc.PropertyList);
            nParams = nParams(1);
            if nargin == nParams
                for i = 1:nParams
                    obj.(mc.PropertyList(i).Name) = varargin{i};
                end
            elseif nargin == 2*nParams
                for i = 1:2:nargin
                    obj.(varargin{i}) = varargin{i+1};
                end
            else
                error("ARRUS:IllegalArgument", "Invalid number of arguments.");
            end
        end
        
    end
    
end
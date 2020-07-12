classdef Pulse
    % Class corresponding to transmit pulse.
    %
    % 
    %   properties:
    %       txFrequency - pulse transmit frequency in [Hz]
    %       txNPeriods - number of pulse periods
    %
    %   methods:
    %       TxPulse() - constructor.
    %       TxPulse() creates TxPulse object with all empty properties.
    %       To pass arguments to the constructor name-value convetion is used.
    %       Example: pulse = TxPulse('txFrequency', 5e6, 'txNPeriods', 2);
    %
    %
    
    properties
        frequency {mustBeNumeric, ...
                     mustBeFinite, ...
                     mustBeNonnegative, ...
                     mustBeReal ...
                     }
        nPeriods {mustBeInteger, ...
                    mustBeNonnegative ...
                    }
        
        % in future:
%         txWaveform
%         txWaveformFs
        
    end
    
    methods
        
        function obj = Pulse(varargin)
            if nargin ~= 0
                p = inputParser;
                
                % adding parameters to parser
                addParameter(p, 'frequency', [] ,@isscalar)
                addParameter(p, 'nPeriods',[] ,@isscalar)
                parse(p,varargin{:})
                                
                obj.frequency = p.Results.frequency;
                obj.nPeriods = p.Results.nPeriods;
            end
        end
    end
    
end

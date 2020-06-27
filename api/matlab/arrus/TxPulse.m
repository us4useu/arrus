classdef TxPulse
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
        txFrequency {mustBeNumeric, ...
                     mustBeFinite, ...
                     mustBeNonnegative, ...
                     mustBeReal ...
                     }
        txNPeriods {mustBeInteger, ...
                    mustBeNonnegative ...
                    }
        
        % in future:
%         txWaveform
%         txWaveformFs
        
    end
    
    methods
        
        function obj = TxPulse(varargin)
            if nargin ~= 0
                p = inputParser;
                
                % adding parameters to parser
                addParameter(p, 'txFrequency', [] ,@isscalar)
                addParameter(p, 'txNPeriods',[] ,@isscalar)
                parse(p,varargin{:})
                                
                obj.txFrequency = p.Results.txFrequency;
                obj.txNPeriods = p.Results.txNPeriods;
            end
        end
    end
    
end

classdef Tgc
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
        tgcStart {mustBeInteger, ...
                     mustBeNonnegative, ...
                     mustBeReal ...
                     }
        tgcSlope {mustBeInteger ...
                    }
        
        
    end
    
    methods
        
        function obj = Tgc(varargin)
            if nargin ~= 0
                p = inputParser;
                
                % validation functions
                tgcStartVld = @(x) isscalar(x) ...
                                && x > 0 ...
                                && x == floor(x) ...
                                && isfinite(x) ...
                                ;
                tgcSlopeVld = @(x) isscalar(x) ...
                                && isfinite(x) ...
                                ;
                
                
                % adding parameters to parser
                addParameter(p, 'tgcStart', [], tgcStartVld)
                addParameter(p, 'tgcSlope', [], tgcSlopeVld)
                parse(p,varargin{:})
                                
                obj.tgcStart = p.Results.tgcStart;
                obj.tgcSlope = p.Results.tgcSlope;
            end
        end
    end
    
end

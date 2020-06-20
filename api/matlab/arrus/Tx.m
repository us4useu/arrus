classdef Tx
    % Class corresponding to single atomic 'transmit' event.
    % 
    %   properties:
    %       txAperture - logical mask where 0 and 1 corresponds to
    %           active and inactive element respectively,
    %       txDelays - vector of transmit delays in [s], 
    %       txNPeriods - number of transmit periods,
    %       txFrequency - transmit frequency [Hz].
    %
    %   methods:
    %       Tx() - constructor.
    %       Tx() creates Rx object with all empty properties.
    %       To pass arguments to the constructor name-value convetion is used.
    %       Example: Tx('txAperture', logical(1:128), 'txFrequency', 5e6)
    %       If txDelays is empty or not given, transmit delays are set to 0.
    %       If txNPeriods is not given, the default value 2 is set.
    %       If txAperture and txDelays are non-empty, they should have the same
    %           size.
    %       If txAperture is nonempty and non-all-zeroes, txFrequency must be
    %           given.
    %
    %
     
    properties
        txAperture
        txDelays 
        txNPeriods
        txFrequency
    end
    
%     aperture, delays, frequency, nPeriods
    methods
        
        function obj = Tx(varargin)
            if nargin ~= 0
                p = inputParser;
                
                % validation functions
                txApertureVld = @(x) islogical(x); 
                txFrequencyVld = @(x) isscalar(x) && isreal(x) && (x > 0);
                txDelaysVld =  @(x) isreal(x);
                txNPeriodsVld = @(x) isscalar(x) && isreal(x) && (x >= 0);
                
                % adding parameters to parser
                addParameter(p, 'txAperture',[], txApertureVld)
                addParameter(p, 'txFrequency',[], txFrequencyVld)
                addParameter(p, 'txDelays',[], txDelaysVld)
                addParameter(p, 'txNPeriods',2, txNPeriodsVld)
                parse(p,varargin{:})

                % additional input validation
                if ~isempty(p.Results.txAperture) && ...
                    any(p.Results.txAperture) && ...
                    isempty(p.Results.txFrequency)
                    
                    error('If txAperture is nonempty, the txFrequency must be given.')
                end
                
                if ~isempty(p.Results.txAperture) && ...
                   ~isempty(p.Results.txDelays) && ...
                   ~isequal(size(p.Results.txAperture), size(p.Results.txDelays))
                    
                    error('If txAperture and txDelays are non-empty, they must have the same size.')
                end
                                
                obj.txAperture = p.Results.txAperture;
                obj.txDelays = p.Results.txDelays;
                obj.txFrequency = p.Results.txFrequency;
                obj.txNPeriods = p.Results.txNPeriods;
            end
        end
    end
    
end

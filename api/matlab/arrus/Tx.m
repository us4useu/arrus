classdef Tx
    % Class corresponding to single atomic 'transmit' event.
    % 
    %   properties:
    %       txAperture - logical mask where 0 and 1 corresponds to
    %           active and inactive element respectively,
    %       txDelays - vector of transmit delays in [s], 
    %       txPulse - txPulse object
    %
    %   methods:
    %       Tx() - constructor.
    %           Tx() creates Tx object with all empty properties.
    %           To pass arguments to the constructor name-value convetion 
    %               is used.
    %           Example: 
    %               tx = Tx('txAperture', logical(1:128), 'txFrequency', 5e6)
    %
    %           If txDelays is empty or not given, transmit delays are set to 0.
    %           If txAperture and txDelays are non-empty, they should have the same
    %           size.
    %           If txAperture is nonempty and non-all-zeroes, txFrequency must be
    %           given.
    %
    %
     
    properties
        txAperture
        txDelays 
        txPulse 
    end
    
    
    methods
        
        function obj = Tx(varargin)
            if nargin ~= 0
                p = inputParser;
                
                % validation functions
                txApertureVld = @(x) islogical(x); 
                txDelaysVld =  @(x) isreal(x);                
                txPulseVld = @(x) isa(x,'TxPulse') || isempty(x);
                
                % adding parameters to parser
                addParameter(p, 'txAperture',[], txApertureVld)
                addParameter(p, 'txDelays',[], txDelaysVld)
                
                addParameter(p, 'txPulse', [], txPulseVld)
                parse(p,varargin{:})

                
                if ~isempty(p.Results.txAperture) && ...
                   ~isempty(p.Results.txDelays) && ...
                   ~isequal(size(p.Results.txAperture), size(p.Results.txDelays))
                    
                    error('If txAperture and txDelays are non-empty, they must have the same size.')
                end
                                
                obj.txAperture = p.Results.txAperture;
                obj.txDelays = p.Results.txDelays;
                obj.txPulse = p.Results.txPulse;
            end
        end
    end
    
end

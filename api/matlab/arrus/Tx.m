classdef Tx
    % Class corresponding to single atomic 'transmit' event.
    % 
    %   properties:
    %       aperture - logical mask where 0 and 1 corresponds to
    %           active and inactive element respectively,
    %       delay - vector of transmit delays in [s], 
    %       pulse - txPulse object
    %
    %   methods:
    %       Tx() - constructor.
    %           Tx() creates Tx object with all empty properties.
    %           To pass arguments to the constructor name-value convetion 
    %               is used.
    %           Example: 
    %               tx = Tx('aerture', logical(1:128))
    %
    %           If delay is empty or not given, transmit delays are set to 0.
    %           If aperture and delay are non-empty, they should have the same
    %           size.
    %           If aperture is nonempty and non-all-zeroes, pulse.frequency must be
    %           given.
    %
    %
     
    properties
        aperture
        delay 
        pulse 
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
                addParameter(p, 'aperture',[], txApertureVld)
                addParameter(p, 'delay',[], txDelaysVld)
                
                addParameter(p, 'pulse', [], txPulseVld)
                parse(p,varargin{:})

                
                if ~isempty(p.Results.aperture) && ...
                   ~isempty(p.Results.delay) && ...
                   ~isequal(size(p.Results.aperture), size(p.Results.delay))
                    
                    error('If txAperture and txDelays are non-empty, they must have the same size.')
                end
                                
                obj.aperture = p.Results.aperture;
                obj.delay = p.Results.delay;
                obj.pulse = p.Results.pulse;
            end
        end
    end
    
end

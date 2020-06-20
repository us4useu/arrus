classdef Rx
    % Class corresponding to single atomic 'receive' event.
    %
    % 
    %   properties:
    %       rxAperture - logical mask where 0 and 1 corresponds to 
    %          active and inactive element respectively,
    %       rxDelay - receive delay in [s],
    %       rxTime - time gate length [s], i.e. how much time the acqusition 
    %           will take.
    %
    %   methods:
    %       Rx() - constructor.
    %       Rx() creates Rx object with all empty properties.
    %       To pass arguments to the constructor name-value convetion is used.
    %       Example: Rx('rxAperture', logical(1:128))
    %
    %
    
    properties
        rxAperture
        rxDelay % rxDelay - value of receive delays in [s].
        rxTime
    end
    
    methods
        
        function obj = Rx(varargin)
            if nargin ~= 0
                p = inputParser;
                
                % validation functions
                rxApertureVld = @(x) islogical(x); 
                rxDelayVld =  @(x) isreal(x);                
                rxTimeVld = @(x) isreal(x) && (x>=0);
                
                % adding parameters to parser
                addParameter(p, 'rxAperture',[], rxApertureVld)
                addParameter(p, 'rxDelays',[], rxDelayVld)
                addParameter(p, 'rxTime',[], rxTimeVld)                
                parse(p,varargin{:})
                                
                obj.rxAperture = p.Results.rxAperture;
                obj.rxDelay = p.Results.rxDelay;
                obj.rxTime = p.Results.rxTime;
            end
        end
    end
    
end

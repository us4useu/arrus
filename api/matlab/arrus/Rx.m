classdef Rx
    % Class corresponding to single atomic 'receive' event.
    %
    % 
    %   properties:
    %       rxAperture - logical mask where 0 and 1 corresponds to 
    %          active and inactive element respectively,
    %       rxDelay - receive delay in [s],
    %       rxTime - time gate length [s], i.e. how much time the acqusition 
    %           will take,
    %       fsDivider - sampling frequency divider, default value 1,
    %       tgc - tgc object for time gain control.
    %
    %   methods:
    %       Rx() - constructor.
    %           Rx() creates Rx object with all empty properties.
    %           To pass arguments to the constructor name-value convetion is used.
    %           Example: Rx('rxAperture', logical(1:128))
    %
    %
    
    properties
        rxAperture
        rxDelay % rxDelay - value of receive delays in [s].
        rxTime
        fsDivider
        tgc
    end
    
    methods
        
        function obj = Rx(varargin)
            if nargin ~= 0
                p = inputParser;
                
                % validation functions
                rxApertureVld = @(x) islogical(x); 
                rxDelayVld =  @(x) isreal(x);                
                rxTimeVld = @(x) isreal(x)  ...
                              && x >= 0 ...
                              ;
                fsDividerVld = @(x) isfinite(x) ...
                                && x == floor(x) ...
                                && isnumeric(x) ...
                                && isscalar(x) ...
                                ;
                
                tgcVld = @(x) isa(x,'Tgc');
                
                % adding parameters to parser
                addParameter(p, 'rxAperture', [], rxApertureVld)
                addParameter(p, 'rxDelay', [], rxDelayVld)
                addParameter(p, 'rxTime', [], rxTimeVld)                
                addParameter(p, 'fsDivider', 1, fsDividerVld)                
                addParameter(p, 'tgc', [], tgcVld)                
                
                parse(p,varargin{:})
                                
                obj.rxAperture = p.Results.rxAperture;
                obj.rxDelay = p.Results.rxDelay;
                obj.rxTime = p.Results.rxTime;
                obj.fsDivider = p.Results.fsDivider;
                obj.tgc = p.Results.tgc;
            end
        end
    end
    
end

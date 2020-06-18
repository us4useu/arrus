classdef Rx
    % Class corresponding to single atomic 'receive' event
    % 
    % The properties of the object are:
    % rxAperture - logical mask where 0 and 1 corresponds to active and
    %              inactive element respectively.
    % rxDelay - receive delay in [s],
    % rxTime - time gate length, i.e. how much time the acqusition will
    %          get.
    %
    % Rx() creates Tx object with all empty properties.
    % To pass arguments to the constructor name-value convetion is used.
    
    properties
        rxAperture
        rxDelay % rxDelay - value of receive delays in [s].
        rxTime
%         rxNSamples        
    % rxNSamples - set number of samples to acquire. If two element vector, the
    %              first value is delay in [samples], and the second is number
    %              of samples + delay in samples
    end
    
%     aperture, delays, frequency, nPeriods
    methods
        
        function obj = Rx(varargin)
            if nargin ~= 0
                p = inputParser;
                
                % validation functions
                rxApertureVld = @(x) islogical(x); 
                rxDelayVld =  @(x) isreal(x);                
                rxTimeVld = @(x) isreal(x) && (x>=0);
%                 rxNSamplesVld = @(x) isfinite(x) && x == floor(x);                
                
                % adding parameters to parser
                addParameter(p, 'rxAperture',[], rxApertureVld)
                addParameter(p, 'rxDelays',[], rxDelayVld)
                addParameter(p, 'rxTime',[], rxTimeVld)                
%                 addParameter(p, 'rxNSamples',[], rxNSamplesVld)                
                parse(p,varargin{:})
                                
                obj.rxAperture = p.Results.rxAperture;
                obj.rxDelay = p.Results.rxDelay;
                obj.rxTime = p.Results.rxTime;
            end
        end
    end
    
end

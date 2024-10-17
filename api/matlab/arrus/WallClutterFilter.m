classdef WallClutterFilter < handle
    % A Wall Clutter Filter class definition.
    %
    % Intended use: filtration of Doppler data to remove 
    % the low frequency, high amplitude clutter.
    
    properties(Access = private)
        
        coeff
        state
        init
        
    end
    
    methods
        
        function obj = WallClutterFilter(b,a,inputGridSize,initMode)
            % Creates WallClutterFilter object
            %
            % :param b: filter numerator coefficients
            % :param a: filter denominator coefficients
            % :param inputGridSize: size [z,x] of the data to be filtered 
            % :param initMode: determines filter initialization (step or zero)
            % :returns: WallClutterFilter object
            
            obj.coeff.b = b;
            obj.coeff.a = a;
            
            filtOrd = max(numel(b),numel(a)) - 1;
            obj.state = zeros([inputGridSize filtOrd],'like',single(1i));
            
            switch initMode
                case 'zero'
                    obj.init = zeros(filtOrd,1);
                case 'step'
                    [~,obj.init] = filter(b,a,ones(1000,1));
                otherwise
                    error('Invalid initMode value, should be "zero" or "step"');
            end
            obj.init = reshape(obj.init,1,1,filtOrd);
            
            obj.coeff.b = gpuArray(single(obj.coeff.b));
            obj.coeff.a = gpuArray(single(obj.coeff.a));
            obj.state   = gpuArray(       obj.state );
            obj.init    = gpuArray(single(obj.init));
        end
        
        function y = filter(obj,x,initEnable,nRejFrames)
            % Performs filtration
            %
            % :param x: data to be filtered
            % :param initEnable: enables filter initialization (reset of the filter internal state)
            % :param nRejFrames: number of output frames to be rejected (if initEnable is true)
            % :returns: filtered data and updated WallClutterFilter object
            
            % Default input values
            if nargin<3 || isempty(initEnable)
                initEnable = false;
            end
            
            if nargin<4 || isempty(nRejFrames)
                nRejFrames = 0;
            end
            
            % Initialization
            if initEnable
                obj.state = x(:,:,1) .* obj.init;
            end
            
            % Filtration
            [y, obj.state] = wcFilter(x, obj.coeff.b, obj.coeff.a, obj.state);
            
            % Rejection of initial frames 
            if initEnable && nRejFrames > 0
                y(:,:,1:nRejFrames) = [];
            end
        end
        
    end
end


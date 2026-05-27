classdef NeuroTimer < handle
    
    properties(Access = private)
        tLim
        timer
    end
    
    methods 
        
        function obj = NeuroTimer(tLim)
            obj.tLim = tLim;
            obj.timer = tic;
        end
        
        function reset(obj)
            obj.timer = tic;
        end

        function state = isContinue(obj)
            t = toc(obj.timer);
            state = t < obj.tLim;
            disp(t);
        end
        
        function doNothing(obj, data)
            % do nothing
        end
        
    end
end
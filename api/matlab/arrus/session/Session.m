classdef Session < MexObject
    methods
        function obj = Session()
            obj = obj@MexObject("Session");
        end
        
        function res = test1(obj)
            res = obj.callMethod("test1");
        end
        
        function [s, a] = test2(obj)
            res = obj.callMethod("test2");
            s = res{1};
            a = res{2};
        end
    end
end
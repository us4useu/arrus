classdef Session < arrus.MexObject
    methods
        function obj = Session()
            obj = obj@arrus.MexObject("Session");
        end
        
        function res = test1(obj)
            res = obj.callMethod("test1");
        end
        
    end
end
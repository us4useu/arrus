classdef RxSettings
    
    properties(GetAccess = public, SetAccess = private)
        dtgcAttenuation
        pgaGain (1, 1)
        lnaGain (1, 1)
        
        % Optional
        tgcSamples
        
    end
    
    methods(Access = public)
        
        function obj = RxSettings() 
        end
        
    end
    
end
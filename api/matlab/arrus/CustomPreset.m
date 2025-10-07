classdef CustomPreset
    % A class that stores parameters of system preset.
    
    properties
        txDelay = []
    end
    
    methods

        function obj = CustomPreset(varargin)
            % Creates a CustomPreset object.
            % 
            % Syntax:
            % obj = CustomPreset(name, value, ..., name, value)
            % 
            % All inputs are organized in name-value pairs.
            % 
            % :param txDelay: Tx delays [s]. Numerical array (nElem x nTx).
            
            if mod(nargin, 2) == 1
                error("ARRUS:params", ...
                      "Input should be a list of  'key', value params.");
            end
            for i = 1:2:nargin
                obj.(varargin{i}) = varargin{i+1};
            end
        end

    end
end



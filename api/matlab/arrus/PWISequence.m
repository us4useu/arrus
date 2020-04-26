classdef PWISequence < SimpleTxRxSequence
    % Plane Wave Imaging Tx/Rx Sequence.
    %
    % The PWI sequence has set txFocus to inf by defualt.
    
    methods
        function obj = PWISequence(varargin)
            obj = obj@SimpleTxRxSequence(varargin{:});
            [~, y] = size(obj.txAngle);
            obj.txFocus = inf(1, y);
        end
        
        % TODO(piotrkarwat) modify setter and getter for txFocus.
        % It should be disabled to set values different than inf.
    end
end


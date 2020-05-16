classdef (Abstract) Us4RSystem < handle
    % A handle to the Us4R system. 
    %
    % This class provides functions to configure the system and perform
    % data acquisition using the Us4R.
    
    
    methods
        % The body of the below methods is intentionally left empty,
        % due to an issue with matlabdomain doc generation. 
        % Check issue: https://github.com/us4useu/matlabdomain/issues/1 
        
        function upload(obj, sequenceOperation, reconstructOperation)
            % Uploads operations to the us4R system.
            %
            % Currently, only supports :class:`SimpleTxRxSequence`
            % and :class:`Reconstruction` implementations.
            %
            % :param sequenceOperation: TX/RX sequence to perform on the us4R system
            % :param reconstructOperation: reconstruction to perform with the collected data
            % :returns: updated Us4R object
        end
        
        function [rf,img] = run(obj)
            % Runs uploaded operations in the us4R system.
            %
            % Currently, only supports :class:`SimpleTxRxSequence` and :class:`Reconstruction`
            % implementations.
            %
            % :returns: RF frame and reconstructed image (if :class:`Reconstruction` operation was uploaded)
            
            rf = [];
            img = [];
        end
        
        function runLoop(obj, isContinue, callback)
            % Runs the uploaded operations in a loop.
            % 
            % Currently, only supports :class:`SimpleTxRxSequence` and \
            % :class:`Reconstruction` implementations.
            %
            % :param isContinue: should the system continue executing \
            %   the op? Takes no parameters and returns a boolean value.
            % :param callback: a function to call after executing the \
            %   operation. Should take one parameter - the output of the \
            %   executed operation.
        end 
        
    end
end


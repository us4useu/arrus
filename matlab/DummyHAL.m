classdef DummyHAL < HAL
    properties
        data
        frameIdx
        isStarted
        isConfigured
    end
    methods
        function obj = DummyHAL(data)
            if nargin == 0
                obj.data = rand(32, 512, 32, 3);
            else 
                if ~ismember(ndims(data), [3, 4])
                    error("Unsupported input data shape, should be 3-D or 4-D.")
                end
                obj.data = double(data);
            end     
            obj.frameIdx = 0;
            obj.isConfigured = false;
            obj.isStarted = false;
            end 
        
        function configure(obj, json)
            if obj.isStarted
                error("Device cannot be configured now. Call 'stop' first.");
            end
            obj.isConfigured = true;
            disp("Loaded configuration file.");
        end

        function start(obj)
            if ~obj.isConfigured
                error("Device is not configured. Call 'configure' first.");
            end
            obj.isStarted = true;
            disp("Device started.");
        end

        function stop(obj)
            obj.assertIsStarted();
            obj.isStarted = false;
            disp("Device stopped.");
        end

        function sync(obj)
            obj.assertIsStarted();
            % Increment current frame index.
            sz = size(obj.data);
            if ndims(obj.data) == 4
                batchDim = sz(4);
            else
                batchDim = 1;
            end
            obj.frameIdx = mod(obj.frameIdx+1, batchDim); 
        end

        function halOutput = getData(obj)
            obj.assertIsStarted();
            % Return current frame.
            halOutput = obj.data(:, :, :, obj.frameIdx+1);
        end
    end
    methods (Access = private)
        function assertIsStarted(obj)
            if ~obj.isStarted
                error("Device is not started. Call 'start' first.");
            end
        end
    end
end
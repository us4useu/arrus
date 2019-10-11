classdef (Abstract) HAL < handle
    properties

    end

    methods (Abstract)
        configure(obj, json)
        % Apply TX/RX configuration stored in given JSON string.

        start(obj)
        % Start the device.

        stop(obj)
        % Stop the device.

        sync(obj)
        % Sync. for the next acquisition.

        halOutput = getData(obj)
        % Returns current data buffer. 
        % Output data type: double. 
        % Output dimensions: ECS (Event, Sample, Channel).
    end
end

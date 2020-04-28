classdef Reconstruction < Operation
    % A reconstruction operation to perform on a device.
    %
    % :param txCenterElement: an array of tx aperture center elements
    % :param txAperturecenter: an array of tx aperture center positions [m]
    % :param txApertureSize: the size of the Tx aperture [element]
    
    properties
        filterEnable
        filterACoeff
        filterBCoeff
        filterDelay
        iqEnable
        cicOrder
        decimation
        xGrid
        zGrid
    end
    
    methods
        function obj = Reconstruction(varargin)
            % assign property to the object
            if mod(nargin, 2) == 1
                throw( ...
                    MException( ...
                        "Arrus:params", ...
                        "Input should be a list of  'key', value params."))
            end
            for i = 1:2:nargin
                obj.(varargin{i}) = varargin{i+1};
            end
        end
    end
end


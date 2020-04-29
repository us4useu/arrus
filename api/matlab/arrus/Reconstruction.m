classdef Reconstruction < Operation
    % A reconstruction operation to perform on a device.
    %
    % :param filterEnable: boolean, enable filtering the input signal
    % :param filterACoeff: 1-D filter denominator coefficient
    % :param filterBCoeff: 1-D filter numerator coefficient
    % :param filterDelay:
    % :param iqEnable: boolean, enable iq signal reconstruction instead of raw RF
    % :param cicOrder:
    % :param decimation: decimation factor
    % :param xGrid: (1, width) vector, x-coordinates of the image pixels [m]
    % :param zGrid: (1, depth) vector z-coordinates of the image pixels [m]

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


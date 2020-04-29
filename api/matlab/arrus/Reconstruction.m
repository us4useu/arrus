classdef Reconstruction < Operation
    % A reconstruction operation to perform on a device.
    %
    % :param filterEnable: boolean, enable filtering the input signal
    % :param filterACoeff: 1-D filter denominator coefficient
    % :param filterBCoeff: 1-D filter numerator coefficient
    % :param filterDelay: delay introduced by the filter [samples] (not implemented yet)
    % :param iqEnable: boolean, enable iq signal reconstruction instead of raw RF
    % :param cicOrder: order of the Cascaded-Integrator-Comb anti-aliasing filter
    % :param decimation: decimation factor
    % :param xGrid: (1, width) vector, x-coordinates of the image pixels [m]
    % :param zGrid: (1, depth) vector z-coordinates of the image pixels [m]

    properties
        filterEnable
        filterACoeff
        filterBCoeff
        filterDelay = 0
        iqEnable
        cicOrder
        decimation
        xGrid
        zGrid
    end
    
    methods
        function obj = Reconstruction(varargin)
            if mod(nargin, 2) == 1
                error("Arrus:params", ...
                      "Input should be a list of  'key', value params.");
            end
            for i = 1:2:nargin
                obj.(varargin{i}) = varargin{i+1};
            end
        end
    end
end


classdef Reconstruction < Operation
    % A reconstruction operation to perform in the system.
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
    % :param bmodeEnable: boolean, enable B-Mode reconstruction
    % :param colorEnable: boolean, enable Color Doppler reconstruction and duplex imaging
    % :param vectorEnable: boolean, enable Vector Doppler reconstruction and duplex imaging
    % :param bmodeFrames: selects frames to be used in B-Mode reconstruction
    % :param colorFrames: selects frames to be used in Color Doppler reconstruction
    % :param vector0Frames: selects frames to be used in Vector Doppler reconstruction as 1st projection
    % :param vector1Frames: selects frames to be used in Vector Doppler reconstruction as 2nd projection
    % :param bmodeRxTangLim: rx tangent limits for B-Mode
    % :param colorRxTangLim: rx tangent limits for Color Doppler
    % :param vector0RxTangLim: rx tangent limits for Vector Doppler (1st projection)
    % :param vector1RxTangLim: rx tangent limits for Vector Doppler (2nd projection)
    % :param wcFilterACoeff: 1-D filter denominator coefficient (Wall Clutter Filter for Color/Vector Doppler)
    % :param wcFilterBCoeff: 1-D filter numerator coefficient (Wall Clutter Filter for Color/Vector Doppler)
    % :param wcFiltInitSize: number of initial filter output samples to be rejected
    
    properties
        filterEnable = false
        filterACoeff
        filterBCoeff
        filterDelay = 0
        iqEnable = true
        cicOrder
        decimation
        xGrid
        zGrid
        rxApod = [1 1]
        bmodeEnable = true
        colorEnable = false
        vectorEnable = false
        bmodeFrames
        colorFrames
        vector0Frames
        vector1Frames
        bmodeRxTangLim = [-0.5 0.5]
        colorRxTangLim
        vector0RxTangLim
        vector1RxTangLim
        wcFilterACoeff
        wcFilterBCoeff
        wcFiltInitSize = 0
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


classdef Reconstruction
    % A class that stores parameters of the reconstruction \
    % operation to be performed in the system.

    properties
        gridModeEnable = true
        filterEnable = false
        filterACoeff
        filterBCoeff
        filterDelay = 0
        swDdcEnable
        decimation {mustBeFinite, mustBeInteger, mustBePositive}
        xGrid
        zGrid
        sos
        rxApod = [1 1]
        bmodeEnable = true
        colorEnable = false
        vectorEnable = false
        bmodeFrames
        colorFrames
        vector0Frames
        vector1Frames
        bmodeRxTangLim = [-0.5 0.5]
        colorRxTangLim = [-0.5 0.5]
        vector0RxTangLim = [-0.5 0.5]
        vector1RxTangLim = [-0.5 0.5]
        wcFilterACoeff
        wcFilterBCoeff = 1
        wcFiltInitSize = 0
        cohFiltEnable = false
        cohCompEnable = false
    end
    
    methods
        function obj = Reconstruction(varargin)
            % Creates a Reconstruction object.
            % 
            % Syntax:
            % obj = Reconstruction(name, value, ..., name, value)
            % 
            % All inputs are organized in name-value pairs.
            % 
            % :param gridModeEnable: If set to true, enables grid-based \
            %   reconstruction. If set to false, reconstruction is done \
            %   classically, in a line-by-line manner. Logical scalar. \
            %   Optional name-value argument, default = true.
            % :param filterEnable: Enables filtration of the raw data. \
            %   Logical scalar. Optional name-value argument, default = false.
            % :param filterACoeff: Denominator coefficients of the raw data \
            %   filter. Numerical vector. Optional name-value argument, default = [].
            % :param filterBCoeff: Numerator coefficients of the raw data \
            %   filter. Numerical vector. Optional name-value argument, default = [].
            % :param filterDelay: Delay introduced by the filter [samples].\
            %   Numerical scalar. Optional name-value argument, default = 0. Not yet implemented.
            % :param swDdcEnable: Enables software DDC (Digital Down
            %   Convertion). Logical scalar. Optional name-value argument.
            % :param decimation: Software decimation factor. Numerical scalar. \
            %   Optional name-value argument.
            % :param xGrid: Coordinate grid x [m]. Numerical vector. \
            %   Optional name-value argument, default = [].
            % :param zGrid: Coordinate grid z [m]. Numerical vector. \
            %   Optional name-value argument, default = [].
            % :param sos: Speed of sound value used for reconstruction [m/s]. \
            %   Numerical scalar. Optional name-value argument.
            % :param rxApod: Rx apodization window. Numerical vector. \
            %   Optional name-value argument, default = [1 1].
            % :param bmodeEnable: Enables B-Mode reconstruction. Logical \
            %   scalar. Optional name-value argument, default = true.
            % :param colorEnable: Enables Color Doppler reconstruction and \
            %   Duplex imaging. Logical scalar. Optional name-value argument, \
            %   default = false.
            % :param vectorEnable: Enables Vector Doppler reconstruction and \
            %   Duplex imaging. Logical scalar. Optional name-value argument, \
            %   default = false.
            % :param bmodeFrames: Frame numbers to be used in B-Mode \
            %   reconstruction. Numerical vector. Optional name-value \
            %   argument.
            % :param colorFrames: Frame numbers to be used in Color Doppler \
            %   reconstruction. Numerical vector. Optional name-value \
            %   argument.
            % :param vector0Frames: Frame numbers to be used in Vector Doppler \
            %   reconstruction as 1st projection. Numerical vector. Optional \
            %   name-value argument.
            % :param vector1Frames: Frame numbers to be used in Vector Doppler \
            %   reconstruction as 2nd projection.  Numerical vector. Optional \
            %   name-value argument.
            % :param bmodeRxTangLim: Rx tangent limits for B-Mode. \
            %   Numerical array (K*, 2). Optional name-value argument, \
            %   default = [-0.5 0.5].
            % :param colorRxTangLim: Rx tangent limits for Color Doppler. \
            %   Numerical array (M*, 2). Optional name-value argument, \
            %   default = [-0.5 0.5].
            % :param vector0RxTangLim: Rx tangent limits for Vector Doppler \
            %   (1st projection). Numerical array (N*, 2). Optional name-value \
            %   argument, default = [-0.5 0.5].
            % :param vector1RxTangLim: Rx tangent limits for Vector Doppler \
            %   (2nd projection). Numerical array (N*, 2). Optional name-value \
            %   argument, default = [-0.5 0.5].
            % :param wcFilterACoeff: Denominator coefficients of the \
            %   Wall Clutter Filter (WCF) for Color/Vector Doppler. \
            %   Numerical vector. Optional name-value argument.
            % :param wcFilterBCoeff: Numerator coefficients of the \
            %   Wall Clutter Filter (WCF) for Color/Vector Doppler. \
            %   Numerical vector. Optional name-value argument.
            % :param wcFiltInitSize: Number of initial WCF output samples \
            %   to be rejected due to the filter initialization. \
            %   Numerical scalar. Optional name-value argument.
            % :param cohFiltEnable: Enables coherence-weighted filtration. \
            %   Logical scalar. Optional name-value argument, default = false.
            % :param cohCompEnable: Enables coherent compounding.  If set \
            %   to false, the compounding is performed on envelope images \
            %   (incoherently). Logical scalar. Optional name-value argument, \
            %   default = false.
            % 
            % * K, M, N - length of bmodeFrames, colorFrames, and vector0Frames or vector1Frames, respectively.
            % 
            % :return: Reconstruction object.

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


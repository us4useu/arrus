classdef DigitalDownConversion
    %
    % Hardware Digital Down Conversion parameters.
    %
    % :param demodulationFrequency: demodulation frequency [Hz]
    % :param decimation factor: decimation factor, can be fractional
    % :param firCoefficients: coefficients of the FIR filter to apply, only the upper half has to be provided
    %     (filter must be symmetric).
    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {"demodulationFrequency", "decimationFactor", "firCoefficients"};
    end
    properties
        demodulationFrequency (1, 1) {mustBeFinite, mustBeReal, mustBePositive} = 1e6
        decimationFactor (1, 1) {mustBeFinite, mustBeReal, mustBePositive} = 1
        firCoefficients (1, :) {mustBeFinite, mustBeReal}
    end
    methods
        function obj = DigitalDownConversion(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end
end

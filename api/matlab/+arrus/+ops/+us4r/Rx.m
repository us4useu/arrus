classdef Rx
    % A data reception operation.
    %
    % Example usage:
    % .. code:: matlab
    %
    %   aperture = false(128);
    %   aperture(2) = 1;
    %   aperture(4) = 1;
    %   Rx('aperture', aperture, 'sampleRange', [0 4096], 'decimationFactor', 2);
    %
    % :param aperture: logical mask where 0 and 1 corresponds to active and inactive element respectively. Required.
    % :param sampleRange: two-element vector determining sample range to acqure [first, last] (closed interval).
    %   NOTE: zero-based numbering applies here. Required.
    % :param decimationFactor: subsampling factor, starts from 1. One means no subsampling, 2 - skip every
    %   2nd sample, 3 - skip every 3rd sample, etc. Optional, 1 is default.
    properties(Constant, Hidden=true)
        REQUIRED_PARAMS = {'aperture', 'sampleRange'};
    end

    properties
        aperture (1, :) {arrus.validators.mustBeLogical}
        sampleRange (1, 2) {arrus.validators.mustBeAllNonnegativeInteger}
        decimationFactor (1, 1) {arrus.validators.mustBeAllPositiveInteger} = 1
    end

    methods
        function obj = Rx(varargin)
            obj = arrus.utils.setArgs(obj, varargin, obj.REQUIRED_PARAMS);
        end
    end

end

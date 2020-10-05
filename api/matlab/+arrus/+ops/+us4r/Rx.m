classdef Rx
    % A data reception op.
    %
    % Example usage:
    % .. code:: matlab
    %
    %   aperture = false(128);
    %   aperture(2) = 1;
    %   aperture(4) = 1;
    %   Rx('aperture', aperture, 'sampleRange', [0 4095], 'decimationFactor', 2);
    %
    % :param aperture: logical mask where 0 and 1 corresponds to active and inactive element respectively
    % :param sampleRange: two-element vector determining sample range to acqure [first, last] (closed interval).
    %   NOTE: zero-based numbering applies here.
    % :param decimationFactor: subsampling factor, starts from 1. One means no subsampling, 2 - skip every
    %   2nd sample, 3 - skip every 3rd sample, etc. Optional, 1 is default.
    properties
        aperture
        sampleRange
        decimationFactor
    end
    
    methods
        function obj = Rx(varargin)
            p = inputParser;
            isAllInteger =  @(x) isnumeric(x) && all(x == floor(x));
            isAllPositiveInteger = @(x) isAllInteger(x) && all(x > 0);
            isAllNonnegativeInteger = @(x) isAllInteger(x) && all(x >= 0);
            addRequired(p, 'aperture', @(x) isvector(x) && ~isempty(x) && islogical(x));
            addRequired(p, 'sampleRange', @(x) isvector(x) && length(x) == 2 && isAllNonnegativeInteger(x));
            addOptional(p, 'decimationFactor', 1, @(x) isscalar(x) && isAllPositiveInteger(x));

            parse(p,varargin{:});
            obj.aperture = p.Results.aperture;
            obj.sampleRange = p.Results.sampleRange;
            obj.decimationFactor = p.Results.decimationFactor;
        end
    end
    
end

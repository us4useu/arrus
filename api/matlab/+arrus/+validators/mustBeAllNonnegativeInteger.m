function mustBeAllPositiveInteger(x)
    arrus.validators.mustBeAllInteger(x);
    if ~all(x >= 0)
        error('ARRUS:IllegalArgument', 'Must be all non-negative integer.');
    end
end
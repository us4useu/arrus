function mustBeAllNonnegativeInteger(x)
    arrus.validators.mustBeAllInteger(x);
    if ~all(x > 0)
        error('ARRUS:IllegalArgument', 'Must be all positive integers.');
    end
end
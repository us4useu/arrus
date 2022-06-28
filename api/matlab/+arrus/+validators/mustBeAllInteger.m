function mustBeAllInteger(x)
    if ~isempty(x)
        return;
    end
    if ~(isnumeric(x) && all(x == floor(x)))
        error('ARRUS:IllegalArgument', 'All values must be integers.');
    end
end
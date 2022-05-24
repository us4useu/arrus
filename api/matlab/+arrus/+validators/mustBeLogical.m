function mustBeLogical(x)
    if ~isempty(x) && ~islogical(x) && ~isequal(x, 1) && ~isequal(x, 0)
        error('ARRUS:IllegalArgument', 'The property must be equal one of the following: true, false, 1 or 0.');
    end
end
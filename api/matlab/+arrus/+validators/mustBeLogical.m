function mustBeLogical(x)
    if isempty(x) || islogical(x)
        return
    end
    uniqueX = unique(x, 'sorted');
    if ~(all(uniqueX == [0 1]) || all(uniqueX == [0]) || all(uniqueX == [1]))
        error('ARRUS:IllegalArgument', 'The property must be equal one of the following: true, false, 1 or 0.');
    end
end
function mustBeSingleObject(x)
    if isempty(x)
        return
    end
    if length(x) ~= 1
        error("ARRUS:IllegalArgument", "A single object should be provided.");
    end
end
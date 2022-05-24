function obj = setArgs(obj, params, requiredParams)
    % Sets a list of key values to the given object.
    % The expected format of the cell array params is {key1, value1, key2, value2, ...}
    % The requiredParams contains names of all parameters that are required by this method.
    % if any of the requiredParams is not in the params list, an ARRUS::IllegalArgument
    % error will be raised.
    %
    % :param obj: object to update
    % :param params: params to set, a cell list of key-value pairs
    % :param requiredParams: a cell list of names of the required parameters.
    %
    if isempty(params)
        error("ARRUS:IllegalArgument", "The list of parameters should not be empty.");
    end
    [~, n] = size(params);
    if mod(n, 2) == 1
        error("ARRUS:IllegalArgument", "Input should be a list of  'key', value params.");
    end
    params = convertStringsToChars(params);
    requiredParams = convertStringsToChars(requiredParams);
    disp(size(params));
    disp(requiredParams);
    keys = params(1:2:end);
    keys = convertStringsToChars(keys);
    disp(keys);
    values = params(2:2:end);
    % NOTE: the below probably is not the most optimal way to check, if the required params
    % are in the provided list.
    for i = 1:length(requiredParams)
        requiredKey = requiredParams{i};
        if(~any(strcmp(keys, requiredKey)))
            error("ARRUS:IllegalArgument", strcat("The parameter ", requiredKey, " is missing."));
        end
    end
    for i = 1:length(keys)
        try
            obj.(keys{i}) = values{i};
        catch ERR
            switch ERR.identifier
                case 'ARRUS:IllegalArgument'
                    error(ERR.identifier, strcat("Error while setting property ", keys{i}, ": ", getReport(ERR)));
                otherwise
                    rethrow(ERR)
            end
        end
    end
end



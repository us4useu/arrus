function addLogFile(filepath, level)
    % Starts logging to the specified file.
    %
    % :param filepath: path to log file
    % :param level: log severity to set, available values: 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE'.
    arrus.arrus_mex_object_wrapper("__global", "addLogFile", convertCharsToStrings(filepath), ...
                                   convertCharsToStrings(level));
end


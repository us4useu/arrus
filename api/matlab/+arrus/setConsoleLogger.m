function setConsoleLogger(logSeverity)
% Sets console log severity.
% 
% NOTE: This should be the first function from arrus package to call if 
% you want to change console log severity.
% 
% :param logSevierty: log severity to set, available values: 'FATAL', \
%   'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE'.   
    arrus.arrus_mex_object_wrapper("__global", "setConsoleLogger", ...
        convertCharsToStrings(logSeverity));
end


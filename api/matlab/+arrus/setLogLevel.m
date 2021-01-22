function setLogLevel(level)
% Sets console logging level.
% 
% NOTE: This should be the first function from arrus package to call if 
% you want to change console log severity.
% 
% :param level: log severity to set, available values: 'FATAL', \
%   'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE'.   
    arrus.arrus_mex_object_wrapper("__global", "setLogLevel", ...
        convertCharsToStrings(level));
end


function initialize(varargin)
    % Initializes arrus package.
    % This function should be called as the first of all functions available in ARRUS.
    %
    % During initialization it is possible to specify:
    % - console log (clog) level,
    % - log file output path and level.
    %
    % Note: When the path to the log file is provided, the log level param is required.
    % Note: the log file level will be set only when the log file output path is provided.
    %
    % Available log levels: 'TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL'.
    %
    % :param clogLevel: console log level to set on initialization
    % :param logFilePath: path to the log output file
    % :param logFileLevel: log file level to set on initialization

    paramsParser = inputParser;
    addParameter(paramsParser, 'clogLevel', []);
    addParameter(paramsParser, 'logFilePath', []);
    addParameter(paramsParser, 'logFileLevel', []);
    parse(paramsParser, varargin{:});
    params = paramsParser.Results;

    arrus.setClogLevel(params.clogLevel);
    if ~isempty(params.logFilePath)
        arrus.addLogFile(params.logFilePath, params.logFileLevel);
    end
end


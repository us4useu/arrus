#pragma once

#include <string>

#define FILE_LOG 1

enum class DispatcherLogType {
    TRACE,
    DEBUG,
    INFO,
    WARNING,
    ERROR_,
    FATAL
};

class DispatcherLogger {
private:
    void configFileLog();

    void configConsoleLog();

    void configBoostLog();

    DispatcherLogger();

    DispatcherLogger(const DispatcherLogger &) = delete;

    DispatcherLogger &operator=(const DispatcherLogger &) = delete;

    static DispatcherLogger *myInstance;

public:
    static DispatcherLogger *getInstance();

    void log(const DispatcherLogType logType, const std::string &logMessage);
};

#ifdef _DEBUG
#define DISPATCHER_LOG(logType, msg) \
    DispatcherLogger::getInstance()->log(logType, std::string(msg) + std::string(" @ [file: ") + std::string(__FILE__) + std::string(", line: ") + std::to_string(__LINE__) + std::string("]"));
#else
#define DISPATCHER_LOG(logType, msg) \
    DispatcherLogger::getInstance()->log(logType, msg);
#endif
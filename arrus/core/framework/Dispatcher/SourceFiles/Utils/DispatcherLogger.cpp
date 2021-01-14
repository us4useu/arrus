#include "Utils/DispatcherLogger.h"

//C4503 decorated name length exceeded, name was truncated
//C4996 BOOST 1.57 vs 1.58: https ://github.com/boostorg/log/commit/dbff19c89c4b43ee4d581028f7256d061d685bd3
#pragma warning( disable : 4503 4996 )
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include "boost/log/utility/setup.hpp"

#pragma warning( default : 4996 )

#include <iostream>

DispatcherLogger* DispatcherLogger::myInstance = nullptr;

DispatcherLogger::DispatcherLogger()
{
	this->configBoostLog();
}

DispatcherLogger* DispatcherLogger::getInstance()
{
	if (myInstance == nullptr)
		myInstance = new DispatcherLogger();

	return myInstance;
}

void DispatcherLogger::configFileLog()
{
	boost::log::add_file_log
	(
		boost::log::keywords::auto_flush = true,
		boost::log::keywords::file_name = "dispatcherLogs_%N.log",
		boost::log::keywords::rotation_size = 1 * 1024 * 1024, // max file size - 1mb
		boost::log::keywords::max_size = 20 * 1024 * 1024, // max logs size - 20mb
		boost::log::keywords::format =
		(
			boost::log::expressions::stream
			<< boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "%d-%m-%Y %H:%M:%S")
			<< ": <" << boost::log::trivial::severity
			<< "> " << boost::log::expressions::smessage
		)
	);
}

void DispatcherLogger::configConsoleLog()
{
	boost::log::add_console_log
	(
		std::cout,
		boost::log::keywords::auto_flush = true,
		boost::log::keywords::format =
		(
			boost::log::expressions::stream
			<< boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "%d-%m-%Y %H:%M:%S")
			<< ": <" << boost::log::trivial::severity
			<< "> " << boost::log::expressions::smessage
		)
	);
}

void DispatcherLogger::configBoostLog()
{
#ifdef FILE_LOG
	this->configFileLog();
#else
	this->configConsoleLog();	
#endif

	boost::log::add_common_attributes();
}

void DispatcherLogger::log(const DispatcherLogType logType, const std::string& logMessage)
{
	switch (logType)
	{		
		case DispatcherLogType::INFO:
			BOOST_LOG_TRIVIAL(info) << logMessage;
			break;
		case DispatcherLogType::WARNING:
			BOOST_LOG_TRIVIAL(warning) << logMessage;
			break;
		case DispatcherLogType::ERROR_:
			BOOST_LOG_TRIVIAL(error) << logMessage;
			break;
		case DispatcherLogType::FATAL:
			BOOST_LOG_TRIVIAL(fatal) << logMessage;
			break;
#ifdef _DEBUG
		case DispatcherLogType::TRACE:
			BOOST_LOG_TRIVIAL(trace) << logMessage;
			break;
		case DispatcherLogType::DEBUG:
			BOOST_LOG_TRIVIAL(debug) << logMessage;
			break;
#endif
	}
}


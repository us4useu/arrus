#ifdef WIN32
#include <Windows.h>
#include <strsafe.h>
#elif linux
#include <cerrno>
#endif

#include <string>

std::string getLastErrorAsString() {
#ifdef WIN32
    LPVOID lpMsgBuf;
    DWORD dw = GetLastError();

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0, NULL);

    std::string lastErrorMessage((LPCTSTR)lpMsgBuf);

    LocalFree(lpMsgBuf);

    return lastErrorMessage;
#elif linux
    return std::string(std::strerror(errno));
#endif
}

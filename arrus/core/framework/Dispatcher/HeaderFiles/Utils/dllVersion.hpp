#ifdef WIN32
#include <Windows.h>
#endif

#include <string>

int getDllVersion(const std::string &dllName, WORD(&versionInfo)[4]) {
#ifdef WIN32
    DWORD verHandle = NULL;
DWORD verSize = GetFileVersionInfoSize(dllName.c_str(), &verHandle);
if (verSize != NULL)
{
    LPSTR verData = new char[verSize];
    if (GetFileVersionInfo(dllName.c_str(), verHandle, verSize, verData))
    {
        UINT size = 0;
        LPBYTE lpBuffer = NULL;
        if (VerQueryValue(verData, "\\", (VOID FAR* FAR*)&lpBuffer, &size))
        {
            if (size)
            {
                VS_FIXEDFILEINFO *verInfo = (VS_FIXEDFILEINFO *)lpBuffer;
                if (verInfo->dwSignature == 0xfeef04bd)
                {
                    versionInfo[0] = HIWORD(verInfo->dwFileVersionMS);
                    versionInfo[1] = LOWORD(verInfo->dwFileVersionMS);
                    versionInfo[2] = HIWORD(verInfo->dwFileVersionLS);
                    versionInfo[3] = LOWORD(verInfo->dwFileVersionLS);
                    delete[] verData;
                    return 0;
                }
            }
        }
    }
    delete[] verData;
}

return -1;
#elif linux
    versionInfo[0] = 1;
    versionInfo[1] = 0;
    versionInfo[2] = 1;
    versionInfo[3] = 0;

    return 0;
#endif

}

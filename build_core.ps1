$ARIUS_PY_PATH="C:\Python\Python37"
$ARIUS_PATH="C:\Users\pjarosik\src\Arius-software"
$ARIUS_BIN_PATH="$ARIUS_PATH\x64\Release"

del arius\core\iarius_wrap.cxx -ErrorAction SilentlyContinue
swig -c++ -python -I"$ARIUS_PATH\Arius\include" -I"$ARIUS_PATH\include" -outdir arius\python\devices\ arius\core\iarius.i
cl.exe /I"$ARIUS_PY_PATH\include" /I"$ARIUS_PATH\include" /I"$ARIUS_PATH\Arius\include" /Od /LD /EHsc arius\core\iarius_wrap.cxx "$ARIUS_PY_PATH\libs\*.lib" "$ARIUS_BIN_PATH\Arius.lib" /oarius\python\devices\_iarius.pyd
del arius\core\dbarlite_wrap.cxx -ErrorAction SilentlyContinue
swig -v -c++ -python -I"$ARIUS_PATH\DBARLite\include" -I"$ARIUS_PATH\include" -outdir arius\python\devices\ arius\core\idbarlite.i
cl.exe /I"$ARIUS_PY_PATH\include" /I"$ARIUS_PATH\include" /I"$ARIUS_PATH\DBARLite\include" /Od /LD /EHsc arius\core\idbarlite_wrap.cxx "$ARIUS_PY_PATH\libs\*.lib" "$ARIUS_BIN_PATH\DBARLite.lib" /oarius\python\devices\_idbarlite.pyd
del arius\core\hv256_wrap.cxx -ErrorAction SilentlyContinue
swig -c++ -python -I"$ARIUS_PATH\HV256\include" -I"$ARIUS_PATH\include" -outdir arius\python\devices\ arius\core\ihv256.i
cl.exe /I"$ARIUS_PY_PATH\include" /I"$ARIUS_PATH\include" /I"$ARIUS_PATH\HV256\include" /Od /LD /EHsc "$ARIUS_BIN_PATH\HV256.lib" arius\core\ihv256_wrap.cxx "$ARIUS_PY_PATH\libs\*.lib" /oarius\python\devices\_ihv256.pyd

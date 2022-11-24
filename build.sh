python scripts/cfg_build.py --targets py --source_dir=$(pwd) --us4r_dir=/home/damian/libs/develop --options ARRUS_EMBED_DEPS=ON CMAKE_BUILD_TYPE=Debug
python scripts/build.py --source_dir=$(pwd) --us4r_dir=/home/damian/libs/develop --j 8

# Install script for directory: /home/pjarosik/src/us4useu/arrus/api/python

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xpython_whlx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE DIRECTORY FILES "/home/pjarosik/src/us4useu/arrus/cmake-build-debug/api/python/dist/" FILES_MATCHING REGEX "/arrus\\-[^/]*\\.whl$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/examples" TYPE FILE FILES
    "/home/pjarosik/src/us4useu/arrus/api/python/examples/plane_wave_imaging.py"
    "/home/pjarosik/src/us4useu/arrus/api/python/examples/classical_beamforming.py"
    "/home/pjarosik/src/us4useu/arrus/api/python/examples/diverging_beams.py"
    "/home/pjarosik/src/us4useu/arrus/api/python/examples/custom_tx_rx_sequence.py"
    "/home/pjarosik/src/us4useu/arrus/api/python/examples/custom_callback.py"
    "/home/pjarosik/src/us4useu/arrus/api/python/examples/phased_array_scanning.py"
    "/home/pjarosik/src/us4useu/arrus/api/python/examples/check_probe.py"
    "/home/pjarosik/src/us4useu/arrus/api/python/examples/requirements.txt"
    )
endif()


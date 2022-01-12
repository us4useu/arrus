Release notes
=============

0.6.x
-----


0.6.6
.....

- core (C++):
    - implemented Us4R::checkState and Us4OEM::checkState methods to verify if the us4OEM module is still available (currently by checking us4OEM module firmware version)
    - implemented Us4OEM::getFirmwareVersion and Us4OEM::getTxFirmwareVersion() to get us4OEM device firmware version
    - implemented Us4OEM::getFPGATemperature() to get the temperature measured by Us4OEM's FPGA


0.6.5
.....

- core (C++ API):

    - fixed memory leak on subsequent re-uploads
    - some improvements in the us4R-lite driver compatibility with the us4R-lite system


0.6.4
.....

- Python API:

    - Fixed linear scanning for tx apertures starting from channels > 0.
    - Added phased array scanning example.  

0.6.3
.....

- Python API: 

    - Added phased array scanning.
    - Added definition for the probe adapter atl/philips-us4r4.
    - Improved IQ raw to LRI CUDA kernel performance.
    - Increased the maximum allowable voltage for Esaote probes to 90 V.

0.6.2
`````

- core (C++ API): virtual destructor for LoggerFactory.

0.6.1
`````

- core (C++ API): fixed exception type thrown at us4OEM initialization phase.

0.6.0
`````

- core (C++ API):

    - Introducing C++ API initial version.
    - Added support for ALS, APEX, ultrasonix probes and adapters.
    - Made it compatible wit Linux x64 and arm64 (tested on NVIDIA Xavier AGX, Jetpack 4.4.).
    - Added reprogramming mode PARALLEL.
    - Exposed DTGC, LPF, ActiveTermination, PGA, LNA setters in C++ interface.
    - Support for esaote2-us4r6 and esoate2-us4r6 adapters.

- Python API:

    - Added extent and colorbar to arrus.utils.gui.Display2D.
    - PWI and diverging beams for convex and linear probes in Python.
    - Now it's possible to set subapertures for Pwi and Sta schemes - check Python examples.
    - Made it possible to display multiple layers of data using Display2D.
    - Changed arrus.utils.imaging.Operation interface: renamed _initialize to initialize, _prepare to prepare, _process to process.
    - arrus.utils.gui.Display2D: now metadata parameter must be set explicitly, see Python example scripts.
    - Now it's possible to nest arrus.utils.imaging.Pipeline into another arrus.utils.imaging.Pipeline (then the buffer.get returns a tuple of arrays, session.upload returns a tuple of metadata)
    - Now session.upload returns a buffer that stores the output of the uploaded Pipeline.

- Matlab API:

    - Doppler extension and examples.
    - Fixed memory issue with digitalDownConv.
    - Created possibility to acquire data tin Us4MEX internal buffer, fixed RF batch acquisition in MATLAB.
    - Verified RX subapertures for Convex PWI.
    - Support for esaote2-us4r6 and esaote2-us4r8 adapters.


0.5.x
-----

0.5.13
``````

- Python API:

    - Added `__version__` attribute in ARRUS Python main module.
    - Fixed Decimation step output shape.

0.5.12
``````

- Python API:

    - Moved scanline location to the center of aperture, when it has even elements.
    - Implemented custom FIR filter.

0.5.11
``````

- core: now us4r-api 0.5.4 is used.

0.5.10
``````

- core and Python API: Added function that allows to close session in run-time.

0.5.9
`````

- Python API: Added an option to start recording after burst factor and TX delay center time.

0.5.8
`````

- core:

    - Added ATL/philips L7-4 probe to dictionary file.
    - Set rx time lower limit to 80 us (equal to minimum PRI).

0.5.7
`````

- Python API:

    - Implemented CIC FIR GPU kernel for Python DDC.


0.5.6
`````

- Python API:

    - Reimplemented us4r remap to be performed on GPU.

0.5.5
`````

- core:

    - Now us4r-api 0.5.3 is used.

- Python API:

    - Made the scan conversion interpolator an op field.
    - Changed the required numpy version.
    - Fixed scan conversion for startSample > 0.

0.5.4
`````

- Python API: Fixed GPU pipeline initialization warning.

0.5.3
`````

- Python API: Made tx/rx center elements to accept a list of values.

0.5.2
`````

- Python API: minor fixes in classical beamforming procedure.

0.5.1
`````

- core and Python API: Implemented streaming feature handling for us4rlite device.

0.5.0
`````

- Implemented core data model for us4oem devices: probe, adapter, us4r system, channel mapping and so on.
- Restructured Python interface for new core data model, implemented classical beamforming reconstruction pipeline.

0.4.x
-----

0.4.6
`````

- MATLAB API:

    - Added parameter ``channelsMask`` to structure ``probe``. The parameter disables selected channels from Tx/Rx.
    - The list of Us4R class constructor parameters has been changed from list of positional \
      parameters to a list of pairs (name, value). E.g. use \
      ``Us4R('nArius', 2, 'voltage', 20, 'probeName', 'SL1543', 'adapterType', 'esaote3')``.
    - Added ``dynamicRange`` parameter to ``BModeDisplay`` class.
    - Temporarily, the maximum number of tx/rxs is limited to 2048.

0.4.5
`````

- MATLAB API:

    - Support for Esaote SP2430 has been added. The new probe is identified by ``SP2430``.

0.4.4
`````

- MATLAB API:

    - The Us4R constructor now requires the ``adapterType`` parameter to be \
      explicitly provided. Please check :ref:`arrus-Us4R` documentation.
    - Support for the new Esaote probes adapter has been added. The new Esaote \
      adapter is identified by the name ``'esaote2'``. The new adapter design \
      allows for wider rx aperture: 64 elements in a single tx/rx. \
      However, due to the adapter design, it may occur that a single rx channel \
      is connected to two probe elements in the rx aperture. \
      In this case, the second of the elements is switched off in the receive. \
      In the worst scenario, this workaround switches off 4 of 64 rx elements. \
      The size of the acquired rf data is unaffected.

0.4.3
`````

- Added signed drivers for Windows, changed driver name from
  ``ARIUS`` to ``us4OEM``.
  Check :ref:`arrus-installation` procedure for information on how to
  update drivers.

0.4.2
`````

- MATLAB API:

    - ``rxApertureCenter``, ``rxCenterElement``, and ``rxApertureSize`` parameters added \
      to control the rx aperture. Use them the same way as parameters for tx aperture control. \
      The rx aperture is now fully defined by those parameters and sequence type no longer affects it.
    - rx delay is set to zero to allow for shallow region imaging.
    - ``nRepetitions``: set it to "max" to acquire the maximum number of repetitions allowable.

0.4.1
`````

- MATLAB API:

    - ``nRepetitions`` parameter added to control the number of rf-data frames to be recorded.
    - ``rxDepthRange`` parameter added to control the starting & ending depth of recorded data.
    - ``rxNSamples`` parameter, if it is a 2-element vector, allows for the setting of the starting & ending \
      sample number for the recorded data. The 1-element option is still valid.

- Python API:

    - Created new API with a model of ``operations`` executed on available \
      devices.
    - Added ``fs_divider`` parameter to ``arrus.ops.Rx`` operation, which allows \
      to reduce the sampling frequency of the module.
    - Added asynchronous communication with the device using the ``arrus.ops.Loop`` \
      operation.
    - Added ``arrus.Us4OEMCfg`` parameter that turns on data transfer loggging \
      time.

0.4.0
`````

- MATLAB API:

    - Created a new version of the Matlab API (with STA and PWI sequences, for Esaote and Ultrasonix probes).
    - Added a simplified TGC control through tgcStart and tgcSlope parameters.
    - Added a classical linear scanning example (check :class:`arrus.LINSequence`).
    - From now on, txPri takes values in seconds (previously in [us]).

- Python API:

    - Simplified ``Us4OEM.set_tgc_samples`` - it now takes values from range \
      range [0, 1], where 0 means maximum gain, 1 means minimum gain
    - Added ``Us4OEM.set_active_channel_group``, which allows to choose which \
      groups of channels can be active.

0.3.x
-----

0.3.0
`````

- Added a function to set an rx/tx aperture mask - see Us4OEM.SetTxAperture(aperture) and Us4OEM.SetRxAperture(aperture)
- Renamed the Arius project to ARRUS. Renamed ``Arius`` module to ``Us4OEM``.

0.2.x
-----

0.2.0
`````
- Added functions to trigger TX pulse asynchronously (Us4OEM.{TriggerStart, TriggerStop}).
- Added functions to enable/disable TGC and set TGC samples.
- Function Us4OEM.SetTxPeriods is no longer available; use Us4OEM.SetTxHalfPeriods instead.

0.1.x
-----

0.1.1
`````
- Fixed some of the important bugs and errors in python API.

0.1.0
`````
Initial release of arrus.

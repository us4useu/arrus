Release notes
=============

0.8.x
-----

0.8.0
.....

- core (C++):

    - us4R: exposed hardware Digital Down Conversion,
    - us4R: scheme stopping procedure was improved (should take less time)
    - us4R: exposed a function that allows to read us4OEM FPGA Wallclock
    - us4R: enabled hardware high-pass filter, by default us4OEMs will be initialized with 300 kHz cutoff frequency
    - us4R: exposed the possibility to change hardware high-pass filter (HPF) cutoff frequency or to disable HPF
    - us4R: exposed methods to read us4OEM serial and revision number (currently mocked up)

- Python API:

    - Exposed Us4OEM FPGA, UCD, UCD external temperature.
    - Implemented new module `arrus.utils.probe_check` for probe checking and automatic channel health, see example: examples/check_probe.py
    - implemented following features for probe checking: amplitude, signal duration, normalized energy and Pearson correlation coefficient (for comparing acquired signal with previous acquired 'footprint' of a probe)
    - exposed hardware Digital Down Conversion
    - us4R: exposed the possibility to change hardware high-pass filter cutoff frequency
    - arrus.utils.imaging: changed the default filter type for BandpassFilter to hamming windows (firwin)
    - Implemented a general delay and sum look up table beamformer for 3D output volume reconstruction
    - From now on TGC curve gain values will be clipped to the [min hardware gain, max hardware gain] range by default. Change arrus.ops.tgc.LinearTGC.clip to False to restore the previous behavior.
    - Changed stop_scheme behavior to close processing runner (fixes the issue with closing arrus.utils processing on session close).
    - Exposed the possibility to change LNA/PGA gain and DTGC attenuation.


- MATLAB API:

    - Exposed ARRUS core API to MATLAB interface.

0.7.x
-----

0.7.1
.....

- core (C++):

    - us4R: Limited the range of available voltage to [5, 90] V.

0.7.0
.....

- core (C++):

    - Now it is possible to set how many times the TxRxSequence should be repeated: check TxRxSequence's nRepeats parameter. By default nRepeats is set to 1.
    - Because now it is possible to acquire batch of RF frame sequences (the nRepeats parameter was added), there were some breaking changes in the implementation of the FrameChannelMapping class. The getLogical method now returns three values: us4oem number, frame number and channel number. For each us4OEM module each frame number is counted from 0 (previously the each physical frame had consecutive numbers from 0 to n, where n is the total number of physical frames acquired by all us4OEM modules). To get the number of frames preceding a given us4OEM frame, use getFirstFrame method.
    - Now it is possible to turn on test AFE patterns, see Us4R::setTestPattern.
    - Now it is possible to remap the order of us4OEM modules numbering in the system configuration, see Us4RSettings, adapterToUs4RModuleNumber. By deafult identity mapping is used. 
    - Now it is possible to specify number of us4OEM modules the system is using, check Us4RSettings, nUs4OEMs parameter.
    - Implemented MANUAL scheme work mode, i.e. it is possible to trigger TX/RX sequence programatically, see Session::run method.
    - Added Us4R::{is,set}StopOnOverflow method, which gives the possibility to continue data acquisition even if the system has detected buffer overflow. By default this property is set to true, i.e. the system will stop on buffer overflow.
    - Added possibility to measure HV P and M voltage (see Us4R::{getVoltage, getMeasuredPVoltage, getMeasuredMVoltage). Supported only by the system with us4us HV hardware (e.g. HV256 or Us4RPSC). 

- Python API:

    - added n_repeats parameter to the simple TX/RX sequences available in ARRUS, the parameter allows to specify the number of times the sequence should be repeated ("batch size"),
    - arrus.utils.imaging reconstruction operators now requires that the input data are in the format (batch size, n TX, n RX, n samples),
    - arrus.utils.imaging.RxBeamformingImg is no longer available. Please use ReconstructLri + some form of compouding (e.g. Mean(axis=1)),

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

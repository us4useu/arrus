Release notes
=============

0.10.x
-----


0.10.5

- core (C++, HAL):

    - Fixed exception handling for exceptions that occurred before/on OEM+ initialization and might cause host PC restart #US4R-572.
    - AFE58JD18 PGA LPF setting bugfix #US4R-584.
    - Initial HVPS calibration time adjustment #US4R-585.
    - Do not verify voltages measured by OEM+ with the external HV #US4R-586.


0.10.4

- core (C++, HAL):

    - Added support for DBARLitePcie fw version 1.2.0.x #US4R-579.
    - Provided possibility to get measured HV voltages on us4OEM+ #US4R-561.


0.10.3

- core (C++):

    - Exposed HVPS measurements and MANUAL_OP work mode #US4R-395.

        - Implemented MANUAL_OP work mode #US4R-395.
        - Exposed the Us4R::setMaximumPulseLength(pulseLength) function, which allows you to set what is the maximum allowable pulse length. The function was prepared in order to provide the possibility to transmit with more than 32 cycles (available only for OEM+), and that is necessary for the new probe_check function to get reliable results. By default (when the method is not explicitly called), the maximum pulse length is set to 32 cycles.

- Python API:

    - Reduced memory overhead of the data structures stroed in the Python Pipeline #ARRUS-351.
    - Implement HVPS-based probe check and MANUAL_OP work mode #US4R-395.

        - Exposed HVPS measurement in the Python API
        - Implemented MANUAL_OP work mode
        - Updated arrus.utils.probe_check module to use the HVPS measurements (OEM+ rev1 only). Exposed the new parameters: signal_type (rf or hvps), current (hvps only).
        - Exposed the us4r.set_maximum_pulse_length(pulse_length) function. See the core (C++) release notes for more details.


0.10.2

- core:

  - Fixed the subsequence selection in case of the start_scheme/stop_scheme sequence of calls #US4R-505
  - us4R/us4R-lite systems with the external HV: added OEM+ bypass mode, enabled suport for DBAR rev3, fixed HV256 initialization procedure #US4R-535, #US4R-538.


0.10.1

- core (C++):

    - Fixed buffer overflow handling in case setSubsequence method was called #US4R-474.


0.10.0

- core (C++):

    - Implemented the possibility to select TX/RX subsequence #ARRUS-280.
    - Fixed SYNC mode fixed the SYNC mode mechanism to work in a expected manner; fixed user-defined uffer overflow interrupts handling, #US4R-456.
    - Removed HV enable on HVPS initialization #US4R-455.


- Python API:

    - Implemented the possibility to select TX/RX subsequence #ARRUS-280.
    - Added set_stop_on_overflow method #US4R-456.


0.9.x
-----

0.9.3

- core (C++):

    - updated us4r-api to 0.10.3:

        - Added support for DBARLite PCIe rev2 (firmware: 1.1.0).

0.9.2

- core (C++):

    - updated us4r-api to 0.10.2:

        - Fixed FPGA decimation setting when hardware DDC not used #US4R-398.

0.9.1

- core (C++):

    - updated us4r-api to 0.10.1:

        - Fixed AFE test patterns generation (RAMP, etc.) #US4R-381.
        - Added HVP1/M1 voltages calibration skip when not used #US4R-376.       

0.9.0

- core (C++): 

    - us4R: exposed SYNC mode #ARRUS-83.
    - Fixed RX aperture < 32 elements #ARRUS-206
    - Added support for us4OEM+ hardware with and without external HVPS. #MD_us4OEM-12, #M_OEM-10, #M_OEM52, #ARRUS-272, #ARRUS-274.
    - Added support for DBARLite-PCIe and DBARLite 8-bit #M_USSS2-103, #US4R-333. 
    - Implemented probe connected check functionality for atl/philips2 adapter #US4R-314.
    - Made it possible to execute the sequence of: triggerStart, triggerStop, triggerStart #US4R-147 
    - Implemented File device (for simulated mode) #ARRUS-221, #ARRUS-248.
    - us4R: exposed us4OEM and digital backplane serial number and revision #ARRUS-267.
    - Made it possible to choose which us4OEM produces frame metadata #US4R-324.
    - Added Session::getCurrentState method, which returns the current state of the session (e.g if the devices are running or not) #ARRUS-273.
    - us4R: Provided the possibility to explicitly specify what digital backplane is used #US4R-352.
    - us4R Linux driver: disabled trigger stopping in the linux character device release: NOTE: please make sure to stop the system before disconnecting the probe #US4R-338.
    - Default HV voltage (5V) is set on us4r device initialization #ARRUS-41.
    
- Python API:

    - us4R: exposed SYNC mode #ARRUS-83.
    - Added support for Python 3.9 and Python 3.10 #ARRUS-208.
    - Added the possibility to change session medium after its initialization #ARRUS-232.
    - Added the possibility to specify rx depth range in the SimpleTxRxSequences #ARRUS-232.
    - Exposed OX min/max position of the probe elements #ARRUS-232.
    - Created cupy -> DL pack operator #ARRUS-232.
    - Created an example for file device (simulated mode) #221, #ARRUS-248.
    - Created dictionary of media #ARRUS-231, #ARRUS-247.
    - Added spacing to echo data description #ARRUS-251.
    - Python API minor package fixes: fixed imaging pipeline parameter setting #ARRUS-265.
    - us4R: exposed us4OEM and digital backplane serial number and revision #ARRUS-267.
    - us4R: provided the possibility to dynamically change TX focus, #US4R-326.
    - us4R: added high pass filter in the probe check implementation to support us4OEM+ #US4R-356.
    - Made tgc_curve parameter optional (default: no TGC) #ARRUS-278.

- MATLAB API:

    - Fixed issues with TGC in Matlab, enable the RF display #ARRUS-202, ARRUS-203.
    - Color Doppler - thresholding and wall clutter filtration #ARRUS-239.


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

0.7.8
.....

- core (C++): Fixed frame channel mapping for RX aperture smaller than 32 elements, and for non-standard probe adapters (that do not satisfy channel group modulo-32 congruency).


0.7.7
.....

- Python API: Fixed classical beamforming RF data reordering for rx aperture < 64 elements, n_repeats > 1.


0.7.6
.....

- Python API: Now it should be possible to use arrus.devices.us4r.Us4r.set_tgc when arrus.ops.us4r.TxRxSequence is uploaded.


0.7.5
.....

- Python API: Fixed CUDA host memory unpinning on the arrus session close.


0.7.4
.....

- Python API: Fixed 2D display for scheme with multiple outputs (layers).


0.7.3
.....

- core: Updated us4r-api to 0.8.7 (support multiple versions of US4RDBAR firmware).

0.7.2
.....

- Matlab API:

    - Removed the requirement of integer values for txNPeriods in MATLAB API.

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

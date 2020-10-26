Release notes
=============

Version 0.4.6
-------------

- MATLAB API:

    - Added parameter ``channelsMask`` to structure ``probe``. The parameter disables selected channels from Tx/Rx.
    - The list of Us4R class constructor parameters has been changed from list of positional \
      parameters to a list of pairs (name, value). E.g. use \
      ``Us4R('nArius', 2, 'voltage', 20, 'probeName', 'SL1543', 'adapterType', 'esaote3')``.
    - Added ``dynamicRange`` parameter to ``BModeDisplay`` class.
    - Temporarily, the maximum number of tx/rxs is limited to 2048.

Version 0.4.5
-------------

- MATLAB API:

    - Support for Esaote SP2430 has been added. The new probe is identified by ``SP2430``.

Version 0.4.4
-------------

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

Version 0.4.3
-------------

- Added signed drivers for Windows, changed driver name from
  ``ARIUS`` to ``us4OEM``.
  Check :ref:`arrus-installation` procedure for information on how to
  update drivers.

Version 0.4.2
-------------

- MATLAB API:

    - ``rxApertureCenter``, ``rxCenterElement``, and ``rxApertureSize`` parameters added \
      to control the rx aperture. Use them the same way as parameters for tx aperture control. \
      The rx aperture is now fully defined by those parameters and sequence type no longer affects it.
    - rx delay is set to zero to allow for shallow region imaging.
    - ``nRepetitions``: set it to "max" to acquire the maximum number of repetitions allowable.

Version 0.4.1
-------------

- MATLAB API:

    - ``nRepetitions`` parameter added to control the number of rf-data frames to be recorded.
    - ``rxDepthRange`` parameter added to control the starting & ending depth of recorded data.
    - ``rxNSamples`` parameter, if it is a 2-element vector, allows for the setting of the starting & ending \
      sample number for the recorded data. The 1-element option is still valid.

- python API:

    - Created new API with a model of ``operations`` executed on available \
      devices.
    - Added ``fs_divider`` parameter to ``arrus.ops.Rx`` operation, which allows \
      to reduce the sampling frequency of the module.
    - Added asynchronous communication with the device using the ``arrus.ops.Loop`` \
      operation.
    - Added ``arrus.Us4OEMCfg`` parameter that turns on data transfer loggging \
      time.

Version 0.4.0
-------------

- MATLAB API:

    - Created a new version of the Matlab API (with STA and PWI sequences, for Esaote and Ultrasonix probes).
    - Added a simplified TGC control through tgcStart and tgcSlope parameters.
    - Added a classical linear scanning example (check :class:`arrus.LINSequence`).
    - From now on, txPri takes values in seconds (previously in [us]).

- python API:

    - Simplified ``Us4OEM.set_tgc_samples`` - it now takes values from range \
      range [0, 1], where 0 means maximum gain, 1 means minimum gain
    - Added ``Us4OEM.set_active_channel_group``, which allows to choose which \
      groups of channels can be active.

Version 0.3.0
-------------
- Added a function to set an rx/tx aperture mask - see Us4OEM.SetTxAperture(aperture) and Us4OEM.SetRxAperture(aperture)
- Renamed the Arius project to ARRUS. Renamed ``Arius`` module to ``Us4OEM``.

Version 0.2.0
-------------
- Added functions to trigger TX pulse asynchronously (Us4OEM.{TriggerStart, TriggerStop}).
- Added functions to enable/disable TGC and set TGC samples.
- Function Us4OEM.SetTxPeriods is no longer available; use Us4OEM.SetTxHalfPeriods instead.

Version 0.1.1
-------------
- Fixed some of the important bugs and errors in python API.

Version 0.1.0
-------------
Initial release of arrus.

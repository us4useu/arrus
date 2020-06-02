Release notes
=============

Version 0.4.1
-------------

- MATLAB API:

    - ``nRepetitions`` parameter added to control the number of rf-data frames to be recorded.
    - ``rxDepthRange`` parameter added to control the starting & ending depth of recorded data.
    - ``rxNSamples`` parameter, if it is 2-elements vector, allows for setting starting & ending \
      sample number of the recorded data. 1-element option is still valid.

- python API:

    - Created new API (for a model of ``operations`` executed on available \
      devices.
    - Added ``fs_divider`` parameter to ``arrus.ops.Rx`` operation, that allows \
      to reduce sampling frequency of the module.
    - Added asynchronous communication with the device using ``arrus.ops.Loop`` \
      operation.
    - Added ``arrus.Us4OEMCfg`` parameter that turns on data transfer loggging \
      time.

Version 0.4.0
-------------

- MATLAB API:

    - Created new version of the Matlab API (with STA and PWI sequences, for Esaote and Ultrasonix probes).
    - Added simplified TGC control through tgcStart and tgcSlope parameters.
    - Added classical linear scanning example (check :class:`arrus.LINSequence`).
    - From now on txPri takes values in seconds (previously was in [us]).

- python API:

    - Simplified ``Us4OEM.set_tgc_samples`` - now it takes values from range \
      range [0, 1], where 0 means maximum gain, 1 means minimum gain
    - Added ``Us4OEM.set_active_channel_group``, which allows to chose which \
      groups of channels can be active.

Version 0.3.0
-------------
- Added a function to set rx/tx aperture mask - see Us4OEM.SetTxAperture(aperture) and Us4OEM.SetRxAperture(aperture)
- Renamed the Arius project to ARRUS. Renamed ``Arius`` module to ``Us4OEM``.

Version 0.2.0
-------------
- Added functions to trigger TX pulse asynchronously (Us4OEM.{TriggerStart, TriggerStop}).
- Added functions to enable/disable TGC and set TGC samples.
- Function Us4OEM.SetTxPeriods is no more available; use Us4OEM.SetTxHalfPeriods instead.

Version 0.1.1
-------------
- Fixed some of the more important bugs and errors in python API.

Version 0.1.0
-------------
Initial release of arrus.

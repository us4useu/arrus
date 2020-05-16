Release notes
=============

Version 0.4.0
-------------

- MATLAB API:

    - Created new version of the Matlab API (with STA and PWI sequences, for Esaote and Ultrasonix probes).
    - Added simplified TGC control through tgcStart and tgcSlope parameters.
    - Added classical linear scanning example (check :class:`arrus.LINSequence`).
    - From now on txPri takes values in seconds (previously was in [us]).

Version 0.3.0
-------------
- Added a function to set rx/tx aperture mask - see Us4OEM.SetTxAperture(aperture) and Us4OEM.SetRxAperture(aperture)
- Renamed the Arius project to ARRUS. Renamed Arius module to Us4OEM.

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

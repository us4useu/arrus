Release notes
=============

Version 0.3.0
-------------
- Added a function to set rx/tx aperture mask - see Us4OEM.set_tx_aperture(aperture) and Us4OEM.set_rx_aperture(aperture)
- Renamed the Arius project to ARRUS. Renamed Arius module to Us4OEM.

Version 0.2.0
-------------
- Added functions to trigger TX pulse asynchronously (Us4OEM.{trigger_start, trigger_stop}).
- Added functions to enable/disable TGC and set TGC samples.
- Function Us4OEM.set_tx_periods is no more available; use Us4OEM.set_tx_half_periods instead.

Version 0.1.1
-------------
- Fixed some of the more important bugs and errors in python API.

Version 0.1.0
-------------
Initial release of arrus.
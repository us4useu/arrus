Release notes
=============

Version 0.3.0
-------------
- Added a function to set rx/tx aperture mask - see AriusCard.set_tx_aperture(aperture) and AriusCard.set_rx_aperture(aperture)

Version 0.2.0
-------------
- Added functions to trigger TX pulse asynchronously (AriusCard.{trigger_start, trigger_stop}).
- Added functions to enable/disable TGC and set TGC samples.
- Function AriusCard.set_tx_periods is no more available; use AriusCard.set_tx_half_periods instead.

Version 0.1.1
-------------
- Fixed some of the more important bugs and errors in python API.

Version 0.1.0
-------------
Initial release of arius-sdk.
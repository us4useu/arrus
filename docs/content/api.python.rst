.. _arrus-api:

=============
API Reference
=============

Session
=======

.. autoclass:: arrus.Session
    :members:

Operations
==========

Scheme
------

.. autoclass:: arrus.ops.us4r.Scheme
    :members:
    :show-inheritance:


.. autoclass:: arrus.ops.us4r.DataBufferSpec
    :members:
    :show-inheritance:


Common Tx/Rx sequences
----------------------

.. autoclass:: arrus.ops.imaging.SimpleTxRxSequence
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.imaging.LinSequence
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.imaging.PwiSequence
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.imaging.StaSequence
    :members:
    :show-inheritance:


Custom Tx/Rx sequences
----------------------

.. autoclass:: arrus.ops.us4r.TxRxSequence
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.us4r.Pulse
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.us4r.TxRx
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.us4r.Tx
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.us4r.Rx
    :members:
    :show-inheritance:


Devices
=======

Probe
-----

.. autoclass:: arrus.devices.probe.ProbeModel
    :members:
    :show-inheritance:

.. autoclass:: arrus.devices.probe.Lens
    :members:
    :show-inheritance:

.. autoclass:: arrus.devices.probe.MatchingLayer
    :members:
    :show-inheritance:


Ultrasound
----------

**Do not create instances of the below classes directly**. Use
``session.get_device`` to acquire the appropriate device, for example:
``session.get_device('/Us4R:0')``.

.. autoclass:: arrus.devices.us4r.Us4R
    :members:
    :exclude-members: start, stop, get_probe_model
    :show-inheritance:


The table below shows which TX/RX parameters can be changed in the system run-time (i.e. after starting
the system with `start_scheme()` method), and which currently require performing the `session.upload` method.

+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| **Name**                                           | **Possible to change in run-time?**   | **Interface**                       | **Possible to set per TX/RX?**   |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| **RX**                                                                                                                                                              |
|                                                                                                                                                                     |
|                                                                                                                                                                     |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| Aperture                                           | No                                | session.upload(),                       | Yes (local)                      |
|                                                    |                                   | arrus.ops.us4r.Rx()                     |                                  |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| Sample Range                                       | No                                | session.upload(),                       | Yes (local), however all Rxs     |
|                                                    |                                   | arrus.ops.us4r.Rx()                     | must acquire the same number of  |
|                                                    |                                   |                                         | samples                          |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| Downsampling factor                                | No                                | session.upload(),                       | No (global per sequence)         |
|                                                    |                                   | arrus.ops.us4r.Rx()                     |                                  |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| PGA gain                                           | Yes                               | set_pga_gain()                          | No (global)                      |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| LNA gain                                           | Yes                               | set_lna_gain()                          | No (global)                      |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| Analog TGC curve                                   | Yes                               | set_tgc()                               | No (global)                      |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| DTGC attenuation                                   | Yes                               | set_dtgc_attenuation()                  | No (global)                      |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| Digital high-pass filter (HPF) corner frequency    | Yes                               | set_hpf_frequency(), disable_hpf()      | No (global)                      |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| Digital Down Conversion (demodulation frequencies, | No                                | session.upload(),                       | No (global)                      |
| filter coefficients)                               |                                   | arrus.ops.us4r.DigitalDownConersion     |                                  |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| **TX**                                                                                                                                                              |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| Aperture                                           | No                                | session.upload(),                       | Yes (local)                      |
|                                                    |                                   | arrus.ops.us4r.Tx()                     |                                  |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| Excitation                                         | No                                | session.upload(),                       | Yes (local)                      |
|                                                    |                                   | arrus.ops.us4r.Tx(),                    |                                  |
|                                                    |                                   | arrus.ops.us4r.Pulse()                  |                                  |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| Delays/TX focus, angle, speed of sound             | Partially yes: possible to create | session.upload(),                       | Yes (local)                      |
|                                                    | a set of pre-defined TX focuses.  | arrus.ops.us4r.Tx()                     |                                  |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| **TX/RX**                                                                                                                                                           |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+
| PRI                                                | No                                | session.upload(), arrus.ops.us4r.TxRx() | Yes (local)                      |
+----------------------------------------------------+-----------------------------------+-----------------------------------------+----------------------------------+


.. autoclass:: arrus.devices.us4oem.Us4OEM
    :members:
    :show-inheritance:

.. autoclass:: arrus.devices.us4r.Backplane
    :members:
    :show-inheritance:


Processing devices
------------------

.. autoclass:: arrus.devices.gpu.GPU
    :show-inheritance:

.. autoclass:: arrus.devices.cpu.CPU
    :show-inheritance:

Output data
===========

An instance of the following class is returned by the ``sesion.upload`` function:

.. autoclass:: arrus.framework.DataBuffer
    :members: append_on_new_data_callback
    :show-inheritance:

Data buffers consists of multiple buffer elements.

.. autoclass:: arrus.framework.DataBufferElement
    :members: data
    :show-inheritance:


Metadata
--------

.. autoclass:: arrus.metadata.Metadata
    :members:
    :show-inheritance:

.. autoclass:: arrus.metadata.EchoDataDescription
    :members:
    :show-inheritance:

.. autoclass:: arrus.metadata.FrameAcquisitionContext
    :members:
    :show-inheritance:


Utility functions
=================

B-mode imaging pipeline using cupy/numpy
----------------------------------------

.. autoclass:: arrus.utils.imaging.Pipeline
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.FirFilter
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.BandpassFilter
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.QuadratureDemodulation
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Decimation
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.RxBeamforming
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.EnvelopeDetection
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Transpose
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.ScanConversion
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.LogCompression
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Lambda
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.SelectFrames
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Squeeze
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.ReconstructLri
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Sum
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Mean
    :show-inheritance:

Logging
-------
.. autofunction:: arrus.set_clog_level

.. autofunction:: arrus.add_log_file

The following log severity levels are available:
``arrus.logging.TRACE``,
``arrus.logging.DEBUG``,
``arrus.logging.INFO``,
``arrus.logging.WARNING``,
``arrus.logging.ERROR``,
``arrus.logging.FATAL``

Probe check functionality
-------------------------

.. autoclass:: arrus.utils.probe_check.FeatureDescriptor
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.ElementValidationVerdict
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.ProbeElementValidatorResult
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.ProbeElementFeatureDescriptor
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.ProbeElementHealthReport
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.ProbeHealthReport
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.MaxAmplitudeExtractor
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.EnergyExtractor
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.SignalDurationTimeExtractor
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.MaxHVPSCurrentExtractor
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.ByThresholdValidator
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.ByNeighborhoodValidator
    :members:
    :show-inheritance:

.. autoclass:: arrus.utils.probe_check.ProbeHealthVerifier
    :members:
    :show-inheritance:


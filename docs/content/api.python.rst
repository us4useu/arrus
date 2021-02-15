.. _api-main:

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

.. autoclass:: arrus.ops.imaging.LinSequence
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.imaging.PwiSequence
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

**Do not create instances of the below classes directly**. Use
``session.get_device`` to acquire the appropriate device, for example:
``session.get_device('/Us4R:0')``.

.. autoclass:: arrus.devices.us4r.Us4R
    :members:
    :exclude-members: start, stop, set_tgc
    :show-inheritance:

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

.. autoclass:: arrus.framework.BufferElement
    :members: data
    :show-inheritance:


Metadata
--------

.. autoclass:: arrus.metadata.ConstMetadata
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

.. autoclass:: arrus.utils.imaging.BandpassFilter
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Filter
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

.. autoclass:: arrus.utils.imaging.Enqueue
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.SelectFrames
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Squeeze
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.ReconstructLri
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Sum
    :show-inheritance:

.. autoclass:: arrus.utils.imaging.Average
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

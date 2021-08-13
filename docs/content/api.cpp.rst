.. _arrus-api:

=================
C++ API Reference
=================

.. caution::

    ARRUS C++ API is currently under development and its API will be modified in the future. Please expect breaking changes.

Notation
========

- ``Class::Handle`` is a typedef for ``std::unique_ptr<Class>``
- ``Class::SharedHandle`` is a typedef for ``std::shared_ptr<Class>``


Session
=======

.. doxygenfunction:: arrus::session::createSession(const std::string &filepath)
    :project: arrus

.. doxygenclass:: arrus::session::Session
    :project: arrus
    :members:


Operations
==========

Scheme
------

..  doxygenclass:: arrus::ops::us4r::Scheme
    :project: arrus
    :members:


.. doxygenclass:: arrus::framework::DataBufferSpec
    :project: arrus
    :members:


Custom Tx/Rx sequences
----------------------

.. doxygenclass:: arrus::ops::us4r::TxRxSequence
    :project: arrus
    :members:

.. doxygenclass:: arrus::ops::us4r::Pulse
    :project: arrus
    :members:

.. doxygenclass:: arrus::ops::us4r::TxRx
    :project: arrus
    :members:

.. doxygenclass:: arrus::ops::us4r::Tx
    :project: arrus
    :members:

.. doxygenclass:: arrus::ops::us4r::Rx
    :project: arrus
    :members:

.. doxygenclass:: arrus::ops::us4r::TxRxSequence
    :project: arrus
    :members:

.. doxygentypedef:: arrus::ops::us4r::TGCCurve

Devices
=======

.. doxygenclass:: arrus::devices::Us4R
    :project: arrus
    :members: setVoltage, disableHV

Output data
===========

An instance of the following class is returned by the ``sesion.upload`` function:

.. doxygenclass:: arrus::session::UploadResult
    :project: arrus
    :members:

Upload result can include some additional information (metadata) about the acquired data:

.. doxygenclass:: arrus::session::UploadConstMetadata
    :project: arrus
    :members:

Currently uploading scheme for the Us4R device returns a metadata with a
key ``frameChannelMapping``; the metadata value type is:

.. doxygenclass:: arrus::devices::FrameChannelMapping
    :project: arrus
    :members:

Upload result contains also a handle to the output data buffer.

.. doxygenclass:: arrus::framework::DataBuffer
    :project: arrus
    :members:

Data buffers consists of multiple elements.

.. doxygenclass:: arrus::framework::BufferElement
    :project: arrus
    :members:

.. doxygenclass:: arrus::framework::NdArray
    :project: arrus
    :members:

.. doxygenclass:: arrus::Tuple
    :project: arrus
    :members:



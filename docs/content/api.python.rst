.. _api-main:

=============
API Reference
=============

.. caution::

    We will do our best to maintain backward compatibility but please note
    that ARRUS is currently under development and its API may be modified in the future.


Devices
=======

**Do not create instances of the below classes directly**. Use
``session.get_device`` to acquire the appropriate device, for example:
``session.get_device('Us4OEM:0')``.

.. autoclass:: arrus.devices.us4oem.Us4OEM
    :members: get_sampling_frequency, get_n_rx_channels, get_n_tx_channels
    :show-inheritance:

.. autoclass:: arrus.devices.hv256.HV256
    :show-inheritance:


Operations
==========

.. autoclass:: arrus.ops.TxRx
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.Tx
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.Rx
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.Sequence
    :members:
    :show-inheritance:

.. autoclass:: arrus.ops.Loop
    :members:
    :show-inheritance:

Parameters
----------

Excitation
~~~~~~~~~~

.. autoclass:: arrus.Excitation
    :members:

.. autoclass:: arrus.SineWave
    :members:
    :show-inheritance:

Aperture
~~~~~~~~

.. autoclass:: arrus.MaskAperture
    :members:
    :show-inheritance:

.. autoclass:: arrus.RegionBasedAperture
    :members:
    :show-inheritance:

.. autoclass:: arrus.SingleElementAperture
    :members:
    :show-inheritance:

Session
=======

.. autoclass:: arrus.Session
    :members:

Configuration
=============

The configuration to apply to each of the Arrus components.

.. autoclass:: arrus.CustomUs4RCfg
    :members:
    :show-inheritance:

.. autoclass:: arrus.Us4OEMCfg
    :members:
    :show-inheritance:

.. autoclass:: arrus.SessionCfg
    :members:
    :show-inheritance:

.. autoclass:: arrus.ChannelMapping
    :members:

Utility functions
=================

Logging
-------
.. autofunction:: arrus.set_log_level

.. autofunction:: arrus.add_log_file

Legacy API Reference
====================

.. danger::

    The API below is deprecated and we discourage using it. The legacy API
    will be removed in the near future.

Interface
---------

An interface contains all the information required to configure the probe adapter
provided with the us4OEM module. Use the following function to obtain the
appropriate interface.


.. autofunction:: arrus.interface.get_interface

Currently, ``esaote`` and ``ultrasonix`` interfaces are implemented.

.. autoclass:: arrus.interface.UltrasoundInterface
    :members:

Interactive session
-------------------

A session object allows you to obtain the appropriate device handler.

.. autoclass:: arrus.session.InteractiveSession
    :members:

Devices
-------

Only the Us4OEM device handler is provided.

.. autoclass:: arrus.devices.us4oem.Us4OEM
    :members:
    :noindex:



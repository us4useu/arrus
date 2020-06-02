.. _api-main:

=============
API Reference
=============

.. caution::

    We will do our best to maintain backward compatibility, but please note
    that ARRUS is under development and its API may be modified in the future.


Devices
=======

**Do not create instances of the below classes directly**. Use
``session.get_device`` to acquire appropriate device, for example:
``session.get_device('Us4OEM:0')``.

.. autoclass:: arrus.devices.us4oem.Us4OEM
    :members: get_sampling_frequency, get_n_rx_channels, get_n_tx_channels

.. autoclass:: arrus.devices.hv256.HV256

Configuration
=============

Configuration to apply to each of the system components.

.. autoclass:: arrus.system.SystemCfg
    :members:
    :undoc-members:


Operations
==========

Utility functions
=================

Logging
-------
.. autofunction:: arrus.set_log_level

.. autofunction:: arrus.add_log_file

Legacy API Reference
====================

.. danger::

    Below API is deprecated and it is discouraged to use it. Legacy API
    will be removed in near future.

Interface
---------

An interface contains all information required to configure probe's adapter
which is provided with us4OEM module. Use following function to obtain an
appropriate interface.


.. autofunction:: arrus.interface.get_interface

Currently, an ``esaote`` and ``ultrasonix`` interface are implemented.

.. autoclass:: arrus.interface.UltrasoundInterface
    :members:

Interactive session
-------------------

A session object allows you to obtain an appropriate device handler.

.. autoclass:: arrus.session.InteractiveSession
    :members:

Devices
-------

Only the Us4OEM device handler is provided.

.. autoclass:: arrus.devices.us4oem.Us4OEM
    :members:
    :noindex:



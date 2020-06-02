.. _api-main:

=============
API Reference
=============

.. caution::

    ARRUS is currently under development and its API will be modified in
    the future. Please expect breaking changes.

Configuration
=============

Logging
-------
.. autofunction:: arrus.set_log_level

Interface
---------

An interface contains all information required to configure probe's adapter
which is provided with us4OEM module. Use following function to obtain an
appropriate interface.


.. autofunction:: arrus.interface.get_interface

Currently, only an ``esaote`` interface is implemented.

.. autoclass:: arrus.interface.UltrasoundInterface
    :members:

Session
=======

A session object allows you to obtain an appropriate device handler.

.. autoclass:: arrus.session.InteractiveSession
    :members:

Devices
=======

Currently only the Us4OEM device handler is provided.

.. autoclass:: arrus.devices.us4oem.Us4OEM
    :members:



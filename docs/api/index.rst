.. _api-main:

===
API
===

.. caution::

    Arius SDK is currently under development and its API will be modified in
    the future. Please expect breaking changes.

Configuration
=============

Interface
---------

An interface contains all information required to configure probe's adapter
which is provided with us4OEM module. Use following function to obtain an
appropriate interface.


.. autofunction:: arius.interface.get_interface

Currently, only an ``esaote`` interface is implemented.

.. autoclass:: arius.interface.UltrasoundInterface
    :members:

Session
=======

A session object allows you to obtain an appropriate device handler.

.. autoclass:: arius.session.InteractiveSession
    :members:

Devices
=======

Currently only the AriusCard device handler is provided.

.. autoclass:: arius.devices.arius.AriusCard
    :members:



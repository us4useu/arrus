.. _arrus-installation:

============
Installation
============

Requirements
============

Microsoft® Windows 10 x64 operating system is supported.

Make sure that you have installed following dependencies:

- `Microsoft Visual C++ Redistributable for Visual Studio 2017 <https://aka.ms/vs/16/release/vc_redist.x64.exe>`_

Installation
============

Startup
-------

.. caution::

    If you use **MSI GS65 notebook** and connect with the device via Thunderbolt-3
    cable, you have to follow this extended startup procedure:

    1. Turn off the notebook and your device, plug off Thunderbolt-3 cable.
    2. Turn on the system, then connect it to the notebook using the provided
       Thunderbolt-3 cable.
    3. Turn on the notebook.
    4. After Windows 10 loads, restart the notebook.

Drivers
-------

Make sure, that your us4R-lite device is properly connected via Thunderbolt-3
cable and is enabled in your Thunderbolt software, e.g.:

.. figure:: img/thunderbolt.png
    :scale: 80%

The `Connection status` should be `Connected` (or something similar).

Uninstall ARIUS drivers (if previously installed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If ARIUS drivers are installed on your computer, uninstall them first. ARIUS
drivers are the legacy drivers that were required before 0.4.3 version.

1. Open Windows Device Manager, uninstall all ``ARIUS`` and ``WinDriver1290``
   devices available in the "Jungo Connectivity" node. **Check
   "Delete the driver software for this device"**.

.. figure:: img/uninstall_arius_drv.png
    :scale: 100%

2. Restart computer.


Install Us4OEM drivers
~~~~~~~~~~~~~~~~~~~~~~

1. Download and extract ``us4oem-drivers-1290.zip`` (ask us4us support to get the newest version).
2. Run ``install.bat`` with **administrative privileges**. Confirm driver
   installation if necessary.

As a result, ``us4oem`` and ``us4OEM`` nodes should be visible in the
Device Manager.

.. figure:: img/dev_manager.png
    :scale: 100%


ARRUS
-----

Before proceeding please make sure the device is properly connected to the computer.

1. Download and extract |arrus|_ package.
2. Run ``install.exe`` file and follow provided instructions.

.. _fig-install_result:
.. figure:: img/install_result.png


.. include:: api.rst






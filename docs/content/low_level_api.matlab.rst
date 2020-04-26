.. _api-main:

============
Us4OEM API
============

.. caution::

    ARRUS is currently under development and its API will be modified in the future. Please expect breaking changes.


General formula
===============

Use ``Us4MEX`` MATLAB function to communicate with the provided hardware. This function allows to perform some particular **operation** with
given **parameters** on a selected **us4OEM module** (a card).

Before calling ``Us4MEX`` function make sure that a variable ``nUs4OEM`` is set in you current workspace and is equal
to the number of available modules.

..  mat:function:: Us4MEX(moduleIndex, operation, varargin)

    Executes an operation on a module with a given index.

    :param moduleIndex: index of a module, on which operation should be performed
    :param operation: a name of an operation (string) to perform
    :param varargin: a list of parameters


Operations
==========

Below we present a list of operations that can be performed on us4OEM module using Us4MEX function.

.. _mex-SetRxChannelMapping:

SetRxChannelMapping
-------------------
..  mat:function:: Us4MEX(moduleIndex, "SetRxChannelMapping", srcChannel, dstChannel)

    Sets RX mapping from the source (module's) channel to the destination (Probe's) channel.

    :param srcChannel: source channel number
    :param dstChannel: destination channel number

.. _mex-SetTxChannelMapping:

SetTxChannelMapping
-------------------

..  mat:function:: Us4MEX(moduleIndex, "SetTxChannelMapping", srcChannel, dstChannel)

    Sets TX mapping from the source (module's) channel to the destination (Probe's) channel.

    :param srcChannel: source channel number
    :param dstChannel: destination channel number

.. _mex-ScheduleReceive:

ScheduleReceive
---------------

..  mat:function:: Us4MEX(moduleIndex, "ScheduleReceive", address, length)

    Schedules a new data transmission from the probe's adapter to the module's internal memory.

    This function queues a new data transmission from all available RX channels to the device's internal memory.
    Data transfer starts with the next "SWTrigger" operation call.

    :param address: module's internal memory address (a number), where RX data should be saved
    :param length: number of samples from channel to acquire

.. _mex-ClearScheduledReceive:

ClearScheduledReceive
---------------------

..  mat:function:: Us4MEX(moduleIndex, "ClearScheduledReceive")

    Clears a queue of RX tasks, should be called before defining any new TX/RX scheme.

.. _mex-SetNumberOfFirings:

SetNumberOfFirings
------------------

..  mat:function:: Us4MEX(moduleIndex, "SetNumberOfFirings", numberOfFirings)

    Sets number firings/acquisitions for new TX/RX sequence. For each firing/acquisition a different TX/RX parameters can be applied.

    :param numberOfFirings: number of firings to set

.. _mex-SetTxDelay:

SetTxDelay
----------

..  mat:function:: Us4MEX(moduleIndex, "SetTxDelay", channel, delay, firingIndex)

    Sets TX delay for a given channel. Returns an exact delay value that has been set on a give module.

    :param channel: channel number, **starts from 1**
    :param delay: delay to set in seconds (double)
    :param firingIndex: a firing, in which the delay should apply, **starts from 0**
    :return: an exact delay value that was set for a given channel

.. _mex-SetTxDelays:

SetTxDelays
-----------

..  mat:function:: Us4MEX(moduleIndex, "SetTxDelays", delays, firingIndex)

    Sets delays on the whole TX aperture. Returns an array of delays that has been set on a given module.

    :param delays: an array of delays to set (with a length the same as the number of available TX channels), in seconds
    :param firingIndex: a firing, in which the delays should apply, **starts from 0**
    :return: an array of delays that has been set on a given module.

.. _mex-SetTxFrequency:

SetTxFrequency
--------------

..  mat:function:: Us4MEX(moduleIndex, "SetTxFrequency", frequency, firingIndex)

    Sets TX frequency.

    :param frequency: frequency to set in Hz
    :param firingIndex: a firing, in which the parameter value should apply, **starts from 0**
    :return: an exact value of TX frequency that was set on given module

.. _mex-SetTxHalfPeriods:

SetTxHalfPeriods
----------------

..  mat:function:: Us4MEX(moduleIndex, "SetTxPeriods", nPeriods, firingIndex)

    Sets number of TX signal half-periods.

    :param nPeriods: number of half-periods to set
    :param firingIndex: a firing, in which the parameter value should apply, **starts from 0**
    :return: an exact number of half-periods that has been set on a given module


.. _mex-SetRxAperture:

SetRxAperture
-------------

..  mat:function:: Us4MEX(moduleIndex, "SetRxAperture", origin, size, acqIndex)

    Sets RX aperture's origin and size.

    :param origin: origin of the aperture
    :param size: size of the aperture
    :param acqIndex: an acquisition, in which the parameter value should apply, **starts from 0**

.. _mex-SetTxAperture:

SetTxAperture
-------------

..  mat:function:: Us4MEX(moduleIndex, "SetTxAperture", origin, size, firingIndex)

    Sets TX aperture's origin and size.

    :param origin: origin of the aperture
    :param size: size of the aperture
    :param firingIndex: a firing, in which the parameter value should apply, **starts from 0**

.. _mex-SetRxTime:

SetRxTime
---------

..  mat:function:: Us4MEX(moduleIndex, "SetRxTime", time, acqIndex)

    Sets length of acquisition time.

    :param time: expected acquisition time, in seconds
    :param acqIndex: an acquisition, in which the parameter value should apply, **starts from 0**

.. _mex-SetTxInvert:

SetTxInvert
-----------
..  mat:function:: Us4MEX(moduleIndex, "SetTxInvert", onoff, firingIndex)

    Enables/disables inversion of TX signal.

    :param onoff: enable/disable inversion
    :param firingIdx:  a firing, in which the parameters values should apply, **starts from 0**

.. _mex-SetTxCw:

SetTxCw
-------
..  mat:function:: Us4MEX(moduleIndex, "SetTxCw", onoff, firingIndex)

    Enables/disables generation of long TX bursts.

    :param onoff: enable/disable
    :param firingIdx:  a firing, in which the parameters values should apply, **starts from 0**

.. _mex-SetRxDelay:

SetRxDelay
----------
..  mat:function:: Us4MEX(moduleIndex, "SetRxDelay", delay, acqIndex)

    Sets the starting point of the acquisition time [s].

    :param delay: expected acquisition time starting point relative to trigger [s]
    :param acqIndex: an acquisition, in which the parameter value should apply, **starts from 0**

.. _mex-EnableReceive:

EnableReceive
-------------

..  mat:function:: Us4MEX(moduleIndex, "EnableReceive")

    Enables RX data transfer from the probe's adapter to the module's internal memory.

.. _mex-EnableTransmit:

EnableTransmit
--------------

..  mat:function:: Us4MEX(moduleIndex, "EnableTransmit")

    Enables TX pulse generation.

.. _mex-SWTrigger:

SWTrigger
---------

..  mat:function:: Us4MEX(moduleIndex, "SWTrigger")

    Triggers pulse generation and starts RX transmissions on all (master and slave) modules. Should be called only for a master module.

.. _mex-SWNextTX:

SWNextTX
--------

..  mat:function:: Us4MEX(moduleIndex, "SWNextTX")

    Sets all TX and RX parameters for next firing/acquisition.


.. _mex-TransferRXBufferToHost:

TransferRXBufferToHost
----------------------

..  mat:function:: Us4MEX(moduleIndex, "TransferRXBuffertToHost", srcAddress, length)

    Transfers data from the given module's memory address to the computer's memory, and returns a new MATALB array
    of shape (number of RX channels, length)

    The resulting data will be of type int16.

    :param srcAddres: module's memory address to copy data from
    :param length: number of collected samples
    :return: a MATLAB array of shape (number of RX channels, length)

.. _mex-EnableTestPatterns:

EnableTestPatterns
------------------

..  mat:function:: Us4MEX(moduleIndex, "EnableTestPatterns")

    Turns off probe's RX data acquisition and turns on test patterns generation.
    When test patterns are enabled, sawtooth signal is generated.

.. _mex-DisableTestPatterns:

DisableTestPatterns
-------------------

..  mat:function:: Us4MEX(moduleIndex, "DisableTestPatterns")

    Turns off test patterns generation and turns on probe's RX data acquisition.


SyncTestPatterns
----------------
.. TODO(pjarosik) not sure of this description.

..  mat:function:: Us4MEX(moduleIndex, "SyncTestPatterns")

    Waits for update of test patterns.

.. _mex-SetPGAGain:

SetPGAGain
----------
..  mat:function:: Us4MEX(moduleIndex, "SetPGAGain", gain)

    Configures programmable-gain amplifier (PGA).

    :param gain: gain to set (**string**); available values: "24dB", "30dB"

.. _mex-SetLNAGain:

SetLNAGain
----------
..  mat:function:: Us4MEX(moduleIndex, "SetLNAGain", gain)

    Configures low-noise amplifier (LNA) gain.

    :param gain: gain to set (**string**); available values: "12dB", "18dB", "24dB"

.. _mex-SetDTGC:

SetDTGC
-------
..  mat:function:: Us4MEX(moduleIndex, "SetDTGC", isEnabled, attenuation)

    Configures time gain compensation (TGC).

    :param isEnabled: whether to enable (string "EN") or disable (string "DIS") time gain compensation
    :param attenuation: attenuation to set (**string**); available values: "0dB", "6dB", "12dB", "18dB", "24dB", "30dB", "36dB", "42dB"

.. _mex-TGCEnable:

TGCEnable
---------
..  mat:function:: Us4MEX(moduleIndex, "TGCEnable")

    Enables time gain compensation (TGC).

.. _mex-TGCDisable:

TGCDisable
----------
..  mat:function:: Us4MEX(moduleIndex, "TGCDisable")

    Disables time gain compensation (TGC).

.. _mex-TGCSetSamples:

TGCSetSamples
-------------
.. TODO(pjarosik) determine more verbose comment here

..  mat:function:: Us4MEX(moduleIndex, "TGCSetSamples")

    Sets samples for a time gain compensation (TGC).

    :param samples: an array of samples to set

.. _mex-SetLPFCutoff:

SetLPFCutoff
------------
..  mat:function:: Us4MEX(moduleIndex, "SetLPFCutoff", cutoffFrequency)

    Sets low-pass filter (LPF) cutoff frequency.

    :param cutoffFrequency: cutoff frequency to set (**string**), available values: "10MHz", "15MHz", "20MHz",
                            "30MHz", "35MHz", "50MHz"

.. _mex-SetActiveTermination:

SetActiveTermination
--------------------
..  mat:function:: Us4MEX(moduleIndex, "SetActiveTermination", isEnabled, value)

    Sets active termination.

    :param isEnabled: whether to enable (string "EN") or disable (string "DIS") active termination
    :param value: active termination value to set (**string**), available: "50", "100", "200", "400"

.. _mex-TriggerStart:

TriggerStart
------------
..  mat:function:: Us4MEX(moduleIndex, "TriggerStart")

    Starts generation of the hardware trigger.

.. _mex-TriggerStop:

TriggerStop
-----------
..  mat:function:: Us4MEX(moduleIndex, "TriggerStop")

    Stops generation of the hardware trigger.

.. _mex-TriggerSync:

TriggerSync
-----------
..  mat:function:: Us4MEX(moduleIndex, "TriggerSync")

    Resumes generation of the hardware trigger.

.. _mex-SetNTriggers:

SetNTriggers
------------
..  mat:function:: Us4MEX(moduleIndex, "SetNTriggers", n)

    Sets the number of triggers to be generated.

    :param n: number of triggers to set

.. _mex-SetTrigger:

SetTrigger
----------
..  mat:function:: Us4MEX(moduleIndex, "SetTrigger", timeToNextTrigger, timeToNextTx, syncReq, idx)

    Sets parameters of the trigger event.
	Each trigger event will generate a trigger signal for the current firing/acquisition and set next firing parameters.

    :param timeToNextTrigger: time between current and the next trigger [uS]
    :param timeToNextTx: delay between current trigger and setting next firing parameters [uS]
    :param syncReq: should the trigger generator pause and wait for the TriggerSync() call
    :param idx: a firing, in which the parameters values should apply, **starts from 0**


Single Module
=============

In the following example we show:

1. how to configure TX and RX subsystems in order to generate a plane wave,
2. how to trigger a pulse generation and acquire a complete RF frame.

A complete source code is available in a ``matlab/examples/basics`` directory and is divided into two parts:

1. script ``init_x1.m``, which initializes system base configuration,
2. script ``us4oem_x1.m``, which configures TX/RX scheme, and then in a loop: acquires current RF data frame and displays it.

To run example, you need to execute ``us4oem_x1.m`` only. This script calls ``init_x1.m`` before performing any
further processing.


Initialization
--------------

A base configuration is set using ``init_x1.m`` script.

First, a path to ``Us4MEX.mexw64`` file must be set. Assuming you are running the example from the directory, in which the script is originally located, following should work:

.. code-block:: matlab

    path(path, '..\..\');

Then indicate the number of available us4OEM modules:

.. code-block:: matlab

    nUs4OEM = 1;

We need to take into account a probe adapter that is currently installed on our board, thus an appropriate
RX and TX channels mapping must be set (using :ref:`mex-SetRxChannelMapping` and :ref:`mex-SetTxChannelMapping`):

.. code-block:: matlab

    probe = "AL2442";
    % Channel mapping for specific probe.
    if probe == "AL2442"
        rxChannelMap = zeros(nUs4OEM, 32);
        rxChannelMap(1,:) = 32:-1:1;
        rxChannelMap(1,16) = 16;
        rxChannelMap(1,17) = 17;
        txChannelMap = zeros(nUs4OEM, 128);
        txChannelMap(1,1:32) = 32:-1:1;
        txChannelMap(1,16) = 16;
        txChannelMap(1,17) = 17;
        txChannelMap(1,33:64) = 64:-1:33;
        txChannelMap(1,48) = 48;
        txChannelMap(1,49) = 49;
        txChannelMap(1,65:96) = 96:-1:65;
        txChannelMap(1,80) = 80;
        txChannelMap(1,81) = 81;
        txChannelMap(1,97:128) = 128:-1:97;
        txChannelMap(1,112) = 112;
        txChannelMap(1,113) = 113;
        for ch=1:32
            Us4MEX(0, "SetRxChannelMapping", rxChannelMap(1,ch), ch);
        end
        for ch=1:128
            Us4MEX(0, "SetTxChannelMapping", txChannelMap(1,ch), ch);
        end
    end

In this place we also set all RX parameters that we will not change later in the example:

.. code-block:: matlab

    % init RX
    Us4MEX(0, "SetPGAGain","30dB");
    Us4MEX(0, "SetLPFCutoff","15MHz");
    Us4MEX(0, "SetActiveTermination","EN", "200");
    Us4MEX(0, "SetLNAGain","24dB");
    % digital TGC
    Us4MEX(0, "SetDTGC","EN", "0dB");
    Us4MEX(0,"TGCSetSamples", uint16([hex2dec('9001'), hex2dec('4000')+(3000:-75:0), hex2dec('4000')+3000]));
    Us4MEX(0, "TGCEnable");

That is:

1. we set amplifier gain using :ref:`mex-SetPGAGain`,
2. set low-pass frequency cutoff using :ref:`mex-SetLPFCutoff`,
3. we enable active termination using :ref:`mex-SetActiveTermination`,
4. we set low-noise amplifier gain using :ref:`mex-SetLNAGain`,
5. enable digital time gain compensation using :ref:`mex-SetDTGC`,
6. turn on and set TGC samples using :ref:`mex-TGCEnable`, :ref:`mex-TGCSetSamples`.

Defining TX/RX acquisitions
---------------------------

In this example we want to transmit and capture a signal using 128 channels. In us4OEM module there are 32 receive channels in total,
but each receive channel is connected to 4 different transducers through the T/R switches.
This architecture enables handling 128 element probes with low-cost hardware. Full 128-channel data capture can be done
with a sequence of 4 transmit/receive acquisitions.

.. credits to DC

Definition of a TX/RX sequence is located in ``us4oem_x1.m`` file. We want to perform 4 TX/RX acquisition to complete one RF frame;
in order to do that, we need to define TX/RX parameters first, for each firing/acquisition (an *event*) separately.

.. code-block:: matlab

    % ...
    NEVENTS = 4;
    NSAMPLES = 8192;
    txDelays = (0:127)*0e-6/128;
    % ...

    % Define TX/RX scheme details to be executed on a Us4OEM module.

    Us4MEX(0, "ClearScheduledReceive")
    Us4MEX(0, "SetNTriggers", NEVENTS);
    Us4MEX(0, "SetNumberOfFirings", NEVENTS);
    for i=0:NEVENTS-1
        % TX
        txDelaysSet = Us4MEX(0, "SetTxDelays", txDelays, i);
        Us4MEX(0, "SetTxFrequency", 5e6, i);
        Us4MEX(0, "SetTxHalfPeriods", 2, i);
        Us4MEX(0, "SetTxAperture", 1, 128, i);
        % RX
        Us4MEX(0, "SetRxTime", 200e-6, i);
        Us4MEX(0, "SetRxDelay", 20e-6, i);
        Us4MEX(0, "SetRxAperture", i*32+1, 32, i);
        Us4MEX(0, "ScheduleReceive", i*NSAMPLES, NSAMPLES);
        % Trigger
        Us4MEX(0, "SetTrigger", 1000, 0, 0, i);
    end

    Us4MEX(0, "EnableTransmit");
    Us4MEX(0, "SetTrigger", 1000, 0, 1, NEVENTS-1);

Please note:

    - before defining TX/RX scheme you need to:
        - clear RX task queue using :ref:`mex-ClearScheduledReceive`,
        - set number of firings and triggers to perform on the device using :ref:`mex-SetNumberOfFirings`
          and :ref:`mex-SetNTriggers`,
    - functions: :ref:`mex-SetTxDelay`, :ref:`mex-SetTxFrequency`, :ref:`mex-SetTxHalfPeriods`, :ref:`mex-SetTxAperture`,
      :ref:`mex-SetRxAperture`, :ref:`mex-SetRxTime`, :ref:`mex-SetRxDelay` and :ref:`mex-SetTrigger`
      take an event number as a last parameter; for example, to set TX
      delay on the fifth channel in the fourth event you need to call: ``Us4MEX(0, "SetTxDelay", 5, delay, 3)``,
    - schedule new transfer from an ADC to us4OEM module's internal memory using :ref:`mex-ScheduleReceive`,
    - you have to set a maximum event number; use ``Us4MEX(0, "SetNumberOfFirings", NEVENTS)``,
    - you have to set a number of triggers to perform to acquire a single RF frame; use ``Us4MEX(0, "SetNTriggers", NEVENTS)``,
    - you have to enable TX before starting trigger pulse generation; use ``Us4MEX(0, "EnableTransmit")``,
    - to stop trigger generation after the last event set ``syncReq`` parameter of the ``SetTrigger`` function to ``1``,
      e.g. ``Us4MEX(0, "SetTrigger", 1000, 0, 1, NEVENTS-1)``

Acquiring data
--------------

To start TX signal generation, call :ref:`mex-TriggerStart`.

Before performing the data capture, we need to enable data reception with :ref:`mex-EnableReceive` function.
Then :ref:`mex-TriggerSync` should be called in order to wait for all the signal data to be collected.
After that a complete RF frame should be placed in the us4OEM module's internal memory.

In order to transfer it to the host computer's memory you can use function :ref:`mex-TransferRXBufferToHost`.

A complete data capture procedure is presented below:

.. code-block:: matlab

    Us4MEX(0, "TriggerStart");
    pause(0.1);
    while(ishghandle(h))
        Us4MEX(0, "EnableReceive");
        Us4MEX(0, "TriggerSync");
        pause(0.005);

        rf0 = Us4MEX(0, "TransferRXBufferToHost", 0, NSAMPLES * NEVENTS);
        rf(1:32,:)      = rf0(:,1:NSAMPLES);
        rf(33:64,:)     = rf0(:,NSAMPLES+1:2*NSAMPLES);
        rf(65:96,:)     = rf0(:,2*NSAMPLES+1:3*NSAMPLES);
        rf(97:128,:)    = rf0(:,3*NSAMPLES+1:4*NSAMPLES);

        % display rf variable...
    end
    Us4MEX(0, "TriggerStop");

Variable ``rf`` should now contain all the collected samples. To stop trigger generation, call :ref:`mex-TriggerStop`.


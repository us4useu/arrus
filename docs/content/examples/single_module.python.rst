Single Module
=============

.. caution::

    ARRUS is currently under development and its API will be modified in the
    future. Please expect breaking changes.

In the following example we show:

1. how to configure TX and RX subsystems in order to generate a plane wave,
2. how to trigger a pulse generation and acquire a complete RF frame.

Make sure that you have installed an appropriate ``arrus`` wheel file.

A complete source code is available in a ``python/examples/basics/us4oem_x1.py``.
To run it in your shell:

1. change your current location to a directory: ``python/example/basics``,
2. execute following command: ``python us4oem_x1.py``.


Initialization
--------------

First, an interactive session with a device must be created.
Assuming you are running the example from the directory, in which the script is originally located, following should work:

.. code-block:: python

    sess = ar.session.InteractiveSession("cfg.yaml")

The configuration file ``python/examples/basics/cfg.yaml`` contains information about
devices that should be visible in a session. In our example, cfg.yaml states
that:

- there is only one us4oem module available,
- no HV256 (power supplier) module will be used.

We also have to obtain a handle to the device with which we want to communicate:

.. code-block:: python

    module = sess.get_device("/Us4OEM:0")

Next, we need to take into account a probe adapter that is currently installed on our board, thus an appropriate
RX and TX channels mapping must be set:

.. code-block:: python

    interface = ar.interface.get_interface("esaote")
    module.store_mappings(
        interface.get_tx_channel_mapping(0),
        interface.get_rx_channel_mapping(0)
    )

Then, we start the device:

.. code-block:: python

    module.start_if_necessary()

In this place we also set all RX parameters that we will not change later in the example:

.. code-block:: python

    module.set_pga_gain(30) # [dB]
    module.set_lpf_cutoff(10e6) # [Hz]
    module.set_active_termination(200)
    module.set_lna_gain(24) #[dB]
    module.set_dtgc(0) # [dB]
    module.set_tgc_samples(
        [0x9001] \
        + (0x4000 + np.arange(1500, 0, -14)).tolist()
        + [0x4000 + 3000])
    module.enable_tgc()

That is:

1. we set amplifier gain,
2. set low-pass cutoff frequency,
3. we enable active termination,
4. we set low-noise amplifier gain,
5. and enable digital time gain compensation,
6. turn on TGC and set TGC samples.

Check :ref:`api-main` for more information on each method.

Defining TX/RX acquisitions
---------------------------

In this example we want to transmit and capture a signal using 128 channels.
In us4OEM module there are 32 receive channels in total, but each receive channel
is connected to 4 different transducers through the T/R switches.
This architecture enables handling 128 element probes with low-cost hardware.
Full 128-channel data capture can be done with a sequence of 4 transmit/receive acquisitions.

.. credits to DC

We want to perform 4 TX/RX acquisition to complete one RF frame;
in order to do that, we need to define TX/RX parameters first,
for each firing/acquisition (an *event*) separately.

.. code-block:: python

    TX_FREQUENCY = 5e6

    NEVENTS = 4
    NSAMPLES = 8192
    NCHANELS = module.get_n_rx_channels()
    delays = np.array([i*0.000e-6 for i in range(module.get_n_tx_channels())])

    # Clear RX tasks queue.
    module.clear_scheduled_receive()
    # Set number of triggers to perform for one RF data frame.
    module.set_n_triggers(NEVENTS)
    # Set number of firings to perform.
    module.set_number_of_firings(NEVENTS)

    for i in range(NEVENTS):
        module.set_tx_delays(delays=delays, firing=i)
        module.set_tx_frequency(frequency=5e6, firing=i)
        module.set_tx_half_periods(n_periods=2, firing=i)
        module.set_tx_invert(is_enable=False)
        module.set_tx_aperture(origin=0, size=128, firing=i)

        module.set_rx_time(time=200e-6, firing=i)
        module.set_rx_delay(delay=20e-6, firing=i)
        module.set_rx_aperture(origin=i*32, size=32, firing=i)
        module.schedule_receive(i*NSAMPLES, NSAMPLES)
        module.set_trigger(
            time_to_next_trigger=PRI,
            time_to_next_tx=0,
            is_sync_required=False,
            idx=i
        )
    module.enable_transmit()
    # In order to stop the device after the last event,
    # set 'is_sync_required=True'.
    module.set_trigger(
            time_to_next_trigger=PRI,
            time_to_next_tx=0,
            is_sync_required=True,
            idx=NEVENTS-1)


Acquiring data
--------------

To start TX signal generation call ``trigger_start`` function.

Before starting data capture, we need to enable it with
``enable_receive`` function. Then ``trigger_sync`` should be called to wait for
all the data to be collected. After that a complete RF frame should be available
in the us4OEM module's internal memory.

In order to transfer the data to the host computer's memory you have to use a
method ``transfer_rx_buffer_to_host``. Note, that this function returns an array
of shape ``(NEVENTS*NSAMPLES, NCHANNELS)``.
An additional reordering may be required - see example below.

.. code-block:: python

    module.trigger_start()
    # ...
    module.enable_receive()
    module.trigger_sync()

    # - transfer data from module's internal memory to the host memory
    buffer = module.transfer_rx_buffer_to_host(0, NEVENTS*NSAMPLES)

    # - reorder acquired data
    for i in range(NEVENTS):
        rf[:, i*NCHANELS:(i+1)*NCHANELS] = buffer[i*NSAMPLES:(i+1)*NSAMPLES, :]

    # ...
    # Stop the automatic trigger when no more data is necessary.
    module.trigger_stop()

Variable ``rf`` should now contain all the collected samples.
To stop trigger generation, call ``trigger_stop``.






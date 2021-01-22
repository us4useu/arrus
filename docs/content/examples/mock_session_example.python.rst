========
Examples
========

Mock session
------------

.. code-block:: python

    import numpy as np
    import arrus
    # import cupy as cp
    import matplotlib.pyplot as plt
    import h5py

    from arrus.ops.imaging import (
        LinSequence
    )
    from arrus.ops.us4r import (
        Pulse
    )

    from arrus.utils.imaging import (
        Pipeline,
        BandpassFilter,
        QuadratureDemodulation,
        Decimation,
        RxBeamforming,
        EnvelopeDetection,
        Transpose,
        ScanConversion,
        LogCompression
    )

    # Read the dataset do display.
    print("Reading data...")
    dataset = h5py.File("data.mat", mode="r")

    dataset = {
        "rf": np.array(dataset["rf"][:5, :, :, :]),
        "sys": dataset["sys"],
        "seq": dataset["seq"]
    }
    print("...done.")

    # Create new session to communicate with the system.
    # Session constructor configures all the necessary devices; in case of the mock,
    # that means to load data from the provided dataset only.
    # A non-mocked session will read a configuration file and create handles
    # to the actual devices that should be available to user.
    print("Creating session.")
    sess = arrus.Session(mock={
        "Us4R:0": dataset
    })

    print("Session created.")

    # Session provides handles to system devices. What devices are available
    # depends on the session configuration file.
    # We will send you an appropriate session configuration file once you receive
    # the us4r-lite hardware.
    # The `Us4R` is an us4r lite device.
    us4r = sess.get_device("/Us4R:0")
    gpu = sess.get_device("/CPU:0")

    # Set HV voltage [0.5*Vpp];
    # maximum value: 90 (can be limited for specific probes in the session
    # configuration file).
    us4r.set_hv_voltage(30)

    # Tx/Rx sequence to perform on the us4r device.
    sequence = LinSequence(
        # Transmit a signal for an aperture centered in element 0, 1, ... 191
        # Note: this should not exceed the number of probe elements.
        tx_aperture_center_element=np.arange(0, 192),
        # The aperture should contain 64 elements.
        tx_aperture_size=64,
        # The beam should be focused on 30 mm depth.
        tx_focus=30e-3,  # [m]
        # Transmit a sine wave with center frequency 4MHz, 2 periods, no inverse.
        pulse=Pulse(center_frequency=4e6, n_periods=2, inverse=False),
        # Receive echo data with aperture centered in elements 0, 1, ..., 191
        # Note: rx_aperture_center_element should have the length as
        # the tx_aperture_center_element vector.
        rx_aperture_center_element=np.arange(0, 192),
        # Record data using 64 elements
        rx_aperture_size=64,
        # Downsampling factor: an integers that divides the output data sampling
        # frequency, i.e. the output sampling frequency is
        # 65e6/n, where n can be 1, 2, ..., 5. One means no downsampling.
        downsampling_factor=1,
        # Pulse repetition interval - the time between successive signal transmits.
        pri=200e-6,
        # Sample range: [start, end) sample
        rx_sample_range=(0, 4096),
        # Linear TGC curve start value.
        tgc_start=14,
        # Linear TGC curve slope.
        tgc_slope=2e2
    )

    # Remember to upload th sequence on the us4r device.
    # The provided buffer will contain acquired RF data.
    # The buffer is a read-only circular queue (only us4r device can write to this
    # buffer).
    # Currently `us4r.upload` is just a nop.
    buffer = us4r.upload(sequence)

    # Output image grid:
    x_grid = np.arange(-50, 50, 0.4)*1e-3
    z_grid = np.arange(0, 60, 0.4)*1e-3

    # Define bmode image reconstruction pipeline.
    # You can find source and docstrings of each step in arrus.utils.imaging
    # module.
    bmode_imaging = Pipeline(
        placement=gpu,
        steps=(
            # Filter the data using bandpass filter,
            # default bandwidth: [0.5*fc, 1.5*fc], where fc is center frequency.
            # Currently FIR filter is available only.
            # The data is filtered along the last axis.
            #
            # input: nd array.
            # output: nd array with the same shape and data type
            BandpassFilter(),
            # Converts to I/Q samples.
            #
            # input: nd array
            # output: nd array with the same shape and dtype=xp.complex64
            QuadratureDemodulation(),
            # Decimate data (CIC filter is also used).
            #
            # input: nd array
            # output: nd array with the last axis `decimation_factor`-times smaller
            Decimation(decimation_factor=4, cic_order=2),
            # Delay and sum; reconstruct scanlines from the provided echo data.
            #
            # input: nd array, shape: n_emissions, n_rx, n_samples
            # output: nd array, shape: n_emissions, n_samples
            RxBeamforming(),
            # Extracts envelope from the RF data.
            #
            # input nd array, dtype=xp.complex64
            # output: nd array, dtype=xp.float32
            EnvelopeDetection(),
            # Transpose the provided image.
            #
            # input: nd array
            # output: nd array with the reversed axes
            Transpose(),
            # Interpolate the RF data to output b-mode image grid.
            #
            # Note! Currently implemented only for CPU.
            #
            # input: nd array, shape: n_samples, n_emissions
            # output: nd array, shape: len(z_grid), len(x_grid)
            ScanConversion(x_grid=x_grid, z_grid=z_grid),
            # Convert to decibel scale.
            LogCompression()
        )
    )

    # Display data with matplotlib
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 7))
    ax.set_xlabel("OX")
    ax.set_ylabel("OZ")
    image_w, image_h = len(x_grid), len(z_grid)
    canvas = plt.imshow(np.zeros((image_w, image_h)), vmin=20, vmax=80, cmap="gray")
    fig.show()

    # Here starts the data acquisition and processing.
    # Starts currently uploaded tx/rx sequence.
    us4r.start()
    # The buffer is now populated with RF data (and some additional metadata).

    # Get data from the buffer, process and display (100 frames).
    for i in range(100):
        # Get data and metadata from the buffer.
        # buffer.pop copies data from the buffer and returns new numpy ndarray.
        # The buffer.pop releases current buffer element.
        # Note: Most likely in the futurewe will add a target 'target_device'
        # parameter which will allow to copy the RF data directly into GPU memory.

        # To avoid data copying the user can use a pair of instructions:
        # - buffer.tail() (returns a numpy array that wraps a pointer to the memory
        #   area with data acquired by the the us4r-lite device)
        # - buffer.release_tail() (notify the us4r-lite device that the
        #   data is not needed anymore and memory area can be reused by the
        #   us4r-lite device for the next acquisitions)
        print("Acquiring data")
        data, metadata = buffer.tail()

        # The metadata structure contains all the information necessary to
        # reconstruct b-mode image from the RF data
        # (e.g. probe's pitch, tx aperture position, etc.).
        # You can find the source and docstrings of the metadata in
        # arrus.metadata module.
        if i == 0:
            # Data acquisition context is constant after starting the us4r.device
            # (you have to stop the device if you want e.g. change some lin sequence
            # parameters), thus metadata.context field
            # is constant;
            #
            # The metadata.context can be saved after acquiring the first frame;
            # then you can ignore this field for consecutive fields.
            print(metadata.context)
            print(metadata.data_description)

        # process
        # gpu_data = cp.asarray(data)
        # We've just copied the data from the us4r-lite buffer, we can release
        # the current buffer element.
        buffer.release_tail()

        # Reconstruct bmode image.
        # Note: metadata.data_description describes data produced at a given step;
        # e.g. metadata.data_description.sampling_frequency can change after
        # `Decimation` operation.
        bmode, metadata = bmode_imaging(data, metadata)
        # display
        canvas.set_data(bmode)
        ax.set_aspect("auto")
        fig.canvas.flush_events()
        plt.draw()
        print(f"Custom metadata: {metadata.custom}")

    # Stop the execution of the tx/rx sequence.
    us4r.stop()


Classical beamforming
---------------------

.. code-block:: python

    seq = LinSequence(
        tx_aperture_center_element=np.arange(7, 182),
        tx_aperture_size=64,
        tx_focus=30e-3,
        pulse=Pulse(center_frequency=5e6, n_periods=3.5, inverse=False),
        rx_aperture_center_element=np.arange(7, 182),
        rx_aperture_size=64,
        rx_sample_range=(0, 4096),
        pri=100e-6,
        downsampling_factor=1,
        tgc_start=14,
        tgc_slope=2e2,
        speed_of_sound=1490)

    bmode_imaging = Pipeline(
         placement=gpu,
         steps=(
                BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4, cic_order=2),
                RxBeamforming(),
                EnvelopeDetection(),
                Transpose(),
                ScanConversion(x_grid=x_grid, z_grid=z_grid),
                LogCompression()))

    # Here starts communication with the device.
    session = arrus.session.Session("cfg.prototxt")

    n = 100

    us4r = session.get_device("/Us4R:0")
    gpu = session.get_device("/GPU:0")

    # Set the pipeline to be executed on the GPU
    bmode_imaging.set_placement(gpu)
    # Set initial voltage on the us4r-lite device.
    us4r.set_hv_voltage(30)
    # Upload sequence on the us4r-lite device.
    buffer = us4r.upload(seq, mode="sync")

    # Start the device.
    us4r.start()
    times = []
    arrus.logging.log(arrus.logging.INFO, f"Running {n} iterations.")
    for i in range(n):
        start = time.time()
        data, metadata = buffer.tail()
        if action_func is not None:
            action_func(i, data, metadata)
        buffer.release_tail()
        times.append(time.time()-start)

    arrus.logging.log(arrus.logging.INFO,
         f"Done, average acquisition + processing time: {np.mean(times)} [s]")

    us4r.stop()

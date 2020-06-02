Communicating with Us4OEM
=========================

In the ``ARRUS`` acquiring echo signal data requires three steps:

1. configure communication session with ``Us4OEM``,
2. define operation (ops) to perform on the device,
3. execute defined operations.

All these three steps are described below.

Configuring session
-------------------

User communicates with the ``Us4OEM`` device in a single ``session``.
Session should be configured before starting it. In particular
it is required to provide:

- description of the system with which the users wants to communicate,
- device initialization parameters.

Currently the system description can be provided as an instance of
:class:`arrus.CustomUs4RCfg`.

.. code-block:: python

    # us4R-lite configuration
    system_cfg = CustomUs4RCfg(
        n_us4oems=2,
        is_hv256=True
    )

``Us4OEM`` initialization parameters should be set using :class:`arrus.Us4OEMCfg`
class, e.g:

.. code-block:: python

    us4oem_cfg = Us4OEMCfg(
        channel_mapping="esaote",
        active_channel_groups=[1]*16,
        dtgc=0,
        active_termination=200,
        log_transfer_time=True
    )

Finally, you can create a :class:`arrus.SessionCfg`:

.. code-block:: python

    session_cfg = SessionCfg(
        system=system_cfg,
        devices={
            "Us4OEM:0": us4oem_cfg,
        }
    )


Defining operations to perform
------------------------------

In ARRUS, user defines **operations** that should be executed on a particular
**device**.

Currently ``Us4OEM`` modules implements :class:`arrus.ops.TxRx` operation,
that is, a single transmit and echo signal reception. The result of this
operation is stored directly in the module's DDR memory, and then is transferred
to the PC for further processing. A single ``TxRx`` allows to transmit a signal
impulse using at most 128 channels and to receive echo data using at most 32
channels.

A single ``TxRx`` operation is limited by module's maximum number of Rx channels.
In most cases user will want to execute a :class:`arrus.ops.Sequence` of
``TxRx`` operations. A single sequence executes given collection of ``TxRx``
operations one by one, stores all acquired data in module's DDR memory, then
transfers them to the computer's memory. For example, a ``Sequence`` of 4
``TxRx`` operations with a shifted Rx aperture (stride 32) allows to acquire
data using 128 Rx channels:

.. code-block:: python

    operations = []
    for i in range(4):
        tx = Tx(excitation=SineWave(frequency=8.125e6, n_periods=1.5,
                                    inverse=False),
                aperture=RegionBasedAperture(origin=0, size=128),
                pri=200e-6)
        rx = Rx(n_samples=8192,
                aperture=RegionBasedAperture(origin=i*32, size=32))
        tx_rx = TxRx(tx, rx)
        operations.append(tx_rx)
    tx_rx_sequence = Sequence(operations)

In real-time imaging user probably would like to execute a given operation
in a loop, until the system is explicitly stopped. In this case
:class:`arrus.ops.Loop` should be used. This operations repeats given
``Sequence`` of operations until the provided ``callback`` function returns
``False``. The callback function should take one input: a numpy ``ndarray``
with the acquire echo data.

.. code-block:: python

    def callback(data):
        print("New data!")
        return True

    sequence_loop = Loop(tx_rx_sequence)


Running operation
-----------------

Operations are performed durring a communication :class:`arrus.Session` with
the device.

In particular, if the user would like to run previously defined sequence of
4 ``TxRx`` operations:

.. code-block:: python

    with arrus.Session(cfg=session_cfg) as sess:
        us4oem = sess.get_device("/Us4OEM:0")
        data = sess.run(tx_rx_sequence, feed_dict={'device': us4oem))

When the context ends, session is closed and all running devices will stop.

A parameter ``feed_dict`` fills operation placeholders with specific values.
An example of such placeholder is a ``device``, on which the operation should be
executed. ``Loop`` operation requires an additional feed value, a ``callback``
function, that should be called when the data acquisiton is finished:

.. code-block:: python

    with arrus.Session(cfg=session_cfg) as sess:
        us4oem = sess.get_device("/Us4OEM:0")
        sess.run(sequence_loop, feed_dict={'device': us4oem),
                                           'callback': callback)


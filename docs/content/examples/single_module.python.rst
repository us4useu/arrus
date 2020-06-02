Communicating with Us4OEM
=========================

Acquiring echo signal data requires three steps:

1. configure a `session` with Us4OEM device,
2. define operations (ops) that should be executed by the device,
3. execute operations within the session.

All these three steps are described below.

Configuring session
-------------------

User communicates with the Us4OEM device in a single `session`.
Session is an abstract object that represents connection between client's
programming interface and the device. The session should be configured before
starting. In particular it is required to provide:

- description of the system to which the user wants to connect to,
- Us4OEM device initialization parameters.

System description should be provided as an instance of
:class:`arrus.CustomUs4RCfg` class.

.. code-block:: python

    # US4R-LITE CONFIGURATION
    system_cfg = CustomUs4RCfg(
        n_us4oems=2,
        is_hv256=True
    )

Us4OEM initialization parameters should be set using :class:`arrus.Us4OEMCfg`
class, e.g:

.. code-block:: python

    us4oem_cfg = Us4OEMCfg(
        channel_mapping="esaote",
        active_channel_groups=[1]*16,
        dtgc=0,
        active_termination=200,
        log_transfer_time=True
    )

Finally, you can prepare a :class:`arrus.SessionCfg`:

.. code-block:: python

    session_cfg = SessionCfg(
        system=system_cfg,
        devices={
            "Us4OEM:0": us4oem_cfg,
        }
    )


Defining operations to perform
------------------------------

In ARRUS, user defines **operations** that will be executed on a particular
**device**.

Us4OEM modules implements :class:`arrus.ops.TxRx` operation,
that is, a single transmit and echo signal reception. The result of this
operation is stored directly in the module's DDR memory, and then is transferred
to the PC for further processing. A single ``TxRx`` allows to transmit a signal
impulse using at most 128 channels and to receive echo data using at most 32
channels.

A single ``TxRx`` operation is limited by module's maximum number of Rx channels.
In most cases a :class:`arrus.ops.Sequence` of ``TxRx`` operations will be
desired. A single ``Sequence`` allows to execute a given collection of
``TxRx`` operations, store all acquired data in module's DDR memory,
then transfer it to the computer's memory. For example, a ``Sequence`` of 4
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
:class:`arrus.ops.Loop` is advised to be used. This operations repeats given
``Sequence`` of ``TxRx`` ops, until the loop is explicitly stopped.
After each execution of the ``Sequence``, a ``callback`` function is called
with the rf data provided as an input. If the acquisition should be continued,
the ``callback`` function should return ``True``, ``False`` otherwise.

.. code-block:: python

    def callback(data):
        print("New data!")
        return True

    sequence_loop = Loop(tx_rx_sequence)


Running operation
-----------------

Operations can be executed within a :class:`arrus.Session`.

In particular, to run the sequence of 4 ``TxRx`` operations:

.. code-block:: python

    with arrus.Session(cfg=session_cfg) as sess:
        us4oem = sess.get_device("/Us4OEM:0")
        data = sess.run(tx_rx_sequence, feed_dict={'device': us4oem})

Please note that ``Session`` is a `python context manager class` with the
following semantic: when the context (an indented block of code) ends, all
running devices are stopped and the session is closed.

A parameter ``feed_dict`` allows to fill the executed operation placeholders
with specific values. An example of such placeholder is a ``device``, on which
the operation should be executed. ``Loop`` operation requires an additional
feed value, a ``callback`` function, that should be called when the data
acquisition is finished.

.. code-block:: python

    with arrus.Session(cfg=session_cfg) as sess:
        us4oem = sess.get_device("/Us4OEM:0")
        sess.run(sequence_loop, feed_dict={'device': us4oem,
                                           'callback': callback}


Examples
--------

Following examples are available in ``python\examples\us4oem`` directory:

- ``us4oem_x1_pwi_single.py``: using Us4OEM to transmit a single plane \
  wave and acquire echo data.
- ``us4oem_x1_sta_single.py``: using Us4OEM to perform a single STA sequence \
- ``us4oem_x1_sta_multiple.py``: using Us4OEM to perform STA sequence multiple\
  times; saves acquired RF data to ``numpy`` file with given frequency.
- ``us4oem_x1_sta_old_api.py``: an example using the old, legacy API.

All examples require ``matplotlib`` package to be installed.


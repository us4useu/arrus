==========
User Guide
==========

This section describes how users can acquire data using ultrasound hardware.
Users can communicate with the device via a communication `session` object.
During the session it is possible to upload and run `operations`.

Configuring session
===================

.. note::

    The below sections contains details on how to configure the
    provided hardware. You can skip to :ref:`running_example`
    if you already have a session configuration file prepared by e.g.
    us4us developers, and you don't need to change any device-related
    parameters.


Session configuration file
--------------------------

A session configuration file consists of device settings valid for a given
session.
Currently, the configuration file can be written in .prototxt file
(a protobuf i/o format readable for humans).
Sample configuration files are available `here <https://github.com/us4useu/arrus/tree/develop/arrus/core/io/test-data>`_.

Currently only us4R device can be configured using session configuration file.

us4R
````

To use the us4R system in a particular session, create a field ``us4r`` in the
session configuration file.

::

    us4r: {
      # here goes us4r configuration spec.
      hv: {
        # ...
      }
      probe: {
        # ...
      }
      adapter: {
        # ...
      }
    }

The us4R device typically includes a high voltage supplier,
which can be configured by providing the ``hv`` field. For now, the us4R device
in the arrus package supports only one type of HV device:

::

    hv: {
      model_id {
        manufacturer: "us4us"
        name: "hv256"
      }
    }

To turn off the high voltage supplier, skip the ``hv`` field.

To configure us4r’s signal transmission/data reception, it is essential to
specify settings of the probe and probe adapter used in the system.

Specify the settings of the probe and probe adapter
'''''''''''''''''''''''''''''''''''''''''''''''''''

Examples:

- `use predefined probe and adapter <https://github.com/us4useu/arrus/blob/develop/arrus/core/io/test-data/us4r.prototxt>`_
- `create custom probe and adapter <https://github.com/us4useu/arrus/blob/develop/arrus/core/io/test-data/custom_us4r.prototxt>`_

Probe Model
...........

The user can specify which probe model they are currently using in one of the
following ways:

1. describe probe model by the providing ``probe`` field, e.g.:

::

    probe: {
      id: {
        manufacturer: "acme"
        name: "my_custom_probe"
      }
      n_elements: 64,
      curvature_radius: 50e-3,
      pitch: 0.21e-3,
      tx_frequency_range: {
        begin: 1e6,
        end: 40e6
      },
      voltage_range: {
        begin: 0,
        end: 100
      }
    }

The following ``probe`` attributes can be specified:

- ``id``: a unique probe model id — a pair: ``(manufacturer, name)``,
- ``n_elements``: number of probe elements,
- ``pitch``: distance between two adjacent probe elements [m],
- ``curvature_radius``: radius of probe’s curvature; when omitted and n_elements is a scalar, a linear probe type is assumed [m],
- ``tx_frequency_range``: acceptable range of center frequencies for this probe [min, max] (a closed interval) [Hz],
- ``voltage_range``: range of acceptable voltage values, 0.5*Vpp.


2. specify probe model by providing ``probe_id``:

::

    probe_id: {
      manufacturer: "esaote",
      name: "sl1543"
    }

If the latter method is used, the probe model description will be searched
in the dictionary file.

When no dictionary file is provided, the
:ref:`default-dictionary` will be assumed.


Probe-to-adapter connection
...........................

The ``probe_to_adapter_connection`` field specifies how the ``probe`` elements
map to the ``adapter`` channels.

There are several ways to specify this mapping:

- ``channel_mapping`` - a list of adapter channels to which the subsequent probe channels should be assigned, i.e. ``channel_mapping[i]`` is the adapter’s channel to be assigned to probe channel ``i``
- ``channel_mapping_ranges`` - a list of adapter channel regions to which the subsequent probe channels should be assigned.

See `here <https://github.com/us4useu/arrus/blob/develop/arrus/core/io/test-data/custom_us4r.prototxt>`_
for an example usage of ``probe_to_adapter_connection`` field.

Note:
This field is required only when a custom probe and adapter are specified in
the session configuration file (i.e. ``probe`` and ``adapter`` fields).
When the ``probe_id`` or ``adapter_id`` are provided and the connection between
them is already defined, this field can be omitted — the arrus package will
try to determine the probe-adapter mapping based on the dictionary file.
When ``probe_to_adapter_connection`` is still given, it will overwrite
the settings from the dictionary file.

Rx Settings
...........

The user can specify the default data reception settings to be set on all
system modules. To do this, add an `rx_settings` with the following attributes:

- ``dtgc_attenuation``: digital time gain compensation to apply (given as attenuation value to apply). Available values: 0, 6, 12, 18, 24, 30, 36, 42 [dB]. Optional, no value means turn off DTGC.
- ``pga_gain``: a gain to apply on a programmable gain amplifier. Available values: 24, 30 [dB]
- ``lna_gain``: a gain to apply on a low-noise amplifier. Available values:  12, 18, 24 [dB]
- ``tgc_samples``: a list of tgc curve samples to apply [dB]. Optional, no value/empty list means turn off TGC
- ``lpf_cutoff``: low-pass filter cut-off frequency, available values: 10000000, 15000000, 20000000, 30000000, 35000000, 50000000 [Hz]
- ``active_termination`` active termination to apply, available values: 50, 100, 200, 400. Optional, no value means turn off active termination.

Channel masks
.............

To turn off specific channels of the us4R system (i.e. the probe elements),
add both of the following fields to the `us4r` settings:

- ``channels_mask``: a list of system channels that should always be disabled
- ``us4oem_channels_mask``: a list of channel masks to apply on each us4OEM module

In order to minimize the risk of including channels that should be turned off,
for example by changing adapter model by mistake
(e.g. using esaote2 adapter mapping when actually esaote3 is installed),
it is necessary to specify the fields:
`channels_mask` and ``us4oem_channels_mask``. If these two mappings do not
match, an error will be reported at the device configuration stage.


Dictionary
----------

It is possible to specify a dictionary of probe models and adapters that are
supported by the us4R system. To do this, add the ``dictionary_file`` field
to the configuration file:

::

    dictionary_file: "dictionary.prototxt"

Currently, the ``dictionary.prototxt`` file will be searched in the same
directory where session settings is located.

When no dictionary file is provided, the
:ref:`default_dictionary` is assumed.

An example dictionary is available here:
https://github.com/us4useu/arrus/blob/develop/arrus/core/io/test-data/dictionary.prototxt

The dictionary file contains a description of ultrasound probes and adapters
that are supported by the us4R device. The file consists of the  following fields:

::

    probe_adapter_models: [
      {
        # probe adapter description, the same as described for us4r.adapter field
      },
      {
        # probe adapter description...
      }
    ]

    probe_models: [
      {
        # probe model description, the same as described for us4r.probe field
      },
      {
        # probe model description...
      }
    ]

    probe_to_adapter_connections: [
      {
        # probe to adapter connection, the same as described for us4r.probe_to_adapter_connection field
      },
      {
        # probe to adapter connection...
      }

    ]

.. _default-dictionary:

Default dictionary
``````````````````

Arrus package already contains a dictionary files of probes and adapters that
were tested on us4r devices.
To use the default dictionary, omit providing ``dictionary_file`` field in your
session configuration file.

Currently, the default dictionary contains definitions of the following probes:

- esaote:

  - probes: ``sl1543``, ``al2442``, ``sp2430``
  - adpaters: ``esoate``, ``esaote2``, ``esaote3``

- als:

  - probes: ``l14-6a``
  - adapters: ``esaote2``, ``esaote3``

- apex:

  - probes: ``tl094``
  - adapters: ``esaote2``, ``esaote3``

- ultrasonix:

  - probes: ``l14-5/38``
  - adapters: ``ultrasonix``

- olympus:

  - probes: ``5L128``
  - adapters: ``esaote3``

- ATL/Philips:

  - probes: ``l7-4``
  - adapters: ``atl/philips``

- custom Vermon:

  - probes: ``la/20/128``
  - adapters: ``atl/philips``

.. _running_example:

Example
=======

.. code-block:: cpp

    #include <iostream>
    #include <thread>
    #include <condition_variable>

    #include <arrus/core/api/arrus.h>

    int main() noexcept {
        using namespace ::arrus::session;
        using namespace ::arrus::devices;
        using namespace ::arrus::ops::us4r;
        using namespace ::arrus::framework;
        try {
            // Read session configuration from the file.
            auto settings = ::arrus::io::readSessionSettings(
                    R"(C:\Users\Public\us4r.prototxt)");
            // Create new session.
            auto session = ::arrus::session::createSession(settings);

            // Get Us4R device handle.
            auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");

            // Tx/Rx sequence:
            // Common Tx parameters:
            ::arrus::BitMask rxAperture(192, true);
            Pulse pulse(4e6, 2, false);

            // Common Rx parameters:
            std::vector<float> delays(192, 0.0f);
            arrus::BitMask txAperture(192, true);
            float pri = 200e-6f;
            ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 2048};
            std::vector<TxRx> txrxs;
            for(int i = 0; i < 10; ++i) {
                txrxs.emplace_back(Tx(txAperture, delays, pulse),
                                   Rx(txAperture, sampleRange),
                                   200e-6f);
            }
            TxRxSequence seq(txrxs, {}, 500e-3f);

            // Define RF channel data output buffer.
            DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};
            // Define scheme to execute.
            Scheme scheme(seq, 2, outputBuffer, Scheme::WorkMode::ASYNC);

            // Upload the scheme.
            auto result = session->upload(scheme);
            // Set HV voltage.
            us4r->setVoltage(10);

            // Create "on new data" callback function.
            // In this example, the callback function counts the number of frames
            // that currently occurred and stops the session when a 10th frame is
            // acquired.
            std::condition_variable cv;
            using namespace std::chrono_literals;
            OnNewDataCallback callback = [&, i = 0](const BufferElement::SharedHandle &ptr) mutable {
                try {
                    std::cout << "Iteration: " << i << ", data: " << std::endl;
                    std::cout << "- memory ptr: " << std::hex
                                               << ptr->getData().get<short>()
                                               << std::dec << std::endl;
                    std::cout << "- size: " << ptr->getSize() << std::endl;
                    std::cout << "- shape: (" << ptr->getData().getShape()[0] <<
                                         ", " << ptr->getData().getShape()[1] <<
                                         ")" << std::endl;

                    // Stop the system after 10-th frame.
                    if(i == 9) {
                        cv.notify_one();
                    }
                    ptr->release();
                    ++i;
                } catch(const std::exception &e) {
                    std::cout << "Exception: " << e.what() << std::endl;
                    cv.notify_all();
                } catch (...) {
                    std::cout << "Unrecognized exception" << std::endl;
                    cv.notify_all();
                }
            };

            // Create callback to be called when overflow occurs.
            OnOverflowCallback overflowCallback = [&] () {
                std::cout << "Data overflow occurred!" << std::endl;
                cv.notify_one();
            };

            // Register callbacks in the data buffer.
            auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
            buffer->registerOnNewDataCallback(callback);
            buffer->registerOnOverflowCallback(overflowCallback);

            // Start the scheme.
            session->startScheme();
            // At this point, data acquisition is started
            // (the occurrence of new data is signaled by the callback function).

            // Wait for callback to signal that the we hit 10-th iteration.
            std::mutex mutex;
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock);

            // Stop the system.
            session->stopScheme();

        } catch(const std::exception &e) {
            std::cerr << e.what() << std::endl;
            return -1;
        }

        return 0;
    }



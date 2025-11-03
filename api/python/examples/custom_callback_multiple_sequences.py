"""
This example shows how to handle mutliple I/Q arrays from multiple sequences
using custom callbacks (i.e. without ARRUS processing).
"""

import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import queue
import numpy as np
import arrus.ops.tgc
import arrus.medium
from collections import deque
import pickle
import time
import scipy.signal
import arrus.ops.us4r

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    Tx,
    Rx,
    TxRx,
    TxRxSequence,
    DataBufferSpec
)
from arrus.ops.imaging import LinSequence
from arrus.utils.imaging import *
from arrus.utils.gui import (
    Display2D, View2D, Layer2D
)

arrus.set_clog_level(arrus.logging.INFO)


def callback(element, metadatas):
    # NOTE: the order of arrays corresponds to the order of TX/RX sequences in the Scheme.tx_rx_sequence field.

    for i, array in enumerate(element.arrays):
        print(f"Sequence:{i} output shape: {array.shape}")
        start_sample, end_sample = metadatas[i].context.sequence.rx_sample_range
        n_samples = end_sample-start_sample
        array = array.reshape(-1, n_samples, 2, 32)
        
        # Frame metadata.
        fcm = metadatas[i].data_description.custom["frame_channel_mapping"]
        # Read metadata from the given OEM (here we are reading OEM:0 frame metadata).
        oem0_n_frames = fcm.n_frames[0]
        print(oem0_n_frames)
        frame_metadata = array[:oem0_n_frames, 0, 0, :]
        timestamps = frame_metadata.view(np.int8)[:, 8:16].copy().view(np.uint64).copy()/65e6
        timestamps = timestamps.squeeze()

        trigger_counters = frame_metadata.view(np.int8)[:, 0:8].copy().view(np.uint64).copy()
        trigger_counters = trigger_counters.squeeze()

        print(f"Sequence:{i} timestamps ({len(timestamps)} values): {timestamps}")
        print(f"Sequence:{i} trigger counters ({len(trigger_counters)} values): {trigger_counters}")

    element.release()


def main():
    # Here starts communication with the device.
    medium = arrus.medium.Medium(name="tissue", speed_of_sound=1540)
    with arrus.Session("us4r.prototxt", medium=medium) as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(10)

        n_elements = us4r.get_probe_model().n_elements

        tx_frequency = 11e6
    
        # 64 TX/RXs
        sequence0 = LinSequence(
            tx_aperture_center_element=np.arange(0, 64),
            tx_aperture_size=64,
            tx_focus=10e-3,
            pulse=Pulse(center_frequency=tx_frequency, n_periods=2, inverse=False),
            rx_aperture_center_element=np.arange(0, 64),
            rx_aperture_size=64,
            rx_sample_range=(0, 1024),
            pri=200e-6,
            downsampling_factor=1,
            speed_of_sound=1450,
            sri=200e-3,
            name="Sequence:0"
        )
        # 32 TX/RXs
        sequence1 = LinSequence(
            tx_aperture_center_element=np.arange(64, 96),
            tx_aperture_size=64,
            tx_focus=10e-3,
            pulse=Pulse(center_frequency=tx_frequency, n_periods=2, inverse=False),
            rx_aperture_center_element=np.arange(64, 96),
            rx_aperture_size=64,
            rx_sample_range=(0, 1024),
            pri=200e-6,
            downsampling_factor=1,
            speed_of_sound=1450,
            sri=200e-3,
            name="Sequence:1"
        )
        
        decimation_factor = 4

        if np.modf(decimation_factor)[0] == 0.25 or np.modf(decimation_factor)[0] == 0.75:
            filter_order = np.uint32(decimation_factor*64)
        elif np.modf(decimation_factor)[0] == 0.5:
            filter_order = np.uint32(decimation_factor*32)
        else:
            filter_order = np.uint32(decimation_factor*16)

        cutoff = tx_frequency
        fs = us4r.sampling_frequency
        coeffs = scipy.signal.firwin(filter_order, cutoff, fs=fs)
        coeffs = coeffs[filter_order//2:]
        digital_down_conversion = arrus.ops.us4r.DigitalDownConversion(
            demodulation_frequency=tx_frequency,
            decimation_factor=decimation_factor,
            fir_coefficients=coeffs)

        scheme = Scheme(
            tx_rx_sequence=(sequence0, sequence1),
            rx_buffer_size=11,
            output_buffer=DataBufferSpec(type="FIFO", n_elements=11),
            digital_down_conversion=digital_down_conversion,
            work_mode="SYNC",
        )

        buffer, metadatas = sess.upload(scheme)
        us4r.set_tgc(arrus.ops.tgc.LinearTgc(start=14, slope=2e2))
        
        print("When using multiple TX/RX sequences, we get list of metadatas. The order of metadata corresponds to the order of sequences in the Scheme.tx_rx_sequence tuple.")
        print([m.context.sequence.name for m in metadatas])
        print("The same order applies to I/Q data arrays (see the definition of the callback function for more details)")
        buffer.append_on_new_data_callback(lambda element: callback(element, metadatas))
        sess.start_scheme()
        time.sleep(10)
        sess.stop_scheme()

    print("Stopping the example.")


if __name__ == "__main__":
    main()


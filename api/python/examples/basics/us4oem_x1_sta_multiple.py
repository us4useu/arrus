import time
import matplotlib.pyplot as plt
import numpy as np
import arrus
import itertools

from arrus.ops import Tx, Rx, TxRx, Sequence, SetHVVoltage, Loop
from arrus import SineWave, SingleElementAperture, RegionBasedAperture

from arrus.system import CustomUs4RCfg
from arrus.devices.us4oem import Us4OEMCfg
from arrus.session import SessionCfg


def callback(rf):
    print("Callback")


def main():
    # Prepare system description.
    system_cfg = CustomUs4RCfg(
        n_us4oems=2,
        is_hv256=True
    )
    # Prepare Us4OEM initial configuration.
    us4oem_cfg = Us4OEMCfg(
        channel_mapping="esaote",
        active_channel_groups=[1]*16,
        dtgc=0,
        active_termination=200,
        log_transfer_time=True
    )

    # Define TX/RX sequence.
    n_firings_per_frame = 4
    n_frames = 128
    n_samples = 8*1024

    def get_full_rx_aperture(element_number):
        operations = []
        for i in range(n_firings_per_frame):
            tx = Tx(excitation=SineWave(frequency=8.125e6, n_periods=1.5,
                                        inverse=False),
                    aperture=RegionBasedAperture(origin=element_number, size=1),
                    pri=200e-6)
            rx = Rx(n_samples=n_samples,
                    fs_divider=1,
                    aperture=RegionBasedAperture(i*32, 32),
                    rx_time=160e-6,
                    rx_delay=5e-6)
            txrx = TxRx(tx, rx)
            operations.append(txrx)
        return operations

    tx_rx_sequence = Sequence(list(itertools.chain(*[
        get_full_rx_aperture(channel)
        for channel in range(n_frames)
    ])))

    sequence_loop = Loop(tx_rx_sequence)

    # Execute the sequence in the session.
    session_cfg = SessionCfg(
        system=system_cfg,
        devices={
            "Us4OEM:0": us4oem_cfg,
        }
    )
    with arrus.Session(cfg=session_cfg) as sess:
        # Enable high voltage supplier.
        hv256 = sess.get_device("/HV256")
        us4oem = sess.get_device("/Us4OEM:0")

        sess.run(SetHVVoltage(50), feed_dict={"device": hv256})

        sess.run(sequence_loop, feed_dict={"device": us4oem,
                                           "callback": callback})
        input("Waiting for input data.")


if __name__ == "__main__":
    main()

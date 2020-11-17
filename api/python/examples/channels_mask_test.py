import numpy as np
import argparse

import arrus
from arrus.ops.imaging import LinSequence
from arrus.ops.us4r import Pulse
import arrus.utils.us4r
import arrus.logging
from arrus.logging import (TRACE, DEBUG, INFO, ERROR)


def check_channels_mask(rf, channel_mask, threshold):
    """
    This function is for test element masking.
    The probe transmit and receive using single element and low voltage,
        and check if there is a signal from transmission in recorded data,
        and if in neighbouring channels signal is high.
    If not, the element is treated as masked.
    This procedure is repeated for all elements.
    Non-zero samples in masked channels or
        too high signals in neigbouring channels raise warnings.
    The test is ok when no warnings shows in the command line.
    (The threshold 500 was ok on tests with mabprobe).
    """
    # skip first n samples
    n_skipped = 10
    n_frames, n_samples, n_channels = rf.shape
    mid = int(np.ceil(n_channels/2)-1)
    stripped_rf = rf[:, n_skipped:, :]

    # mid is a channel number from subaperture corresponding to tested channel

    print(channel_mask)

    for i, channel in enumerate(channel_mask):
        arrus.logging.log(INFO, f"Checking channel {channel}")
        nonmasked = False

        # All samples (also the first 10 samples) should be smaller than the
        # given threshold
        # We try to detect a peak that occurs in the first of couple samples.
        masked_line_max = np.max(rf[i, :, mid])
        if masked_line_max >= threshold:
            arrus.logging.log(ERROR,
                        f"Too high signal ({masked_line_max}) detected in the "
                        f"masked channel.")
            nonmasked = True

        # check if neighboring elements did not receive high signal
        # (ommit first couple of samples which usually contains a large peak)
        mx = np.amax(np.absolute(stripped_rf[i, :, :]))
        if mx >= threshold:
            arrus.logging.log(ERROR,
                        f"Too high signal ({mx}) detected in one of "
                        f"neighboring channels of channel #{channel}.")
            nonmasked = True

        if nonmasked:
            arrus.logging.log(ERROR, f"The channel {channel} is not masked!")
        else:
            arrus.logging.log(INFO, f"The channel {channel} is masked.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="The script checks if the given channels are actually "
                    "masked.")

    parser.add_argument("--cfg", dest="cfg", help="Path to configuration file.",
                        required=True)
    parser.add_argument("--channels_off", dest="channels_off",
                        help="A list of channels that are expected to be "
                             "turned off.",
                        type=int, nargs="+")
    parser.add_argument("--threshold", dest="threshold",
                        help="A sample value threshold that determines if the "
                             "given channel is turned off.",
                        required=False, type=float, default=550)
    parser.add_argument("--dump_file", dest="dump_file",
                        help="A file to which the examined data should "
                             "be saved.",
                        required=False, default=None)

    arrus.set_clog_level(arrus.logging.TRACE)
    arrus.add_log_file("channels_mask_test.log", arrus.logging.TRACE)

    args = parser.parse_args()
    cfg_path = args.cfg
    expected_channels_off = args.channels_off
    threshold = args.threshold
    dump_file = args.dump_file

    seq = LinSequence(
        tx_aperture_center_element=np.array(expected_channels_off),
        tx_aperture_size=1,
        tx_focus=20e-3,
        pulse=Pulse(center_frequency=5e6, n_periods=1, inverse=False),
        rx_aperture_center_element=np.array(expected_channels_off),
        rx_aperture_size=32,
        rx_sample_range=(0, 256),
        pri=2000e-6,
        downsampling_factor=1,
        tgc_start=14,
        tgc_slope=0,
        speed_of_sound=1490)

    session = arrus.Session(cfg_path)

    us4r = session.get_device("/Us4R:0")
    us4r.set_hv_voltage(1)
    buffer = us4r.upload(seq, host_buffer_size=2)

    us4r.start()
    data, metadata = buffer.tail()
    remap_step = arrus.utils.us4r.RemapToLogicalOrder()
    remap_step.set_pkgs(num_pkg=np)
    remapped_data, metadata = remap_step(data, metadata)
    print("Calling channels mask check.")
    check_channels_mask(rf=remapped_data,
                        channel_mask=expected_channels_off,
                        threshold=threshold)
    print("Atfter channels mask check.")
    buffer.release_tail()
    if dump_file is not None:
        arrus.logging.log(INFO, f"Saving data to {dump_file}")
        np.save(f"{dump_file}.npy", remapped_data)
        np.save(f"{dump_file}_raw.npy", data)
        np.save(f"{dump_file}_fcm_frames.npy",
                metadata.data_description.custom["frame_channel_mapping"].frames)
        np.save(f"{dump_file}_fcm_channels.npy",
                metadata.data_description.custom["frame_channel_mapping"].channels)
    us4r.stop()


import dataclasses

import arrus.metadata
import arrus.exceptions


@dataclasses.dataclass
class Transfer:
    src_frame: int
    src_range: tuple
    dst_frame: int
    dst_range: tuple


def group_transfers(frame_channel_mapping):
    result = []
    frame_mapping = frame_channel_mapping.frames
    channel_mapping = frame_channel_mapping.channels

    if frame_mapping.size == 0 or channel_mapping.size == 0:
        raise RuntimeError("Empty frame channel mappings")

    # Number of logical frames
    n_frames, n_channels = channel_mapping.shape

    for dst_frame in range(n_frames):
        current_dst_range = None

        prev_src_frame = None
        prev_src_channel = None
        current_src_frame = None
        current_src_range = None

        for dst_channel in range(n_channels):
            src_frame = frame_mapping[dst_frame, dst_channel]
            src_channel = channel_mapping[dst_frame, dst_channel]

            if src_channel < 0:
                # Omit current channel.
                # Negative src channel means, that the given channel
                # is not available and should be treated as missing.
                continue

            if (prev_src_frame is None  # the first transfer
                    # new src frame
                    or src_frame != prev_src_frame
                    # a gap in current frame
                    or src_channel != prev_src_channel+1):
                # Close current source range
                if current_src_frame is not None:
                    transfer = Transfer(
                        src_frame=current_src_frame,
                        src_range=tuple(current_src_range),
                        dst_frame=dst_frame,
                        dst_range=tuple(current_dst_range)
                    )
                    result.append(transfer)
                # Start a new range
                current_src_frame = src_frame
                # [start, end)
                current_src_range = [src_channel, src_channel + 1]
                current_dst_range = [dst_channel, dst_channel + 1]
            else:
                # Continue current range
                current_src_range[1] = src_channel + 1
                current_dst_range[1] = dst_channel + 1
            prev_src_frame = src_frame
            prev_src_channel = src_channel
        # End a range for current frame.
        current_src_range = int(current_src_range[0]), int(current_src_range[1])
        transfer = Transfer(
            src_frame=int(current_src_frame),
            src_range=tuple(current_src_range),
            dst_frame=dst_frame,
            dst_range=tuple(current_dst_range)
        )
        result.append(transfer)
    return result


def remap(output_array, input_array, transfers):
    input_array = input_array
    for t in transfers:
        dst_l, dst_r = t.dst_range
        src_l, src_r = t.src_range
        output_array[t.dst_frame, :, dst_l:dst_r] = \
            input_array[t.src_frame, :, src_l:src_r]


class RemapToLogicalOrder:
    """
    Remaps the order of the data to logical order defined by the us4r device.

    In particular, the raw ultrasound RF data with shape
    (n_us4oems*n_samples*n_frames, 32) will be reordered to
    (n_frames, n_samples, n_channels).
    """

    def __init__(self, num_pkg=None):
        self._transfers = None
        self._output_buffer = None
        self.xp = num_pkg

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _is_prepared(self):
        return self._transfers is not None and self._output_buffer is not None

    def _prepare(self, data, metadata: arrus.metadata.Metadata):
        xp = self.xp
        # get shape, create an array with given shae
        # create required transfers
        # perform the transfers
        fcm = metadata.data_description.custom["frame_channel_mapping"]
        n_frames, n_channels = fcm.frames.shape
        n_samples_set = {op.rx.get_n_samples()
                         for op in metadata.context.raw_sequence.ops}
        if len(n_samples_set) > 1:
            raise arrus.exceptions.IllegalArgumentError(
                f"Each tx/rx in the sequence should acquire the same number of "
                f"samples (actual: {n_samples_set})")
        n_samples = next(iter(n_samples_set))
        output_shape = (n_frames, n_samples, n_channels)
        self._output_buffer = xp.zeros(shape=output_shape, dtype=xp.int16)
        self._transfers = group_transfers(fcm)
        n_samples_raw, n_channels_raw = data.shape
        self._input_shape = (n_samples_raw//n_samples, n_samples,
                             n_channels_raw)

    def __call__(self, data, metadata: arrus.metadata.Metadata):
        if not self._is_prepared():
            self._prepare(data, metadata)
        remap(
            output_array=self._output_buffer,
            input_array=data.reshape(self._input_shape),
            transfers=self._transfers)
        return self._output_buffer, metadata


import dataclasses
import numpy as np

import arrus.metadata
import arrus.exceptions


@dataclasses.dataclass
class Transfer:
    src_frame: int
    src_range: tuple
    dst_frame: int
    dst_range: tuple


def get_batch_data(data, metadata, frame_nr):
    batch_size = metadata.data_description.custom["frame_channel_mapping"].batch_size
    n_samples = metadata.context.raw_sequence.get_n_samples()
    if len(n_samples) > 1:
        raise ValueError("This function doesn't support tx/rx sequences with variable number of samples")
    n_samples = next(iter(n_samples))

    # TODO here is an assumption, that each output frame has exactly the same number of samples
    # This might not be the case in the future.
    # Data from the first module.
    firstm_n_scanlines = metadata.custom["frame_metadata_view"].shape[0]
    # Number of scanlines in a single RF frame
    assert firstm_n_scanlines % batch_size == 0, "Incorrect number of the result scanlines and samples."
    firstm_n_scanlines_frame = firstm_n_scanlines // batch_size

    # Number of sample for the first module
    firstm_n_samples_frame = firstm_n_scanlines_frame * n_samples

    first = data[frame_nr*firstm_n_samples_frame:
                 (frame_nr+1)*firstm_n_samples_frame, :]

    # Data from the second module.
    # FIXME: here is an assumption, that there are no rx nops in the sequence
    # This may not be the case in the future
    offset = firstm_n_scanlines * n_samples  # the number of samples
    total_n_scanlines = np.max(metadata.data_description.custom["frame_channel_mapping"].frames+1)*batch_size
    secondm_n_scanlines = total_n_scanlines - firstm_n_scanlines
    assert secondm_n_scanlines % batch_size == 0, "Incorrect number of " \
                                                  "the result scanlines " \
                                                  "and samples."
    assert firstm_n_scanlines == secondm_n_scanlines
    secondm_n_scanlines_frame = secondm_n_scanlines // batch_size
    secondm_n_samples_frame = secondm_n_scanlines_frame * n_samples

    second = data[frame_nr*secondm_n_samples_frame+offset:
                  (frame_nr+1)*secondm_n_samples_frame+offset, :]
    return np.concatenate((first, second), axis=0)


def get_batch_metadata(metadata, frame_nr):
    batch_size = metadata.data_description.custom["frame_channel_mapping"].batch_size
    # Number of scanlines in the first module
    n_scanlines_total = metadata.custom["frame_metadata_view"].shape[0]
    n_samples_in_scanline = n_scanlines_total // batch_size
    frame_metadata_view = metadata.custom["frame_metadata_view"][
                          frame_nr*n_samples_in_scanline:(frame_nr+1)*n_samples_in_scanline]
    new_metadata = arrus.metadata.Metadata(context=metadata.context,
                                           data_desc=metadata.data_description,
                                           custom={"frame_metadata_view" :
                                                       frame_metadata_view})
    return new_metadata


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

    If the batch size was equal 1, the raw ultrasound RF data with shape.
    (n_frames, n_samples, n_channels).
    A single metadata object will be returned.

    If the batch size was > 1, the the raw ultrasound RF data with shape
    (n_us4oems*n_samples*n_frames*n_batches, 32) will be reordered to
    (batch_size, n_frames, n_samples, n_channels). A list of metadata objects
    will be returned.

    """

    def __init__(self, num_pkg=None):
        self._transfers = None
        self._output_buffer = None
        self.xp = num_pkg

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _is_prepared(self):
        return self._transfers is not None and self._output_buffer is not None

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        xp = self.xp
        # get shape, create an array with given shape
        # create required transfers
        # perform the transfers
        fcm = const_metadata.data_description.custom["frame_channel_mapping"]
        n_frames, n_channels = fcm.frames.shape
        batch_size = fcm.batch_size
        n_samples_set = {op.rx.get_n_samples()
                         for op in const_metadata.context.raw_sequence.ops}
        if len(n_samples_set) > 1:
            raise arrus.exceptions.IllegalArgumentError(
                f"Each tx/rx in the sequence should acquire the same number of "
                f"samples (actual: {n_samples_set})")
        n_samples = next(iter(n_samples_set))
        self.output_shape = (n_frames, n_samples, n_channels)
        self._output_buffer = xp.zeros(shape=self.output_shape, dtype=xp.int16)
        self._transfers = group_transfers(fcm)
        n_samples_raw, n_channels_raw = const_metadata.input_shape
        self._input_shape = (n_samples_raw//n_samples, n_samples,
                             n_channels_raw)
        self.batch_size = fcm.batch_size
        return const_metadata.copy(input_shape=self.output_shape)

    def __call__(self, data):
        remap(output_array=self._output_buffer,
            input_array=data.reshape(self._input_shape),
            transfers=self._transfers)
        return self._output_buffer


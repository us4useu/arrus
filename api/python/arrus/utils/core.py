import numpy as np
import arrus.core
import arrus.exceptions
import arrus.devices.probe


def convert_to_core_sequence(seq):
    """
    Converts given tx/rx sequence to arrus.core.TxRxSequence
    TODO this function can be simplified by improving swig core.i mapping

    :param seq: arrus.ops.us4r.TxRxSequence
    :return: arrus.core.TxRxSequence
    """
    core_seq = arrus.core.TxRxVector()
    n_samples = None
    for op in seq.ops:
        tx, rx = op.tx, op.rx
        # TODO validate shape
        # TX
        core_delays = np.zeros(tx.aperture.shape, dtype=np.float32)
        core_delays[tx.aperture] = tx.delays
        core_excitation = arrus.core.Pulse(
            centerFrequency=tx.excitation.center_frequency,
            nPeriods=tx.excitation.n_periods,
            inverse=tx.excitation.inverse
        )
        core_tx = arrus.core.Tx(
            aperture=arrus.core.VectorBool(tx.aperture.tolist()),
            delays=arrus.core.VectorFloat(core_delays.tolist()),
            excitation=core_excitation
        )
        # RX
        core_rx = arrus.core.Rx(
            arrus.core.VectorBool(rx.aperture.tolist()),
            arrus.core.PairUint32(int(rx.sample_range[0]), int(rx.sample_range[1])),
            rx.downsampling_factor,
            arrus.core.PairChannelIdx(int(rx.padding[0]), int(rx.padding[1]))
        )


        core_txrx = arrus.core.TxRx(core_tx, core_rx, op.pri)
        arrus.core.TxRxVectorPushBack(core_seq, core_txrx)

        start_sample, end_sample = rx.sample_range
        if n_samples is None:
            n_samples = end_sample - start_sample
        elif n_samples != end_sample - start_sample:
            raise arrus.exceptions.IllegalArgumentError(
                "Sequences with the constant number of "
                "samples are supported only.")

    sri = -1 if seq.sri is None else seq.sri
    core_seq = arrus.core.TxRxSequence(core_seq, seq.tgc_curve.tolist(), sri)
    return core_seq


def convert_fcm_to_np_arrays(fcm):
    """
    Converts frame channel mapping to a tupple of numpy arrays.

    :param fcm: arrus.core.FrameChannelMapping
    :return: a pair of numpy arrays: fcm_frame, fcm_channel
    """
    fcm_frame = np.zeros(
        (fcm.getNumberOfLogicalFrames(), fcm.getNumberOfLogicalChannels()),
        dtype=np.int16)
    fcm_channel = np.zeros(
        (fcm.getNumberOfLogicalFrames(), fcm.getNumberOfLogicalChannels()),
        dtype=np.int8)
    for frame in range(fcm.getNumberOfLogicalFrames()):
        for channel in range(fcm.getNumberOfLogicalChannels()):
            frame_channel = fcm.getLogical(frame, channel)
            src_frame = frame_channel[0]
            src_channel = frame_channel[1]
            fcm_frame[frame, channel] = src_frame
            fcm_channel[frame, channel] = src_channel
    return fcm_frame, fcm_channel


def convert_to_py_probe_model(core_model):
    n_elements = arrus.core.getNumberOfElements(core_model)
    pitch = arrus.core.getPitch(core_model)
    curvature_radius = core_model.getCurvatureRadius()
    model_id = core_model.getModelId()
    return arrus.devices.probe.ProbeModel(
        model_id=arrus.devices.probe.ProbeModelId(
            manufacturer=model_id.getManufacturer(),
            name=model_id.getName()),
        n_elements=n_elements,
        pitch=pitch,
        curvature_radius=curvature_radius)


def convert_to_core_scheme(scheme):
    seq = scheme.tx_rx_sequence
    rx_buffer_size = scheme.rx_buffer_size
    output_buffer = scheme.output_buffer

    # Convert output buffer to core.DataBufferSpec
    core_buffer_type = {
        "FIFO": arrus.core.DataBufferSpec.Type_FIFO
    }[output_buffer.type]
    data_buffer_spec = arrus.core.DataBufferSpec(core_buffer_type,
                                                 output_buffer.n_elements)

    # Convert sequence to core sequence.
    core_seq = arrus.utils.core.convert_to_core_sequence(seq)

    core_work_mode = {
        "ASYNC": arrus.core.Scheme.WorkMode_ASYNC,
        "HOST": arrus.core.Scheme.WorkMode_HOST
    }[scheme.work_mode]

    return arrus.core.Scheme(core_seq, rx_buffer_size, data_buffer_spec,
                             workMode=core_work_mode)


def convert_from_tuple(core_tuple):
    """
    Converts arrus core tuple to python tuple.
    """
    v = [core_tuple.get(i) for i in range(core_tuple.size())]
    return tuple(v)


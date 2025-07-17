import numpy as np
import arrus.core
import arrus.exceptions
import arrus.devices.probe
from arrus.devices.device import parse_device_id, DeviceId
from typing import Dict, Any, Iterable, Tuple, Union, List
import arrus.ops.us4r

_UINT16_MIN = 0
_UINT16_MAX = 2**16-1


def to_core_device_id(tx_placement):
    core_type = arrus.core.parseToDeviceTypeEnum(tx_placement.device_type.type)
    return arrus.core.DeviceId(core_type, tx_placement.ordinal)


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
        excitation = tx.excitation
        if isinstance(excitation, arrus.ops.us4r.Pulse):
            core_excitation = arrus.core.Pulse(
                centerFrequency=excitation.center_frequency,
                nPeriods=excitation.n_periods,
                inverse=excitation.inverse,
                amplitudeLevel=excitation.amplitude_level
            )
        elif isinstance(excitation, arrus.ops.us4r.Waveform):
            waveformBuilder = arrus.core.WaveformBuilder()
            for segment, n_repeats in zip(excitation.segments, excitation.n_repeats):
                duration = arrus.core.VectorFloat(np.asarray(segment.duration).tolist())
                state = arrus.core.VectorInt8(np.asarray(segment.state).tolist())
                segment = arrus.core.WaveformSegment(duration, state)
                waveformBuilder.add(segment, n_repeats)
            core_excitation = waveformBuilder.build()
        else:
            raise ValueError(f"Unrecognized TX pulse type: {excitation}")
        tx_placement = parse_device_id(tx.placement)
        tx_placement = to_core_device_id(tx_placement)
        core_tx = arrus.core.Tx(
            arrus.core.VectorBool(tx.aperture.tolist()),
            arrus.core.VectorFloat(core_delays.tolist()),
            core_excitation,
            tx_placement
        )
        # RX
        rx_placement = parse_device_id(rx.placement)
        rx_placement = to_core_device_id(rx_placement)
        core_rx = arrus.core.Rx(
            arrus.core.VectorBool(rx.aperture.tolist()),
            arrus.core.PairUint32(int(rx.sample_range[0]), int(rx.sample_range[1])),
            rx.downsampling_factor,
            arrus.core.PairChannelIdx(int(rx.padding[0]), int(rx.padding[1])),
            rx_placement
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
    if seq.n_repeats < _UINT16_MIN or seq.n_repeats > _UINT16_MAX:
        raise arrus.exceptions.IllegalArgumentError(
            f"Parameter n_repeats should be in range "
            f"[{_UINT16_MIN}, {_UINT16_MAX}]"
        )
    core_seq = arrus.core.TxRxSequence(core_seq, seq.tgc_curve.tolist(), sri,
                                       seq.n_repeats)
    return core_seq


def convert_fcm_to_np_arrays(fcm, n_us4oems):
    """
    Converts frame channel mapping to a tupple of numpy arrays.

    :param fcm: arrus.core.FrameChannelMapping
    :return: a pair of numpy arrays: fcm_frame, fcm_channel
    """
    fcm_us4oem = np.zeros(
        (fcm.getNumberOfLogicalFrames(), fcm.getNumberOfLogicalChannels()),
        dtype=np.uint8)
    fcm_frame = np.zeros(
        (fcm.getNumberOfLogicalFrames(), fcm.getNumberOfLogicalChannels()),
        dtype=np.int16)
    fcm_channel = np.zeros(
        (fcm.getNumberOfLogicalFrames(), fcm.getNumberOfLogicalChannels()),
        dtype=np.int8)
    frame_offsets = np.zeros(n_us4oems, dtype=np.uint32)
    for frame in range(fcm.getNumberOfLogicalFrames()):
        for channel in range(fcm.getNumberOfLogicalChannels()):
            frame_channel = fcm.getLogical(frame, channel)
            src_us4oem = frame_channel.getUs4oem()
            src_frame = frame_channel.getFrame()
            src_channel = frame_channel.getChannel()
            fcm_us4oem[frame, channel] = src_us4oem
            fcm_frame[frame, channel] = src_frame
            fcm_channel[frame, channel] = src_channel
    frame_offsets = [fcm.getFirstFrame(i) for i in range(n_us4oems)]
    frame_offsets = np.array(frame_offsets, dtype=np.uint32)
    n_frames = [fcm.getNumberOfFrames(i) for i in range(n_us4oems)]
    n_frames = np.array(n_frames, dtype=np.uint32)
    return fcm_us4oem, fcm_frame, fcm_channel, frame_offsets, n_frames


def convert_to_py_probe_model(core_model):
    n_elements = arrus.core.getNumberOfElements(core_model)
    pitch = arrus.core.getPitch(core_model)
    curvature_radius = core_model.getCurvatureRadius()
    model_id = core_model.getModelId()
    core_fr = core_model.getTxFrequencyRange()
    core_vr = core_model.getVoltageRange()
    tx_frequency_range = (core_fr.start(), core_fr.end())
    voltage_range = (core_vr.start(), core_vr.end())
    if core_model.isLensDefined():
        lens = core_model.getLensOrRaiseException()
        lens = arrus.devices.probe.Lens(
            thickness=lens.getThickness(),
            speed_of_sound=lens.getSpeedOfSound(),
            focus=lens.getFocus()
        )
    else:
        lens = None
    if core_model.isMatchingLayerDefined():
        matching_layer = core_model.getMatchingLayerOrRaiseException()
        matching_layer = arrus.devices.probe.MatchingLayer(
            thickness=matching_layer.getThickness(),
            speed_of_sound=matching_layer.getSpeedOfSound(),
        )
    else:
        matching_layer = None
    return arrus.devices.probe.ProbeModel(
        model_id=arrus.devices.probe.ProbeModelId(
            manufacturer=model_id.getManufacturer(),
            name=model_id.getName()),
        n_elements=n_elements,
        pitch=pitch,
        curvature_radius=curvature_radius,
        tx_frequency_range=tx_frequency_range,
        lens=lens,
        matching_layer=matching_layer,
        voltage_range=voltage_range
    )


def convert_array_to_vector_float(array):
    result = arrus.core.VectorFloat()
    for v in array:
        arrus.core.VectorFloatPushBack(result, float(v))
    return result


def convert_to_core_scheme(scheme):
    builder = arrus.core.SchemeBuilder()
    seqs = scheme.tx_rx_sequence
    if not isinstance(seqs, Iterable):
        seqs = (seqs, )
    rx_buffer_size = scheme.rx_buffer_size
    output_buffer = scheme.output_buffer

    # Convert output buffer to core.DataBufferSpec
    core_buffer_type = {
        "FIFO": arrus.core.DataBufferSpec.Type_FIFO
    }[output_buffer.type]
    data_buffer_spec = arrus.core.DataBufferSpec(core_buffer_type,
                                                 output_buffer.n_elements)
    builder.withRxBufferSize(rx_buffer_size)
    builder.withOutputBufferDefinition(data_buffer_spec)
    # Convert sequence to core sequence.
    for s in seqs:
        core_seq = arrus.utils.core.convert_to_core_sequence(s)
        builder.addSequence(core_seq)

    core_work_mode = {
        "ASYNC": arrus.core.Scheme.WorkMode_ASYNC,
        "SYNC": arrus.core.Scheme.WorkMode_SYNC,
        "HOST": arrus.core.Scheme.WorkMode_HOST,
        "MANUAL": arrus.core.Scheme.WorkMode_MANUAL,
        "MANUAL_OP": arrus.core.Scheme.WorkMode_MANUAL_OP
    }[scheme.work_mode]
    builder.withWorkMode(core_work_mode)
    ddc = scheme.digital_down_conversion
    # TODO constants
    if(scheme.digital_down_conversion is not None):
        ddc = arrus.core.DigitalDownConversion(
            ddc.demodulation_frequency,
            convert_array_to_vector_float(ddc.fir_coefficients),
            ddc.decimation_factor,
            ddc.gain
        )
        builder.withDigitalDownConversion(ddc)
    return builder.build()


def convert_to_test_pattern(test_pattern_str):
    return {
        "OFF": arrus.core.Us4OEM.RxTestPattern_OFF,
        "RAMP": arrus.core.Us4OEM.RxTestPattern_RAMP
    }[test_pattern_str]


def convert_from_tuple(core_tuple):
    """
    Converts arrus core tuple to python tuple.
    """
    v = [core_tuple.get(i) for i in range(core_tuple.size())]
    return tuple(v)


def convert_to_core_parameters(params: Dict[str, Any]):
    builder = arrus.core.ParametersBuilder()
    for k, v in params.items():
        builder.add(k, v)
    return builder.build()


def convert_constants_to_arrus_ndarray(py_constants):
    result = arrus.core.ArrusNdArrayVector()
    for py_const in py_constants:
        value = py_const.value
        if not isinstance(value, np.ndarray):
            # NOTE: assuming scalar!
            value = np.asarray(value)
        value = np.atleast_2d(value).astype(np.float32)
        placement: str = py_const.placement
        placement = placement.strip()
        if placement.startswith("/"):
            placement = placement[1:].strip()
        placement_name, placement_ordinal = placement.split(":")
        placement_name = placement_name.strip()
        placement_ordinal = int(placement_ordinal.strip())
        arrus.core.Arrus2dArrayVectorPushBack(
            result,
            # rows, columns
            value.shape[0], value.shape[1], arrus.core.VectorFloat(value.flatten().tolist()),
            placement_name, placement_ordinal,
            py_const.name
        )
    return result


def convert_to_hv_voltages(values: List[Union[int, Tuple[int, int]]]):
    result = []
    for v in values:
        if isinstance(v, tuple):
            vm, vp = v
            if not isinstance(vm, int) or not isinstance(vp, int):
                raise ValueError("Voltages are expected to be integers")
        elif isinstance(v, int):
            vm, vp = v, v
        else:
            raise ValueError("Voltages are expected to be integers "
                             "or pair of integers.")
        assert_hv_voltage_correct(vm)
        assert_hv_voltage_correct(vp)
        result.append(arrus.core.HVVoltage(vm, vp))
    return arrus.core.VectorHVVoltage(result)


def assert_hv_voltage_correct(value):
    min_v, max_v = 0, 255  # 255 -- max uint8 (expected by C++ API)
    if not (min_v <= value <= max_v):
        raise ValueError("Voltages are expected to be values in range "
                         f"[{min_v}, {max_v}]")

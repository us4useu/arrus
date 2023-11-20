import unittest
import dataclasses
import math
import numpy as np

import arrus.metadata
import arrus.kernels
import arrus.ops.imaging
import arrus.ops.us4r


# ---- Mock classes
# TODO extract device specification to separate classes
@dataclasses.dataclass(frozen=True)
class ProbeModelMock:
    n_elements: int
    pitch: float
    curvature_radius: float
        
    def __post_init__(self):
        element_pos_x, element_pos_z, element_angle = self._compute_element_position()
        super().__setattr__("element_pos_x", element_pos_x)
        super().__setattr__("element_pos_z", element_pos_z)
        super().__setattr__("element_angle", element_angle)

    # TODO move the below functions to some other package
    # (it should be part of image reconstruction implementation)
    def _compute_element_position(self):
        # element position along the surface
        element_position = np.arange(-(self.n_elements - 1) / 2,
                                     self.n_elements / 2)
        element_position = element_position * self.pitch

        if not self.is_convex_array():
            x_pos = element_position
            z_pos = np.zeros((1, self.n_elements))
            angle = np.zeros((1,self.n_elements))
        else:
            angle = element_position / self.curvature_radius
            x_pos = self.curvature_radius * np.sin(angle)
            z_pos = self.curvature_radius * np.cos(angle)
            z_pos = z_pos - np.min(z_pos)
        return x_pos, z_pos, angle

    def is_convex_array(self):
        return not (math.isnan(self.curvature_radius)
                    or self.curvature_radius == 0.0)

    @property
    def model_id(self):
        return "TestModel"


@dataclasses.dataclass(frozen=True)
class ProbeMock:
    model: ProbeModelMock


@dataclasses.dataclass(frozen=True)
class UltrasoundDeviceMock:
    probe: ProbeMock
    sampling_frequency: float


class ArrusTestCase(unittest.TestCase):
    pass


class ArrusImagingTestCase(ArrusTestCase):
    """
    Arrus processing test case.

    :param op: operator to test
    """

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.op = None

    def run_op(self, **kwargs):
        """
        Run TestCase op with given parameters.

        This function assumes, that the `TestCase.op` was already set.

        Currently the below list describes a list of parameters that are
        required. If the parameters `xyz` is None, TestCase.xyz will be used.
        All the parameters not listed below will be passed to the operator constructor.

        Currently GPU implementation can only be tested.

        Currently the function assumes, that the op is an Operation from
        the arrus.utils.imaging module.

        :param data: data to process, numpy ndarray
        :param context: data processing context, including the description of
            the source of the data (ultrasound system, medium, eventually
            the history of processing that was already done)
            Note: the above is the FrameAcquisitionContext
        :return: op output (numpy.ndarray)
        """
        data = self.__get_param_or_field("data", kwargs)
        context = self.__get_param_or_field("context", kwargs)

        # Get arrus.utils.imaging.Operation constructor parameters
        constructor_params = kwargs.copy()
        constructor_params.pop("data", None)
        constructor_params.pop("context", None)

        # Create op instance
        op_class = self.op  # We assume field `op` is the subclass of Operation
        op_instance = op_class(**constructor_params)

        # Set op backend and processing device.
        import cupy as cp
        import cupyx.scipy.ndimage as cupy_scipy_ndimage
        pkgs = dict(num_pkg=cp, filter_pkg=cupy_scipy_ndimage)
        op_instance.set_pkgs(**pkgs)
        data = cp.asarray(data)

        # Prepare and initialize (context)
        const_metadata = arrus.metadata.ConstMetadata(
            context=context,
            # Note: the below are some assumptions, that may change in
            # the future
            data_desc=arrus.metadata.EchoDataDescription(
                sampling_frequency=context.device.sampling_frequency,
                custom={}
            ),
            input_shape=data.shape,
            is_iq_data=data.dtype == cp.complex64,
            dtype=data.dtype
        )
        op_instance.prepare(const_metadata=const_metadata)
        init_data = cp.zeros(data.shape, dtype=data.dtype) + 1000
        op_instance.initialize(init_data)
        # Now run the op for the given data:
        op_result = op_instance(data)
        if isinstance(op_result, cp.ndarray):
            op_result = op_result.get()
        elif not isinstance(op_result, np.ndarray):
            raise ValueError(f"Invalid output result type: {type(op_result)}")
        return op_result

    def get_probe_model_instance(self, **kwargs):
        """
        Note: the function takes ProbeModelMock input parameters.
        """
        return ProbeMock(ProbeModelMock(**kwargs))

    def get_ultrasound_device(self, **kwargs):
        return UltrasoundDeviceMock(**kwargs)

    def get_default_context(self, device=None, sequence=None, medium=None):
        if sequence is None:
            sequence = arrus.ops.imaging.PwiSequence(
                angles=[0.0],
                pulse=arrus.ops.us4r.Pulse(center_frequency=6e6, n_periods=2,
                                           inverse=False),
                rx_sample_range=(0, 2048),
                downsampling_factor=1,
                speed_of_sound=1490,
                pri=100e-6,
                sri=50e-3
            )
        if device is None:
            device = self.get_ultrasound_device(
                probe=self.get_probe_model_instance(
                    n_elements=64,
                    pitch=0.2e-3,
                    curvature_radius=0.0
                ),
                sampling_frequency=65e6
            )
        # if medium is none, keep it None

        raw_sequence = self.__convert_to_raw_sequence(
            device, sequence, medium)

        acquisition_context = arrus.metadata.FrameAcquisitionContext(
            device=device, sequence=sequence, raw_sequence=raw_sequence,
            medium=medium, custom_data={}, constants=[]
        )
        return acquisition_context

    def __get_param_or_field(self, param_name, params):
        if param_name in params:
            return params[param_name]
        else:
            return getattr(self, param_name)

    def __convert_to_raw_sequence(self, device, sequence, medium):
        import arrus.kernels
        import arrus.kernels.kernel
        kernel = arrus.kernels.get_kernel(type(sequence))
        kernel_context = arrus.kernels.kernel.KernelExecutionContext(
            device=device, medium=medium, op=sequence, custom={})
        result = kernel(kernel_context)
        return result.sequence



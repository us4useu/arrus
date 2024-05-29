import numpy as np
import math
import arrus.kernels.tx_rx_sequence
import arrus.ops.imaging
import arrus.medium
from arrus.kernels.kernel import KernelExecutionContext
import arrus.ops.us4r
from arrus.ops.us4r import (
    TxRxSequence, Tx, Rx, TxRx, Pulse, Aperture
)
import unittest
import dataclasses
from arrus.devices.probe import ProbeModelId, ProbeModel


@dataclasses.dataclass(frozen=True)
class ProbeMock:
    model: ProbeModel


@dataclasses.dataclass(frozen=True)
class DeviceMock:
    probe: ProbeMock
    sampling_frequency: float = 65e6
    data_sampling_freuency: float = 65e6

    def get_probe_by_id(self, id):
        return self.probe


@dataclasses.dataclass(frozen=True)
class ContextMock:
    device: DeviceMock
    medium: arrus.medium.MediumDTO
    op: object
    hardware_ddc: object = None
    constants: tuple = ()


class SimpleTxRxSequenceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.default_device = DeviceMock(ProbeMock(
            ProbeModel(model_id=ProbeModelId("a", "a"), pitch=1, n_elements=8,
                       curvature_radius=0.0)))
        self.default_medium = None

    def assert_ok_for_moving_aperture(self, aperture_size,
                                      expected_delays_profile, n_elements,
                                      ops,
                                      expected_tx_elements=None,
                                      expected_rx_elements=None,
                                      center_elements=None):

        op_expected_tx_elements = expected_tx_elements
        op_expected_rx_elements = expected_rx_elements

        if center_elements is None:
            center_elements = np.arange(len(ops))

        for i, op in zip(center_elements, ops):
            tx_aperture = op.tx.aperture
            rx_aperture = op.rx.aperture
            tx_delays = op.tx.delays

            # Aperture
            # e.g. i=0, 64-element aperture -> origin: -31, end: 33
            aperture_origin = i - int(math.ceil(aperture_size/2)) + 1

            aperture_end = aperture_origin + aperture_size
            delays_start = -aperture_origin if aperture_origin < 0 else 0
            delays_end = aperture_size - (
                (aperture_end - n_elements) if aperture_end > n_elements else 0)

            aperture_origin = max(0, aperture_origin)
            aperture_end = min(n_elements, aperture_end)
            expected_delays = expected_delays_profile[delays_start:delays_end]
            # aperture: [origin, end)
            expected_elements = np.arange(aperture_origin, aperture_end)

            if expected_tx_elements is None:
                op_expected_tx_elements = expected_elements

            if expected_rx_elements is None:
                op_expected_rx_elements = expected_elements

            np.testing.assert_equal(np.squeeze(np.argwhere(tx_aperture)),
                                    op_expected_tx_elements)
            np.testing.assert_equal(np.squeeze(np.argwhere(rx_aperture)),
                                    op_expected_rx_elements)
            np.testing.assert_almost_equal(expected_delays, tx_delays,
                                           decimal=12)


class PwiSequenceTest(SimpleTxRxSequenceTest):

    def setUp(self) -> None:
        super().setUp()
        self.default_excitation = arrus.ops.us4r.Pulse(
            center_frequency=1,
            n_periods=1,
            inverse=False)
        self.default_sample_range = (0, 8)
        ops = [
            TxRx(
                tx=Tx(
                    aperture=Aperture(center_element=3.5, size=8),
                    excitation=self.default_excitation,
                    focus=np.inf,
                    angle=angle,
                    speed_of_sound=1,
                ),
                rx=Rx(
                    aperture=Aperture(center_element=3.5, size=8),
                    sample_range=self.default_sample_range,
                    downsampling_factor=1,
                ),
                pri=1,
            )
            for angle in [-np.pi/4, 0.0, np.pi/4]
        ]
        self.default_sequence = TxRxSequence(ops=ops, tgc_curve=[])

    def test_linear_array_three_tx_rxs(self):
        # Given
        device = self.default_device
        n_elements = device.probe.model.n_elements
        pitch = device.probe.model.pitch
        context = ContextMock(device=self.default_device,
                              medium=self.default_medium,
                              op=self.default_sequence)
        result = arrus.kernels.tx_rx_sequence.process_tx_rx_sequence(context)
        sequence = result.sequence
        # Expect
        tx_rxs = sequence.ops
        # should be 3 TX/RXs
        self.assertEqual(len(tx_rxs), 3)
        # TX:
        # - aperture: all device elements
        for tx_rx in tx_rxs:
            self.assertEqual(np.sum(tx_rx.tx.aperture == 1), n_elements)
        # - delays:
        expected_delays_pi_4 = np.arange(0, n_elements)-(n_elements-1)/2
        expected_delays_pi_4 *= pitch
        expected_delays_pi_4 = expected_delays_pi_4/np.sqrt(2) # sin(pi/4)
        min_expected_delay = np.min(expected_delays_pi_4)
        expected_delays_pi_4 = expected_delays_pi_4-np.min(expected_delays_pi_4)
        tx_aperture_center_delay = 0-min_expected_delay

        # angle: -pi/4 [rad]
        np.testing.assert_almost_equal(
            tx_rxs[0].tx.delays,
            np.flip(expected_delays_pi_4),
            decimal=12)
        # angle: 0 [rad]
        np.testing.assert_almost_equal(
            tx_rxs[1].tx.delays,
            np.repeat(tx_aperture_center_delay, len(expected_delays_pi_4)),
            decimal=12)
        # angle: pi/4 [rad]
        np.testing.assert_almost_equal(
            tx_rxs[2].tx.delays,
            expected_delays_pi_4,
            decimal=12)
        # - pulse:
        for tx_rx in tx_rxs:
            self.assertEqual(tx_rx.tx.excitation, self.default_excitation)
        # RX:
        # - aperture: all device elements
        for tx_rx in tx_rxs:
            self.assertEqual(np.sum(tx_rx.rx.aperture == 1), n_elements)
            self.assertEqual(tx_rx.rx.sample_range, self.default_sample_range)
            self.assertEqual(tx_rx.rx.padding, (0, 0))

    def test_reference_compliance_linear_array_ultrasonix_l14_5_38_parameters(self):
        """Tests linear array + convex PWI delays."""
        n_elements = 128
        device = DeviceMock(ProbeMock(
            ProbeModel(model_id=ProbeModelId("ultrasonix", "l14-5/38"),
                       pitch=0.3048e-3,
                       n_elements=n_elements,
                       curvature_radius=0.0)))
        n_elements = device.probe.model.n_elements

        ops = [
            TxRx(
                tx=Tx(
                    aperture=Aperture(center=0),
                    excitation=self.default_excitation,
                    focus=np.inf,
                    angle=angle,
                    speed_of_sound=1450,
                ),
                rx=Rx(
                    aperture=Aperture(center=0),
                    sample_range=self.default_sample_range,
                    downsampling_factor=1,
                ),
                pri=1,
            )
            for angle in np.array([-10, 0, 10])*np.pi/180
        ]
        input_sequence = TxRxSequence(ops=ops, tgc_curve=[])
        context = ContextMock(device=device,
                              medium=self.default_medium,
                              op=input_sequence)
        result = arrus.kernels.tx_rx_sequence.process_tx_rx_sequence(context)
        tx_rx_sequence = result.sequence
        ops = tx_rx_sequence.ops
        self.assertEqual(len(ops), 3)
        # TX/RX apertures:
        expected_apertures = np.zeros((len(ops), n_elements), dtype=bool)
        expected_apertures[0, :] = True
        expected_apertures[1, :] = True
        expected_apertures[2, :] = True

        expected_delays = [
            np.array([4.635760e-06, 4.599258e-06, 4.562756e-06, 4.526254e-06, 4.489751e-06, 4.453249e-06, 4.416747e-06, 4.380245e-06, 4.343743e-06, 4.307241e-06, 4.270739e-06, 4.234237e-06, 4.197735e-06, 4.161233e-06, 4.124731e-06, 4.088229e-06, 4.051727e-06, 4.015225e-06, 3.978723e-06, 3.942221e-06, 3.905719e-06, 3.869217e-06, 3.832715e-06, 3.796213e-06, 3.759711e-06, 3.723209e-06, 3.686706e-06, 3.650204e-06, 3.613702e-06, 3.577200e-06, 3.540698e-06, 3.504196e-06, 3.467694e-06, 3.431192e-06, 3.394690e-06, 3.358188e-06, 3.321686e-06, 3.285184e-06, 3.248682e-06, 3.212180e-06, 3.175678e-06, 3.139176e-06, 3.102674e-06, 3.066172e-06, 3.029670e-06, 2.993168e-06, 2.956666e-06, 2.920164e-06, 2.883662e-06, 2.847159e-06, 2.810657e-06, 2.774155e-06, 2.737653e-06, 2.701151e-06, 2.664649e-06, 2.628147e-06, 2.591645e-06, 2.555143e-06, 2.518641e-06, 2.482139e-06, 2.445637e-06, 2.409135e-06, 2.372633e-06, 2.336131e-06, 2.299629e-06, 2.263127e-06, 2.226625e-06, 2.190123e-06, 2.153621e-06, 2.117119e-06, 2.080617e-06, 2.044114e-06, 2.007612e-06, 1.971110e-06, 1.934608e-06, 1.898106e-06, 1.861604e-06, 1.825102e-06, 1.788600e-06, 1.752098e-06, 1.715596e-06, 1.679094e-06, 1.642592e-06, 1.606090e-06, 1.569588e-06, 1.533086e-06, 1.496584e-06, 1.460082e-06, 1.423580e-06, 1.387078e-06, 1.350576e-06, 1.314074e-06, 1.277572e-06, 1.241070e-06, 1.204567e-06, 1.168065e-06, 1.131563e-06, 1.095061e-06, 1.058559e-06, 1.022057e-06, 9.855552e-07, 9.490532e-07, 9.125511e-07, 8.760491e-07, 8.395470e-07, 8.030450e-07, 7.665429e-07, 7.300409e-07, 6.935388e-07, 6.570368e-07, 6.205348e-07, 5.840327e-07, 5.475307e-07, 5.110286e-07, 4.745266e-07, 4.380245e-07, 4.015225e-07, 3.650204e-07, 3.285184e-07, 2.920164e-07, 2.555143e-07, 2.190123e-07, 1.825102e-07, 1.460082e-07, 1.095061e-07, 7.300409e-08, 3.650204e-08, 0.000000e+00]),
            np.array([2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06, 2.317880e-06]),
            np.array([0.000000e+00, 3.650204e-08, 7.300409e-08, 1.095061e-07, 1.460082e-07, 1.825102e-07, 2.190123e-07, 2.555143e-07, 2.920164e-07, 3.285184e-07, 3.650204e-07, 4.015225e-07, 4.380245e-07, 4.745266e-07, 5.110286e-07, 5.475307e-07, 5.840327e-07, 6.205348e-07, 6.570368e-07, 6.935388e-07, 7.300409e-07, 7.665429e-07, 8.030450e-07, 8.395470e-07, 8.760491e-07, 9.125511e-07, 9.490532e-07, 9.855552e-07, 1.022057e-06, 1.058559e-06, 1.095061e-06, 1.131563e-06, 1.168065e-06, 1.204567e-06, 1.241070e-06, 1.277572e-06, 1.314074e-06, 1.350576e-06, 1.387078e-06, 1.423580e-06, 1.460082e-06, 1.496584e-06, 1.533086e-06, 1.569588e-06, 1.606090e-06, 1.642592e-06, 1.679094e-06, 1.715596e-06, 1.752098e-06, 1.788600e-06, 1.825102e-06, 1.861604e-06, 1.898106e-06, 1.934608e-06, 1.971110e-06, 2.007612e-06, 2.044114e-06, 2.080617e-06, 2.117119e-06, 2.153621e-06, 2.190123e-06, 2.226625e-06, 2.263127e-06, 2.299629e-06, 2.336131e-06, 2.372633e-06, 2.409135e-06, 2.445637e-06, 2.482139e-06, 2.518641e-06, 2.555143e-06, 2.591645e-06, 2.628147e-06, 2.664649e-06, 2.701151e-06, 2.737653e-06, 2.774155e-06, 2.810657e-06, 2.847159e-06, 2.883662e-06, 2.920164e-06, 2.956666e-06, 2.993168e-06, 3.029670e-06, 3.066172e-06, 3.102674e-06, 3.139176e-06, 3.175678e-06, 3.212180e-06, 3.248682e-06, 3.285184e-06, 3.321686e-06, 3.358188e-06, 3.394690e-06, 3.431192e-06, 3.467694e-06, 3.504196e-06, 3.540698e-06, 3.577200e-06, 3.613702e-06, 3.650204e-06, 3.686706e-06, 3.723209e-06, 3.759711e-06, 3.796213e-06, 3.832715e-06, 3.869217e-06, 3.905719e-06, 3.942221e-06, 3.978723e-06, 4.015225e-06, 4.051727e-06, 4.088229e-06, 4.124731e-06, 4.161233e-06, 4.197735e-06, 4.234237e-06, 4.270739e-06, 4.307241e-06, 4.343743e-06, 4.380245e-06, 4.416747e-06, 4.453249e-06, 4.489751e-06, 4.526254e-06, 4.562756e-06, 4.599258e-06, 4.635760e-06])
        ]
        for i, op in enumerate(ops):
            tx_aperture = op.tx.aperture
            rx_aperture = op.rx.aperture
            tx_delays = op.tx.delays
            np.testing.assert_equal(np.argwhere(tx_aperture), np.argwhere(expected_apertures[i]))
            np.testing.assert_equal(np.argwhere(rx_aperture), np.argwhere(expected_apertures[i]))
            np.testing.assert_almost_equal(expected_delays[i], tx_delays, decimal=12)

    def test_reference_compliance_convex_array_ac2541_128_elements(self):
        """ Tests aperture position and convex PWI delays."""
        n_elements = 192
        device = DeviceMock(ProbeMock(
            ProbeModel(model_id=ProbeModelId("esaote", "ac2541"),
                       pitch=0.3e-3,
                       n_elements=n_elements,
                       curvature_radius=50e-3)))
        n_elements = device.probe.model.n_elements
        pitch = device.probe.model.pitch

        angles = np.tile([0], 9)
        aperture_centers = np.repeat(np.linspace(-12, 12, 9)*1e-3, 1)

        ops = [
            TxRx(
                tx=Tx(
                    aperture=Aperture(center=ap_center, size=128),
                    excitation=self.default_excitation,
                    focus=np.inf,
                    angle=angle,
                    speed_of_sound=1490,
                ),
                rx=Rx(
                    aperture=Aperture(center=ap_center, size=128),
                    sample_range=self.default_sample_range,
                    downsampling_factor=1,
                ),
                pri=1,
            )
            for angle, ap_center in zip(angles, aperture_centers)
        ]
        input_sequence = TxRxSequence(ops=ops, tgc_curve=[])
        context = ContextMock(device=device,
                              medium=self.default_medium,
                              op=input_sequence)
        result = arrus.kernels.tx_rx_sequence.process_tx_rx_sequence(context)
        tx_rx_sequence = result.sequence
        ops = tx_rx_sequence.ops
        self.assertEqual(len(ops), 9)
        # TX/RX apertures:
        expected_apertures = np.zeros((len(ops), n_elements), dtype=bool)
        expected_apertures[0,  0:120] = True
        expected_apertures[1,  2:130] = True
        expected_apertures[2, 12:140] = True
        expected_apertures[3, 22:150] = True
        expected_apertures[4, 32:160] = True
        expected_apertures[5, 42:170] = True
        expected_apertures[6, 52:180] = True
        expected_apertures[7, 62:190] = True
        expected_apertures[8, 72:192] = True

        expected_delays = [
            np.array([5.628425e-07, 6.280860e-07, 6.921854e-07, 7.551385e-07, 8.169430e-07, 8.775966e-07, 9.370973e-07, 9.954427e-07, 1.052631e-06, 1.108660e-06, 1.163527e-06, 1.217232e-06, 1.269771e-06, 1.321142e-06, 1.371345e-06, 1.420377e-06, 1.468237e-06, 1.514922e-06, 1.560431e-06, 1.604763e-06, 1.647916e-06, 1.689887e-06, 1.730677e-06, 1.770283e-06, 1.808703e-06, 1.845937e-06, 1.881983e-06, 1.916840e-06, 1.950507e-06, 1.982982e-06, 2.014264e-06, 2.044352e-06, 2.073245e-06, 2.100942e-06, 2.127442e-06, 2.152744e-06, 2.176847e-06, 2.199751e-06, 2.221453e-06, 2.241955e-06, 2.261254e-06, 2.279350e-06, 2.296243e-06, 2.311932e-06, 2.326416e-06, 2.339695e-06, 2.351768e-06, 2.362635e-06, 2.372296e-06, 2.380750e-06, 2.387997e-06, 2.394036e-06, 2.398868e-06, 2.402492e-06, 2.404908e-06, 2.406116e-06, 2.406116e-06, 2.404908e-06, 2.402492e-06, 2.398868e-06, 2.394036e-06, 2.387997e-06, 2.380750e-06, 2.372296e-06, 2.362635e-06, 2.351768e-06, 2.339695e-06, 2.326416e-06, 2.311932e-06, 2.296243e-06, 2.279350e-06, 2.261254e-06, 2.241955e-06, 2.221453e-06, 2.199751e-06, 2.176847e-06, 2.152744e-06, 2.127442e-06, 2.100942e-06, 2.073245e-06, 2.044352e-06, 2.014264e-06, 1.982982e-06, 1.950507e-06, 1.916840e-06, 1.881983e-06, 1.845937e-06, 1.808703e-06, 1.770283e-06, 1.730677e-06, 1.689887e-06, 1.647916e-06, 1.604763e-06, 1.560431e-06, 1.514922e-06, 1.468237e-06, 1.420377e-06, 1.371345e-06, 1.321142e-06, 1.269771e-06, 1.217232e-06, 1.163527e-06, 1.108660e-06, 1.052631e-06, 9.954427e-07, 9.370973e-07, 8.775966e-07, 8.169430e-07, 7.551385e-07, 6.921854e-07, 6.280860e-07, 5.628425e-07, 4.964574e-07, 4.289329e-07, 3.602716e-07, 2.904759e-07, 2.195483e-07, 1.474914e-07, 7.430775e-08, 1.694066e-21]),
            np.array([3.388132e-21, 7.430775e-08, 1.474914e-07, 2.195483e-07, 2.904759e-07, 3.602716e-07, 4.289329e-07, 4.964574e-07, 5.628425e-07, 6.280860e-07, 6.921854e-07, 7.551385e-07, 8.169430e-07, 8.775966e-07, 9.370973e-07, 9.954427e-07, 1.052631e-06, 1.108660e-06, 1.163527e-06, 1.217232e-06, 1.269771e-06, 1.321142e-06, 1.371345e-06, 1.420377e-06, 1.468237e-06, 1.514922e-06, 1.560431e-06, 1.604763e-06, 1.647916e-06, 1.689887e-06, 1.730677e-06, 1.770283e-06, 1.808703e-06, 1.845937e-06, 1.881983e-06, 1.916840e-06, 1.950507e-06, 1.982982e-06, 2.014264e-06, 2.044352e-06, 2.073245e-06, 2.100942e-06, 2.127442e-06, 2.152744e-06, 2.176847e-06, 2.199751e-06, 2.221453e-06, 2.241955e-06, 2.261254e-06, 2.279350e-06, 2.296243e-06, 2.311932e-06, 2.326416e-06, 2.339695e-06, 2.351768e-06, 2.362635e-06, 2.372296e-06, 2.380750e-06, 2.387997e-06, 2.394036e-06, 2.398868e-06, 2.402492e-06, 2.404908e-06, 2.406116e-06, 2.406116e-06, 2.404908e-06, 2.402492e-06, 2.398868e-06, 2.394036e-06, 2.387997e-06, 2.380750e-06, 2.372296e-06, 2.362635e-06, 2.351768e-06, 2.339695e-06, 2.326416e-06, 2.311932e-06, 2.296243e-06, 2.279350e-06, 2.261254e-06, 2.241955e-06, 2.221453e-06, 2.199751e-06, 2.176847e-06, 2.152744e-06, 2.127442e-06, 2.100942e-06, 2.073245e-06, 2.044352e-06, 2.014264e-06, 1.982982e-06, 1.950507e-06, 1.916840e-06, 1.881983e-06, 1.845937e-06, 1.808703e-06, 1.770283e-06, 1.730677e-06, 1.689887e-06, 1.647916e-06, 1.604763e-06, 1.560431e-06, 1.514922e-06, 1.468237e-06, 1.420377e-06, 1.371345e-06, 1.321142e-06, 1.269771e-06, 1.217232e-06, 1.163527e-06, 1.108660e-06, 1.052631e-06, 9.954427e-07, 9.370973e-07, 8.775966e-07, 8.169430e-07, 7.551385e-07, 6.921854e-07, 6.280860e-07, 5.628425e-07, 4.964574e-07, 4.289329e-07, 3.602716e-07, 2.904759e-07, 2.195483e-07, 1.474914e-07, 7.430775e-08, 1.270549e-21]),
            np.array([0.0,          7.430775e-08, 1.474914e-07, 2.195483e-07, 2.904759e-07, 3.602716e-07, 4.289329e-07, 4.964574e-07, 5.628425e-07, 6.280860e-07, 6.921854e-07, 7.551385e-07, 8.169430e-07, 8.775966e-07, 9.370973e-07, 9.954427e-07, 1.052631e-06, 1.108660e-06, 1.163527e-06, 1.217232e-06, 1.269771e-06, 1.321142e-06, 1.371345e-06, 1.420377e-06, 1.468237e-06, 1.514922e-06, 1.560431e-06, 1.604763e-06, 1.647916e-06, 1.689887e-06, 1.730677e-06, 1.770283e-06, 1.808703e-06, 1.845937e-06, 1.881983e-06, 1.916840e-06, 1.950507e-06, 1.982982e-06, 2.014264e-06, 2.044352e-06, 2.073245e-06, 2.100942e-06, 2.127442e-06, 2.152744e-06, 2.176847e-06, 2.199751e-06, 2.221453e-06, 2.241955e-06, 2.261254e-06, 2.279350e-06, 2.296243e-06, 2.311932e-06, 2.326416e-06, 2.339695e-06, 2.351768e-06, 2.362635e-06, 2.372296e-06, 2.380750e-06, 2.387997e-06, 2.394036e-06, 2.398868e-06, 2.402492e-06, 2.404908e-06, 2.406116e-06, 2.406116e-06, 2.404908e-06, 2.402492e-06, 2.398868e-06, 2.394036e-06, 2.387997e-06, 2.380750e-06, 2.372296e-06, 2.362635e-06, 2.351768e-06, 2.339695e-06, 2.326416e-06, 2.311932e-06, 2.296243e-06, 2.279350e-06, 2.261254e-06, 2.241955e-06, 2.221453e-06, 2.199751e-06, 2.176847e-06, 2.152744e-06, 2.127442e-06, 2.100942e-06, 2.073245e-06, 2.044352e-06, 2.014264e-06, 1.982982e-06, 1.950507e-06, 1.916840e-06, 1.881983e-06, 1.845937e-06, 1.808703e-06, 1.770283e-06, 1.730677e-06, 1.689887e-06, 1.647916e-06, 1.604763e-06, 1.560431e-06, 1.514922e-06, 1.468237e-06, 1.420377e-06, 1.371345e-06, 1.321142e-06, 1.269771e-06, 1.217232e-06, 1.163527e-06, 1.108660e-06, 1.052631e-06, 9.954427e-07, 9.370973e-07, 8.775966e-07, 8.169430e-07, 7.551385e-07, 6.921854e-07, 6.280860e-07, 5.628425e-07, 4.964574e-07, 4.289329e-07, 3.602716e-07, 2.904759e-07, 2.195483e-07, 1.474914e-07, 7.430775e-08, 4.235165e-22]),
            np.array([4.235165e-22, 7.430775e-08, 1.474914e-07, 2.195483e-07, 2.904759e-07, 3.602716e-07, 4.289329e-07, 4.964574e-07, 5.628425e-07, 6.280860e-07, 6.921854e-07, 7.551385e-07, 8.169430e-07, 8.775966e-07, 9.370973e-07, 9.954427e-07, 1.052631e-06, 1.108660e-06, 1.163527e-06, 1.217232e-06, 1.269771e-06, 1.321142e-06, 1.371345e-06, 1.420377e-06, 1.468237e-06, 1.514922e-06, 1.560431e-06, 1.604763e-06, 1.647916e-06, 1.689887e-06, 1.730677e-06, 1.770283e-06, 1.808703e-06, 1.845937e-06, 1.881983e-06, 1.916840e-06, 1.950507e-06, 1.982982e-06, 2.014264e-06, 2.044352e-06, 2.073245e-06, 2.100942e-06, 2.127442e-06, 2.152744e-06, 2.176847e-06, 2.199751e-06, 2.221453e-06, 2.241955e-06, 2.261254e-06, 2.279350e-06, 2.296243e-06, 2.311932e-06, 2.326416e-06, 2.339695e-06, 2.351768e-06, 2.362635e-06, 2.372296e-06, 2.380750e-06, 2.387997e-06, 2.394036e-06, 2.398868e-06, 2.402492e-06, 2.404908e-06, 2.406116e-06, 2.406116e-06, 2.404908e-06, 2.402492e-06, 2.398868e-06, 2.394036e-06, 2.387997e-06, 2.380750e-06, 2.372296e-06, 2.362635e-06, 2.351768e-06, 2.339695e-06, 2.326416e-06, 2.311932e-06, 2.296243e-06, 2.279350e-06, 2.261254e-06, 2.241955e-06, 2.221453e-06, 2.199751e-06, 2.176847e-06, 2.152744e-06, 2.127442e-06, 2.100942e-06, 2.073245e-06, 2.044352e-06, 2.014264e-06, 1.982982e-06, 1.950507e-06, 1.916840e-06, 1.881983e-06, 1.845937e-06, 1.808703e-06, 1.770283e-06, 1.730677e-06, 1.689887e-06, 1.647916e-06, 1.604763e-06, 1.560431e-06, 1.514922e-06, 1.468237e-06, 1.420377e-06, 1.371345e-06, 1.321142e-06, 1.269771e-06, 1.217232e-06, 1.163527e-06, 1.108660e-06, 1.052631e-06, 9.954427e-07, 9.370973e-07, 8.775966e-07, 8.169430e-07, 7.551385e-07, 6.921854e-07, 6.280860e-07, 5.628425e-07, 4.964574e-07, 4.289329e-07, 3.602716e-07, 2.904759e-07, 2.195483e-07, 1.474914e-07, 7.430775e-08, 0.0]),
            np.array([1.270549e-21, 7.430775e-08, 1.474914e-07, 2.195483e-07, 2.904759e-07, 3.602716e-07, 4.289329e-07, 4.964574e-07, 5.628425e-07, 6.280860e-07, 6.921854e-07, 7.551385e-07, 8.169430e-07, 8.775966e-07, 9.370973e-07, 9.954427e-07, 1.052631e-06, 1.108660e-06, 1.163527e-06, 1.217232e-06, 1.269771e-06, 1.321142e-06, 1.371345e-06, 1.420377e-06, 1.468237e-06, 1.514922e-06, 1.560431e-06, 1.604763e-06, 1.647916e-06, 1.689887e-06, 1.730677e-06, 1.770283e-06, 1.808703e-06, 1.845937e-06, 1.881983e-06, 1.916840e-06, 1.950507e-06, 1.982982e-06, 2.014264e-06, 2.044352e-06, 2.073245e-06, 2.100942e-06, 2.127442e-06, 2.152744e-06, 2.176847e-06, 2.199751e-06, 2.221453e-06, 2.241955e-06, 2.261254e-06, 2.279350e-06, 2.296243e-06, 2.311932e-06, 2.326416e-06, 2.339695e-06, 2.351768e-06, 2.362635e-06, 2.372296e-06, 2.380750e-06, 2.387997e-06, 2.394036e-06, 2.398868e-06, 2.402492e-06, 2.404908e-06, 2.406116e-06, 2.406116e-06, 2.404908e-06, 2.402492e-06, 2.398868e-06, 2.394036e-06, 2.387997e-06, 2.380750e-06, 2.372296e-06, 2.362635e-06, 2.351768e-06, 2.339695e-06, 2.326416e-06, 2.311932e-06, 2.296243e-06, 2.279350e-06, 2.261254e-06, 2.241955e-06, 2.221453e-06, 2.199751e-06, 2.176847e-06, 2.152744e-06, 2.127442e-06, 2.100942e-06, 2.073245e-06, 2.044352e-06, 2.014264e-06, 1.982982e-06, 1.950507e-06, 1.916840e-06, 1.881983e-06, 1.845937e-06, 1.808703e-06, 1.770283e-06, 1.730677e-06, 1.689887e-06, 1.647916e-06, 1.604763e-06, 1.560431e-06, 1.514922e-06, 1.468237e-06, 1.420377e-06, 1.371345e-06, 1.321142e-06, 1.269771e-06, 1.217232e-06, 1.163527e-06, 1.108660e-06, 1.052631e-06, 9.954427e-07, 9.370973e-07, 8.775966e-07, 8.169430e-07, 7.551385e-07, 6.921854e-07, 6.280860e-07, 5.628425e-07, 4.964574e-07, 4.289329e-07, 3.602716e-07, 2.904759e-07, 2.195483e-07, 1.474914e-07, 7.430775e-08, 1.270549e-21]),
            np.array([0.0,          7.430775e-08, 1.474914e-07, 2.195483e-07, 2.904759e-07, 3.602716e-07, 4.289329e-07, 4.964574e-07, 5.628425e-07, 6.280860e-07, 6.921854e-07, 7.551385e-07, 8.169430e-07, 8.775966e-07, 9.370973e-07, 9.954427e-07, 1.052631e-06, 1.108660e-06, 1.163527e-06, 1.217232e-06, 1.269771e-06, 1.321142e-06, 1.371345e-06, 1.420377e-06, 1.468237e-06, 1.514922e-06, 1.560431e-06, 1.604763e-06, 1.647916e-06, 1.689887e-06, 1.730677e-06, 1.770283e-06, 1.808703e-06, 1.845937e-06, 1.881983e-06, 1.916840e-06, 1.950507e-06, 1.982982e-06, 2.014264e-06, 2.044352e-06, 2.073245e-06, 2.100942e-06, 2.127442e-06, 2.152744e-06, 2.176847e-06, 2.199751e-06, 2.221453e-06, 2.241955e-06, 2.261254e-06, 2.279350e-06, 2.296243e-06, 2.311932e-06, 2.326416e-06, 2.339695e-06, 2.351768e-06, 2.362635e-06, 2.372296e-06, 2.380750e-06, 2.387997e-06, 2.394036e-06, 2.398868e-06, 2.402492e-06, 2.404908e-06, 2.406116e-06, 2.406116e-06, 2.404908e-06, 2.402492e-06, 2.398868e-06, 2.394036e-06, 2.387997e-06, 2.380750e-06, 2.372296e-06, 2.362635e-06, 2.351768e-06, 2.339695e-06, 2.326416e-06, 2.311932e-06, 2.296243e-06, 2.279350e-06, 2.261254e-06, 2.241955e-06, 2.221453e-06, 2.199751e-06, 2.176847e-06, 2.152744e-06, 2.127442e-06, 2.100942e-06, 2.073245e-06, 2.044352e-06, 2.014264e-06, 1.982982e-06, 1.950507e-06, 1.916840e-06, 1.881983e-06, 1.845937e-06, 1.808703e-06, 1.770283e-06, 1.730677e-06, 1.689887e-06, 1.647916e-06, 1.604763e-06, 1.560431e-06, 1.514922e-06, 1.468237e-06, 1.420377e-06, 1.371345e-06, 1.321142e-06, 1.269771e-06, 1.217232e-06, 1.163527e-06, 1.108660e-06, 1.052631e-06, 9.954427e-07, 9.370973e-07, 8.775966e-07, 8.169430e-07, 7.551385e-07, 6.921854e-07, 6.280860e-07, 5.628425e-07, 4.964574e-07, 4.289329e-07, 3.602716e-07, 2.904759e-07, 2.195483e-07, 1.474914e-07, 7.430775e-08, 4.235165e-22]),
            np.array([4.235165e-22, 7.430775e-08, 1.474914e-07, 2.195483e-07, 2.904759e-07, 3.602716e-07, 4.289329e-07, 4.964574e-07, 5.628425e-07, 6.280860e-07, 6.921854e-07, 7.551385e-07, 8.169430e-07, 8.775966e-07, 9.370973e-07, 9.954427e-07, 1.052631e-06, 1.108660e-06, 1.163527e-06, 1.217232e-06, 1.269771e-06, 1.321142e-06, 1.371345e-06, 1.420377e-06, 1.468237e-06, 1.514922e-06, 1.560431e-06, 1.604763e-06, 1.647916e-06, 1.689887e-06, 1.730677e-06, 1.770283e-06, 1.808703e-06, 1.845937e-06, 1.881983e-06, 1.916840e-06, 1.950507e-06, 1.982982e-06, 2.014264e-06, 2.044352e-06, 2.073245e-06, 2.100942e-06, 2.127442e-06, 2.152744e-06, 2.176847e-06, 2.199751e-06, 2.221453e-06, 2.241955e-06, 2.261254e-06, 2.279350e-06, 2.296243e-06, 2.311932e-06, 2.326416e-06, 2.339695e-06, 2.351768e-06, 2.362635e-06, 2.372296e-06, 2.380750e-06, 2.387997e-06, 2.394036e-06, 2.398868e-06, 2.402492e-06, 2.404908e-06, 2.406116e-06, 2.406116e-06, 2.404908e-06, 2.402492e-06, 2.398868e-06, 2.394036e-06, 2.387997e-06, 2.380750e-06, 2.372296e-06, 2.362635e-06, 2.351768e-06, 2.339695e-06, 2.326416e-06, 2.311932e-06, 2.296243e-06, 2.279350e-06, 2.261254e-06, 2.241955e-06, 2.221453e-06, 2.199751e-06, 2.176847e-06, 2.152744e-06, 2.127442e-06, 2.100942e-06, 2.073245e-06, 2.044352e-06, 2.014264e-06, 1.982982e-06, 1.950507e-06, 1.916840e-06, 1.881983e-06, 1.845937e-06, 1.808703e-06, 1.770283e-06, 1.730677e-06, 1.689887e-06, 1.647916e-06, 1.604763e-06, 1.560431e-06, 1.514922e-06, 1.468237e-06, 1.420377e-06, 1.371345e-06, 1.321142e-06, 1.269771e-06, 1.217232e-06, 1.163527e-06, 1.108660e-06, 1.052631e-06, 9.954427e-07, 9.370973e-07, 8.775966e-07, 8.169430e-07, 7.551385e-07, 6.921854e-07, 6.280860e-07, 5.628425e-07, 4.964574e-07, 4.289329e-07, 3.602716e-07, 2.904759e-07, 2.195483e-07, 1.474914e-07, 7.430775e-08, 0.0]),
            np.array([1.270549e-21, 7.430775e-08, 1.474914e-07, 2.195483e-07, 2.904759e-07, 3.602716e-07, 4.289329e-07, 4.964574e-07, 5.628425e-07, 6.280860e-07, 6.921854e-07, 7.551385e-07, 8.169430e-07, 8.775966e-07, 9.370973e-07, 9.954427e-07, 1.052631e-06, 1.108660e-06, 1.163527e-06, 1.217232e-06, 1.269771e-06, 1.321142e-06, 1.371345e-06, 1.420377e-06, 1.468237e-06, 1.514922e-06, 1.560431e-06, 1.604763e-06, 1.647916e-06, 1.689887e-06, 1.730677e-06, 1.770283e-06, 1.808703e-06, 1.845937e-06, 1.881983e-06, 1.916840e-06, 1.950507e-06, 1.982982e-06, 2.014264e-06, 2.044352e-06, 2.073245e-06, 2.100942e-06, 2.127442e-06, 2.152744e-06, 2.176847e-06, 2.199751e-06, 2.221453e-06, 2.241955e-06, 2.261254e-06, 2.279350e-06, 2.296243e-06, 2.311932e-06, 2.326416e-06, 2.339695e-06, 2.351768e-06, 2.362635e-06, 2.372296e-06, 2.380750e-06, 2.387997e-06, 2.394036e-06, 2.398868e-06, 2.402492e-06, 2.404908e-06, 2.406116e-06, 2.406116e-06, 2.404908e-06, 2.402492e-06, 2.398868e-06, 2.394036e-06, 2.387997e-06, 2.380750e-06, 2.372296e-06, 2.362635e-06, 2.351768e-06, 2.339695e-06, 2.326416e-06, 2.311932e-06, 2.296243e-06, 2.279350e-06, 2.261254e-06, 2.241955e-06, 2.221453e-06, 2.199751e-06, 2.176847e-06, 2.152744e-06, 2.127442e-06, 2.100942e-06, 2.073245e-06, 2.044352e-06, 2.014264e-06, 1.982982e-06, 1.950507e-06, 1.916840e-06, 1.881983e-06, 1.845937e-06, 1.808703e-06, 1.770283e-06, 1.730677e-06, 1.689887e-06, 1.647916e-06, 1.604763e-06, 1.560431e-06, 1.514922e-06, 1.468237e-06, 1.420377e-06, 1.371345e-06, 1.321142e-06, 1.269771e-06, 1.217232e-06, 1.163527e-06, 1.108660e-06, 1.052631e-06, 9.954427e-07, 9.370973e-07, 8.775966e-07, 8.169430e-07, 7.551385e-07, 6.921854e-07, 6.280860e-07, 5.628425e-07, 4.964574e-07, 4.289329e-07, 3.602716e-07, 2.904759e-07, 2.195483e-07, 1.474914e-07, 7.430775e-08, 2.541099e-21]),
            np.array([1.694066e-21, 7.430775e-08, 1.474914e-07, 2.195483e-07, 2.904759e-07, 3.602716e-07, 4.289329e-07, 4.964574e-07, 5.628425e-07, 6.280860e-07, 6.921854e-07, 7.551385e-07, 8.169430e-07, 8.775966e-07, 9.370973e-07, 9.954427e-07, 1.052631e-06, 1.108660e-06, 1.163527e-06, 1.217232e-06, 1.269771e-06, 1.321142e-06, 1.371345e-06, 1.420377e-06, 1.468237e-06, 1.514922e-06, 1.560431e-06, 1.604763e-06, 1.647916e-06, 1.689887e-06, 1.730677e-06, 1.770283e-06, 1.808703e-06, 1.845937e-06, 1.881983e-06, 1.916840e-06, 1.950507e-06, 1.982982e-06, 2.014264e-06, 2.044352e-06, 2.073245e-06, 2.100942e-06, 2.127442e-06, 2.152744e-06, 2.176847e-06, 2.199751e-06, 2.221453e-06, 2.241955e-06, 2.261254e-06, 2.279350e-06, 2.296243e-06, 2.311932e-06, 2.326416e-06, 2.339695e-06, 2.351768e-06, 2.362635e-06, 2.372296e-06, 2.380750e-06, 2.387997e-06, 2.394036e-06, 2.398868e-06, 2.402492e-06, 2.404908e-06, 2.406116e-06, 2.406116e-06, 2.404908e-06, 2.402492e-06, 2.398868e-06, 2.394036e-06, 2.387997e-06, 2.380750e-06, 2.372296e-06, 2.362635e-06, 2.351768e-06, 2.339695e-06, 2.326416e-06, 2.311932e-06, 2.296243e-06, 2.279350e-06, 2.261254e-06, 2.241955e-06, 2.221453e-06, 2.199751e-06, 2.176847e-06, 2.152744e-06, 2.127442e-06, 2.100942e-06, 2.073245e-06, 2.044352e-06, 2.014264e-06, 1.982982e-06, 1.950507e-06, 1.916840e-06, 1.881983e-06, 1.845937e-06, 1.808703e-06, 1.770283e-06, 1.730677e-06, 1.689887e-06, 1.647916e-06, 1.604763e-06, 1.560431e-06, 1.514922e-06, 1.468237e-06, 1.420377e-06, 1.371345e-06, 1.321142e-06, 1.269771e-06, 1.217232e-06, 1.163527e-06, 1.108660e-06, 1.052631e-06, 9.954427e-07, 9.370973e-07, 8.775966e-07, 8.169430e-07, 7.551385e-07, 6.921854e-07, 6.280860e-07, 5.628425e-07])
        ]

        for i, op in enumerate(ops):
            tx_aperture = op.tx.aperture
            rx_aperture = op.rx.aperture
            tx_delays = op.tx.delays
            np.testing.assert_equal(np.argwhere(tx_aperture), np.argwhere(expected_apertures[i]))
            np.testing.assert_equal(np.argwhere(rx_aperture), np.argwhere(expected_apertures[i]))
            np.testing.assert_almost_equal(expected_delays[i], tx_delays, decimal=12)


# TODO already tested in simple_tx_rx_sequence_test.py module, consider enabling below
# When the functionality of the simple_tx_rx_sequence.py will be minimized
# class LinSequenceTest(SimpleTxRxSequenceTest):
#
#     def setUp(self) -> None:
#         super().setUp()
#         self.default_sequence = arrus.ops.imaging.LinSequence(
#             pulse=arrus.ops.us4r.Pulse(
#                 center_frequency=1,
#                 n_periods=1,
#                 inverse=False),
#             tx_focus=20e-3,
#             rx_sample_range=(0, 8),
#             downsampling_factor=1,
#             speed_of_sound=1,
#             pri=1,
#             sri=1,
#             tgc_start=1,
#             tgc_slope=1,
#             tx_aperture_center_element=3.5,
#             tx_aperture_size=8,
#             rx_aperture_center_element=3.5,
#             rx_aperture_size=8,
#         )
#
#     def test_simple_sequence_with_paddings(self):
#         # three tx/rxs with aperture centers: 0, 16, 31
#         seq = arrus.ops.imaging.LinSequence(
#             tx_aperture_center_element=np.array([0, 15, 16, 31]),
#             tx_aperture_size=32,
#             tx_focus=30e-3,
#             pulse=arrus.ops.us4r.Pulse(center_frequency=5e6, n_periods=3,
#                                        inverse=False),
#             rx_aperture_center_element=np.array([0, 15, 16, 31]),
#             rx_aperture_size=32,
#             pri=1000e-6,
#             downsampling_factor=1,
#             rx_sample_range=(0, 4096),
#             tgc_start=14,
#             tgc_slope=0)
#
#         medium = arrus.medium.MediumDTO(name="test", speed_of_sound=1540)
#         probe_model = ProbeModel(
#             model_id="id",
#             pitch=0.2e-3, n_elements=32, curvature_radius=0.0)
#         device = DeviceMock(probe=ProbeMock(model=probe_model))
#         context = ContextMock(device=device, medium=medium, op=seq)
#         tx_rx_sequence = arrus.kernels.simple_tx_rx_sequence.process_simple_tx_rx_sequence(
#             context)
#         # A TX/RX sequence without
#         seq2 = arrus.kernels.simple_tx_rx_sequence.convert_to_tx_rx_sequence(
#             c=medium.speed_of_sound, op=seq, probe_model=probe_model
#         )
#         seq2_with_mask: TxRxSequence = arrus.kernels.tx_rx_sequence.set_aperture_masks(
#             sequence=seq2, probe=probe_model
#         )
#         delays, _ = arrus.kernels.tx_rx_sequence.get_tx_delays(probe_model, seq2, seq2_with_mask)
#
#         # expected delays
#         tx_rx_params = arrus.kernels.simple_tx_rx_sequence.compute_tx_rx_params(
#             probe_model, seq)
#
#         # Expected aperture tx/rx 1
#         expected_aperture = np.zeros((32,), dtype=bool)
#         expected_aperture[0:16+1] = True
#         expected_delays = np.squeeze(delays[0])
#         tx_rx = tx_rx_sequence.ops[0]
#         np.testing.assert_array_equal(tx_rx.tx.aperture, expected_aperture)
#         np.testing.assert_array_equal(tx_rx.rx.aperture, expected_aperture)
#         np.testing.assert_almost_equal(tx_rx.tx.delays, expected_delays)
#
#         # Expected aperture tx/rx 2
#         expected_aperture = np.ones((32,), dtype=bool)
#         tx_rx = tx_rx_sequence.ops[1]
#         np.testing.assert_array_equal(tx_rx.tx.aperture, expected_aperture)
#         np.testing.assert_array_equal(tx_rx.rx.aperture, expected_aperture)
#
#         # Expected aperture tx/rx 3
#         expected_aperture = np.zeros((32,), dtype=bool)
#         expected_aperture[1:] = True
#         tx_rx = tx_rx_sequence.ops[2]
#         np.testing.assert_array_equal(tx_rx.tx.aperture, expected_aperture)
#         np.testing.assert_array_equal(tx_rx.rx.aperture, expected_aperture)
#
#         # Expected aperture tx/rx 4
#         expected_aperture = np.zeros((32,), dtype=bool)
#         expected_aperture[16:] = True
#         tx_rx = tx_rx_sequence.ops[3]
#         np.testing.assert_array_equal(tx_rx.tx.aperture, expected_aperture)
#         np.testing.assert_array_equal(tx_rx.rx.aperture, expected_aperture)
#
#     # TODO check the basic properties of the delays profile, e.g.
#     # that is symmetrical.
#
#     def test_compliance_linear_array_ultrasonix_l14_5_38_parameters(self):
#         """Tests linear array + LIN delays."""
#         n_elements = 128
#         device = DeviceMock(ProbeMock(
#             ProbeModel(model_id=ProbeModelId("ultrasonix", "l14-5/38"),
#                        pitch=0.3048e-3,
#                        n_elements=n_elements,
#                        curvature_radius=0.0)))
#
#         n_elements = device.probe.model.n_elements
#         input_sequence = self.default_sequence
#         aperture_size = 64
#         input_sequence = dataclasses.replace(
#             input_sequence,
#             tx_aperture_center_element=np.arange(0, n_elements),
#             tx_aperture_size=aperture_size,
#             rx_aperture_center_element=np.arange(0, n_elements),
#             rx_aperture_size=aperture_size,
#             speed_of_sound=1540
#         )
#         context = ContextMock(device=device,
#                               medium=self.default_medium,
#                               op=input_sequence)
#         tx_rx_sequence = arrus.kernels.simple_tx_rx_sequence.process_simple_tx_rx_sequence(context)
#         ops = tx_rx_sequence.ops
#         self.assertEqual(len(ops), 128)
#         # TX/RX expected delays.
#         # In this sequence, all TX delays will have the same profile.
#         # Only the delays for the aperture located at the probe border will
#         # differ, i.e. the list will be appropriately clipped.
#         expected_delays_profile = np.array([8.565409e-08, 1.690786e-07, 2.502341e-07, 3.290816e-07, 4.055822e-07, 4.796973e-07, 5.513887e-07, 6.206189e-07, 6.873505e-07, 7.515471e-07, 8.131729e-07, 8.721930e-07, 9.285730e-07, 9.822800e-07, 1.033282e-06, 1.081547e-06, 1.127047e-06, 1.169753e-06, 1.209637e-06, 1.246674e-06, 1.280840e-06, 1.312113e-06, 1.340471e-06, 1.365897e-06, 1.388372e-06, 1.407883e-06, 1.424414e-06, 1.437955e-06, 1.448497e-06, 1.456032e-06, 1.460555e-06, 1.462063e-06, 1.460555e-06, 1.456032e-06, 1.448497e-06, 1.437955e-06, 1.424414e-06, 1.407883e-06, 1.388372e-06, 1.365897e-06, 1.340471e-06, 1.312113e-06, 1.280840e-06, 1.246674e-06, 1.209637e-06, 1.169753e-06, 1.127047e-06, 1.081547e-06, 1.033282e-06, 9.822800e-07, 9.285730e-07, 8.721930e-07, 8.131729e-07, 7.515471e-07, 6.873505e-07, 6.206189e-07, 5.513887e-07, 4.796973e-07, 4.055822e-07, 3.290816e-07, 2.502341e-07, 1.690786e-07, 8.565409e-08, 0.000000e+00])
#         self.assert_ok_for_moving_aperture(aperture_size,
#                                            expected_delays_profile,
#                                            n_elements, ops)
#
#     def test_compliance_convex_array_esaote_ac2541(self):
#         """Tests convex array + LIN delays."""
#         n_elements = 192
#         device = DeviceMock(ProbeMock(
#             ProbeModel(model_id=ProbeModelId("esaote", "2541"),
#                        pitch=0.3e-3,
#                        n_elements=n_elements,
#                        curvature_radius=50e-3)))
#
#         n_elements = device.probe.model.n_elements
#         input_sequence = self.default_sequence
#         aperture_size = 32
#         input_sequence = dataclasses.replace(
#             input_sequence,
#             tx_focus=10e-3,
#             tx_aperture_center_element=np.arange(0, n_elements),
#             tx_aperture_size=aperture_size,
#             rx_aperture_center_element=np.arange(0, n_elements),
#             rx_aperture_size=aperture_size,
#             speed_of_sound=1540
#         )
#         context = ContextMock(device=device,
#                               medium=self.default_medium,
#                               op=input_sequence)
#         tx_rx_sequence = arrus.kernels.simple_tx_rx_sequence.process_simple_tx_rx_sequence(context)
#         ops = tx_rx_sequence.ops
#         self.assertEqual(len(ops), n_elements)
#         expected_delays_profile = np.array([9.671847e-08, 1.883984e-07, 2.748402e-07, 3.55846e-07, 4.312216e-07, 5.007779e-07, 5.643331e-07, 6.217144e-07, 6.727603e-07, 7.173223e-07, 7.55267e-07, 7.86478e-07, 8.108577e-07, 8.283283e-07, 8.388334e-07, 8.42339e-07, 8.388334e-07, 8.283283e-07, 8.108577e-07, 7.86478e-07, 7.55267e-07, 7.173223e-07, 6.727603e-07, 6.217144e-07, 5.643331e-07, 5.007779e-07, 4.312216e-07, 3.55846e-07, 2.748402e-07, 1.883984e-07, 9.671847e-08, 5.082198e-21])
#         self.assert_ok_for_moving_aperture(aperture_size,
#                                            expected_delays_profile,
#                                            n_elements, ops)
#
#
class StaSequenceTest(SimpleTxRxSequenceTest):

    def setUp(self) -> None:
        super().setUp()
        self.default_excitation = arrus.ops.us4r.Pulse(
            center_frequency=1,
            n_periods=1,
            inverse=False)
        self.default_sample_range = (0, 8)

    def test_compliance_linear_array_ultrasonix_l14_5_38_parameters(self):
        """Tests linear array + STA delays."""
        n_elements = 128
        device = DeviceMock(ProbeMock(
            ProbeModel(model_id=ProbeModelId("ultrasonix", "l14-5/38"),
                       pitch=0.3048e-3,
                       n_elements=n_elements,
                       curvature_radius=0.0)))

        n_elements = device.probe.model.n_elements
        aperture_size = 32

        ops = [
            TxRx(
                tx=Tx(
                    aperture=Aperture(center_element=ap_center_element, size=aperture_size),
                    excitation=self.default_excitation,
                    focus=-6e-3,
                    angle=0,
                    speed_of_sound=1540,
                ),
                rx=Rx(
                    aperture=Aperture(center=0),
                    sample_range=self.default_sample_range,
                    downsampling_factor=1,
                ),
                pri=1,
            )
            for ap_center_element in np.arange(0, n_elements)
        ]
        input_sequence = TxRxSequence(ops=ops, tgc_curve=[])
        context = ContextMock(device=device,
                              medium=self.default_medium,
                              op=input_sequence)
        result = arrus.kernels.tx_rx_sequence.process_tx_rx_sequence(context)
        tx_rx_sequence = result.sequence
        ops = tx_rx_sequence.ops
        self.assertEqual(len(ops), 128)
        # TX/RX expected delays.
        # In this sequence, all TX delays will have the same profile.
        # Only the delays for the aperture located at the probe border will
        # differ, i.e. the list will be appropriately clipped.
        expected_delays_profile = np.array([1.002221e-06, 8.848545e-07, 7.729311e-07, 6.668512e-07, 5.670314e-07, 4.739007e-07, 3.878955e-07, 3.094528e-07, 2.390031e-07, 1.769612e-07, 1.237163e-07, 7.962194e-08, 4.498528e-08, 2.005726e-08, 5.023982e-09, 0.000000e+00, 5.023982e-09, 2.005726e-08, 4.498528e-08, 7.962194e-08, 1.237163e-07, 1.769612e-07, 2.390031e-07, 3.094528e-07, 3.878955e-07, 4.739007e-07, 5.670314e-07, 6.668512e-07, 7.729311e-07, 8.848545e-07, 1.002221e-06, 1.124648e-06])
        self.assert_ok_for_moving_aperture(
            aperture_size, expected_delays_profile, n_elements, ops,
            expected_rx_elements=np.arange(n_elements))

    def test_compliance_single_element_tx_aperture(self):
        """Tests linear array + LIN delays."""
        n_elements = 128
        device = DeviceMock(ProbeMock(
            ProbeModel(model_id=ProbeModelId("ultrasonix", "l14-5/38"),
                       pitch=0.3048e-3,
                       n_elements=n_elements,
                       curvature_radius=0.0)))

        n_elements = device.probe.model.n_elements
        aperture_size = 1

        ops = [
            TxRx(
                tx=Tx(
                    aperture=Aperture(center_element=ap_center_element, size=aperture_size),
                    excitation=self.default_excitation,
                    focus=0e-3,
                    angle=0,
                    speed_of_sound=1540,
                ),
                rx=Rx(
                    aperture=Aperture(center=0),
                    sample_range=self.default_sample_range,
                    downsampling_factor=1,
                ),
                pri=1,
            )
            for ap_center_element in np.arange(0, n_elements)
        ]
        input_sequence = TxRxSequence(ops=ops, tgc_curve=[])
        context = ContextMock(device=device,
                              medium=self.default_medium,
                              op=input_sequence)
        result = arrus.kernels.tx_rx_sequence.process_tx_rx_sequence(context)
        tx_rx_sequence = result.sequence
        ops = tx_rx_sequence.ops
        self.assertEqual(len(ops), n_elements)
        expected_delays_profile = np.array([0])
        self.assert_ok_for_moving_aperture(
            aperture_size, expected_delays_profile, n_elements, ops,
            expected_rx_elements=np.arange(n_elements))
#
    def test_compliance_convex_array_esaote_ac2541(self):
        """Tests convex array + STA delays."""
        n_elements = 192
        device = DeviceMock(ProbeMock(
            ProbeModel(model_id=ProbeModelId("esaote", "2541"),
                       pitch=0.3e-3,
                       n_elements=n_elements,
                       curvature_radius=50e-3)))

        n_elements = device.probe.model.n_elements
        aperture_size = 128
        aperture_center = np.arange(-15, 16, step=3)*1e-3

        ops = [
            TxRx(
                tx=Tx(
                    aperture=Aperture(center=center, size=aperture_size),
                    excitation=self.default_excitation,
                    focus=-10e-3,
                    angle=0,
                    speed_of_sound=1540,
                ),
                rx=Rx(
                    aperture=Aperture(center=center, size=aperture_size),
                    sample_range=self.default_sample_range,
                ),
                pri=1,
            )
            for center in aperture_center
        ]
        input_sequence = TxRxSequence(ops=ops, tgc_curve=[])
        context = ContextMock(device=device,
                              medium=self.default_medium,
                              op=input_sequence)

        result = arrus.kernels.tx_rx_sequence.process_tx_rx_sequence(context)
        tx_rx_sequence = result.sequence
        ops = tx_rx_sequence.ops
        self.assertEqual(len(ops), len(aperture_center))
        expected_delays_profile = np.array([6.277186e-06, 6.130122e-06, 5.983590e-06, 5.837612e-06, 5.692214e-06, 5.547423e-06, 5.403266e-06, 5.259772e-06, 5.116970e-06, 4.974892e-06, 4.833570e-06, 4.693038e-06, 4.553331e-06, 4.414486e-06, 4.276541e-06, 4.139537e-06, 4.003516e-06, 3.868520e-06, 3.734596e-06, 3.601792e-06, 3.470156e-06, 3.339741e-06, 3.210600e-06, 3.082790e-06, 2.956370e-06, 2.831401e-06, 2.707946e-06, 2.586072e-06, 2.465848e-06, 2.347345e-06, 2.230638e-06, 2.115805e-06, 2.002925e-06, 1.892081e-06, 1.783360e-06, 1.676850e-06, 1.572642e-06, 1.470831e-06, 1.371514e-06, 1.274789e-06, 1.180759e-06, 1.089527e-06, 1.001199e-06, 9.158811e-07, 8.336823e-07, 7.547116e-07, 6.790784e-07, 6.068922e-07, 5.382617e-07, 4.732946e-07, 4.120965e-07, 3.547708e-07, 3.014175e-07, 2.521329e-07, 2.070084e-07, 1.661303e-07, 1.295787e-07, 9.742690e-08, 6.974070e-08, 4.657782e-08, 2.798728e-08, 1.400893e-08, 4.673056e-09, 1.694066e-21, 1.694066e-21, 4.673056e-09, 1.400893e-08, 2.798728e-08, 4.657782e-08, 6.974070e-08, 9.742690e-08, 1.295787e-07, 1.661303e-07, 2.070084e-07, 2.521329e-07, 3.014175e-07, 3.547708e-07, 4.120965e-07, 4.732946e-07, 5.382617e-07, 6.068922e-07, 6.790784e-07, 7.547116e-07, 8.336823e-07, 9.158811e-07, 1.001199e-06, 1.089527e-06, 1.180759e-06, 1.274789e-06, 1.371514e-06, 1.470831e-06, 1.572642e-06, 1.676850e-06, 1.783360e-06, 1.892081e-06, 2.002925e-06, 2.115805e-06, 2.230638e-06, 2.347345e-06, 2.465848e-06, 2.586072e-06, 2.707946e-06, 2.831401e-06, 2.956370e-06, 3.082790e-06, 3.210600e-06, 3.339741e-06, 3.470156e-06, 3.601792e-06, 3.734596e-06, 3.868520e-06, 4.003516e-06, 4.139537e-06, 4.276541e-06, 4.414486e-06, 4.553331e-06, 4.693038e-06, 4.833570e-06, 4.974892e-06, 5.116970e-06, 5.259772e-06, 5.403266e-06, 5.547423e-06, 5.692214e-06, 5.837612e-06, 5.983590e-06, 6.130122e-06, 6.277186e-06])
        self.assert_ok_for_moving_aperture(aperture_size,
                                           expected_delays_profile,
                                           n_elements, ops,
                                           center_elements=np.arange(45, 146, 10))


if __name__ == "__main__":
    unittest.main()

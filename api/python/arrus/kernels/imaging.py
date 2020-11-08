import numpy as np
from arrus.ops.us4r import (
    Tx, Rx, TxRx, TxRxSequence, Pulse
)

class LinSequenceKernel:

    def process(self, op) -> TxRxSequence:
        # TODO implement
        pass



def LINSequence(context):
    """
    The function creates list of TxRx objects describing classic scheme

    :param context: KernelExecutionContext object
    """
    # device parameters
    n_elem = context.device.probe.n_elements
    pitch = context.device.probe.pitch
    
    # sequence parameters
    op = context.op
    n_elem_sub = op.tx_aperture_size
    focal_depth = op.tx_focus
    sample_range = op.sample_range
    pulse = op.pulse
    downsampling_factor = op.downsampling_factor
    pri = op.pri
    tx_ap_center_element = op.tx_aperture_center_element
    tx_ap_size = op.tx_aperture_size
    rx_ap_center_element = op.rx_aperture_center_element
    rx_ap_size = op.rx_aperture_size
    # TODO validate: check if aperture center element size is the same as center element
    # TODO validate check if aperture size is not greater than the number of probe elements

    # medium parameters
    c = context.medium.speed_of_sound

    # When the number of elements in subaperture is odd,
    #   the focal point is shifted to be above central element of subaperture
    if np.mod(n_elem_sub, 2):
        focus = [-pitch/2, focal_depth]
    else:
        focus = [0, focal_depth]

    # enumerate delays mask and padding for each txrx event
    subaperture_delays = enum_classic_delays(n_elem_sub, pitch, c, focus)

    # create txrx objects list
    def create_tx_rx(tx_center_element, rx_center_element, tx_size, rx_size):
        mask = ap_masks[i, :]
        padding = ap_padding[i]


    txrxlist = [create_tx_rx(i) for i in range(n_elem)]
    return TxRxSequence(txrxlist, tgc_curve=np.ndarray([]))


def create_tx_rx(tx_center_element, rx_center_element, tx_size, rx_size,
                 pulse, sample_range, downsampling_factor, pri):

    tx = Tx(tx_mask, pulse, delays)
    rx = Rx(rx_mask, sample_range, downsampling_factor, padding)
    return TxRx(tx, rx, pri)

def simple_aperture_scan(n_elem, subaperture_delays):
    """
    Function generates array which describes which elements should be turned
    on during classical scan.

    :param n_elem: number of elements in the aperture
    :param subaperture_delays: subaperture delays
    :return: tuple: masks(array of subsequent aperture masks, (n_lines, n_elem))
      , delays (array of subsequent aperture delays (n_lines, n_elem)), padding
    """
    subap_len = int(np.shape(subaperture_delays)[0])
    right_half = int(np.ceil(subap_len/2))
    left_half = int(np.floor(subap_len/2))
    ap_masks = np.zeros((n_elem, n_elem), dtype=bool)
    ap_delays = np.zeros((n_elem, n_elem))
    ap_padding = [None]*n_elem

    if np.mod(subap_len, 2):
        some_one = 0
    else:
        some_one = 1

    # TODO iteracja po elementach srodkowych
    for i_element in range(0, n_elem):
        # masks
        v_aperture = np.zeros(left_half + n_elem + right_half, dtype=bool)
        # TODO tutaj powinno byc liczone wzgledem elementu srodkowego (przesunac element srodkowy do uzycia w aperturze wirtualnej)
        v_aperture[i_element:i_element+subap_len] = True
        ap_masks[i_element, :] = v_aperture[left_half:-right_half]
        # padding
        left_padding = (left_half - i_element - some_one)*np.heaviside(left_half - i_element - some_one,0)
        right_padding = (i_element-n_elem + right_half + some_one)*np.heaviside(i_element-n_elem + right_half + some_one, 0)
        ap_padding[i_element] = (left_padding.astype(int), right_padding.astype(int))
        # delays
        v_delays = np.zeros(left_half + n_elem + right_half)
        v_delays[i_element:i_element+subap_len] = subaperture_delays
        ap_delays[i_element, :] = v_delays[(right_half-1):-left_half-1]
    return ap_masks, ap_delays, ap_padding


def enum_classic_delays(n_elem, pitch, c, focus):
    """
    The function enumerates classical focusing delays for linear array.
    It assumes that the 0 is in the center of the aperture.

    :param n_elem: number of elements in aperture,
    :param pitch: distance between two adjacent probe elements [m],
    :param c: speed of sound [m/s],
    :param focus: coordinates of the focal point ([xf, zf]),
      or focal length only (then xf = 0 is assumed) [m],
    :param delays: output delays vector.
    """
    if np.isscalar(focus):
        xf = 0
        zf = focus
    elif np.shape(focus) == (2,):
        xf, zf = focus
    else:
        raise ValueError("Bad focus - should be scalar, 1-dimensional ndarray, "
                         "or 2-dimensional ndarray")

    aperture_width = (n_elem-1)*pitch
    el_coord_x = np.linspace(-aperture_width/2, aperture_width/2, n_elem)
    element2focus_distance = np.sqrt((el_coord_x - xf)**2 + zf**2)
    dist_max = np.amax(element2focus_distance)
    delays = (dist_max - element2focus_distance)/c
    return delays

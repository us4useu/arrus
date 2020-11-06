import numpy as np

class LinSequenceKernel:

    def process(self, op) -> TxRxSequence:
        # TODO implement
        pass



def LINSequence(context):
    """
    The function creates list of TxRx objects describing classic scheme
    :paramt context: KernelExecutionContext object

    """

    # device parameters
    n_elem = context.device.probe.n_elements
    pitch = context.device.probe.pitch
    

    # sequence parameters
    n_elem_sub = context.op.tx_aperture_size
    focal_depth = context.op.tx_focus
    sample_range = context.op.sample_range
    pulse = context.op.pulse
    downsampling_factor = context.op.downsampling_factor
    
    # medium parameters
    c = context.medium.speed_of_sound


    # When the number of elements in subaperture is odd, \
    #   the focal point is shifted to be above central element of subaperture
    if np.mod(n_elem_sub,2):
        focus = [-pitch/2,focal_depth]
    else:
        focus = [0,focal_depth]

    # enumerate delays mask and padding for each txrx event
    subaperture_delays = enum_classic_delays(n_elem_sub, pitch, c, focus)
    ap_masks, ap_delays, ap_padding = simple_aperture_scan(n_elem, subaperture_delays)

    # create txrx objects list
    txrxlist = [None]* n_elem
    for i_elem in range(0,n_elem):
        this_mask = ap_masks[i_elem,:]
        this_del = ap_delays[i_elem,:]
        this_padding  = ap_padding[i_elem]

        tx = Tx(this_mask, pulse, this_del)
        rx = Rx(this_mask, sample_range, downsampling_factor, this_padding)
        txrx = TxRx(tx,rx,pri)
        txrxlist[i_elem] = txrx 

    return txrxlist


def simple_aperture_scan(n_elem, subaperture_delays):
    """    Function generate array which describes which elements are turn on during
    classic' txrx scheme scan. The supaberture step is eq
                
                :param n_elem: number   # medium parameters of elements in the array, i.e. full aperture length,
    :param subaperture_delays: subaperture length,
    :param ap_masks: output array of subsequent aperture masks
            (n_lines x n_elem size),
    :param ap_delays: output array of subsequent aperture delays
            (n_lines x n_elem size).
    """


    subap_len = int(np.shape(subaperture_delays)[0])
    right_half = int(np.ceil(subap_len/2))
    print(right_half)
    left_half = int(np.floor(subap_len/2))
    print(left_half)
    ap_masks = np.zeros((n_elem, n_elem), dtype=bool)
    ap_delays = np.zeros((n_elem, n_elem))
    ap_padding = [None]*n_elem

    if np.mod(subap_len,2):
        some_one = 0
    else:
        some_one = 1
        



    for i_element in range(0,n_elem):
        
        # masks
        v_aperture = np.zeros(left_half + n_elem + right_half, dtype=bool)
        v_aperture[i_element:i_element+subap_len] = 1
        ap_masks[i_element, :] = v_aperture[(right_half-1):-left_half-1]


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
        :param focus: coordinates of the focal point ([xf,zf]),
            or focal length only (then xf = 0 is assumed) [m],
        :param delays: output delays vector.
    """
    if np.isscalar(focus):
        xf = 0
        zf = focus
    elif np.shape(focus) == (2,):
        xf = focus[0]
        zf = focus[1]
    else:
        raise ValueError("Bad focus - should be scalar, 1-dimensional ndarray, or 2-dimensional ndarray")

    probe_width = (n_elem-1)*pitch
    el_coord_x = np.linspace(-probe_width/2, probe_width/2, n_elem)
    element2focus_distance = np.sqrt((el_coord_x - xf)**2 + zf**2)
    dist_max = np.amax(element2focus_distance)
    delays = (dist_max - element2focus_distance)/c
    return delays

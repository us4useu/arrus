import numpy as np


def LINSequence(n_elem, n_elem_sub, fc,pitch, c, focal_depth, sample_range,pri):


    if np.mod(n_elem,2):
        focus = [-pitch/2,focal_depth]
    else:
        focus = [0,focal_depth]




    subaperture_delays = enum_classic_delays(n_elem_sub, pitch, c, focus)
    ap_masks, ap_delays = simple_aperture_scan(n_elem, subaperture_delays)


    # 
    n_periods = 2
    pulse = Pulse(fc, n_periods)

    for i_elem in range(0,n_elem):
        this_ap = ap_masks[i_elem,:]
        this_del = ap_delays[i_elem,:]

        tx = Tx(this_ap, this_del, pulse)
        rx = Rx(this_ap, sample_range)
        txrx = TxRx(tx,rx,pri)
        txrxlist[i_elem] = txrx

    return txrxlist


def simple_aperture_scan(n_elem, subaperture_delays):
    """
    Function generate array which describes which elements are turn on during
    classic' txrx scheme scan. The supaberture step is equal 1.  
    
    :param n_elem: number of elements in the array, i.e. full aperture length,
    :param subaperture_delays: subaperture length,
    :param ap_masks: output array of subsequent aperture masks 
            (n_lines x n_elem size),
    :param ap_delays: output array of subsequent aperture delays
            (n_lines x n_elem size).
    """
    

    subap_len = int(np.shape(subaperture_delays)[0])
    big_half = int(np.ceil(subap_len/2))
    small_half = int(np.floor(subap_len/2))
    ap_masks = np.zeros((n_elem, n_elem), dtype=bool)
    ap_delays = np.zeros((n_elem, n_elem))
    for i_element in range(0,n_elem):
        # masks
        v_aperture = np.zeros(small_half + n_elem + big_half, dtype=bool)
        v_aperture[i_element:i_element+subap_len] = 1
        ap_masks[i_element, :] = v_aperture[(big_half-1):-small_half-1]

        # delays
        v_delays = np.zeros(small_half + n_elem + big_half)
        v_delays[i_element:i_element+subap_len] = subaperture_delays
        ap_delays[i_element, :] = v_delays[(big_half-1):-small_half-1]

    return ap_masks, ap_delays


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
        raise ValueError("Bad focus - should be scalar, 1-dimensional ndarray, or 2-dimensional ndarra")

    probe_width = (n_elem-1)*pitch
    el_coord_x = np.linspace(-probe_width/2, probe_width/2, n_elem)
    element2focus_distance = np.sqrt((el_coord_x - xf)**2 + zf**2)
    dist_max = np.amax(element2focus_distance)
    delays = (dist_max - element2focus_distance)/c
    return delays


# test code for checking if simple_aperture_scan() works correctly
#
# ap_masks, ap_delays = simple_aperture_scan(10, [1,2,3,4])
# print(ap_masks)
# print(ap_delays)

import numpy as np
import matplotlib.pyplot as plt

def get_lin_coords(nel=128, pitch=0.2*1e-3):
    '''
    Auxiliary tool for generating array transducer elements coordinates for linear array.

    :param nel: number of elements,
    :param pitch: distance between elements,
    :return: numpy array with elements coordinates (x,z)
    '''
    elx = np.linspace(-(nel-1)*pitch/2, (nel-1)*pitch/2, nel)
    elz = np.zeros(nel)
    coords = np.array(list(zip(elx,elz)))
    return coords


def gen_data(el_coords=None, dels=None, wire_coords=None,
             c=1540, fs=65e6, wire_amp=100):
    '''
    Function for generation of artificial non-beamformed data
    corresponding to single point (wire) within empty medium.

    :param el_coords: coordinates of transducer elements (numpy array),
    :param dels: initial delays,
    :param wire_coords: wire coordinates,
    :param c: speed of sound,
    :param fs: sampling frequency,
    :param wire_amp: amplitude of the wire
    :return: 2D numpy array of zeros and single pixel
             with amplitude equal to 'wire_amp' parameter.
    '''

    # check input and get default parameters if needed
    if el_coords is None:
        el_coords = get_lin_coords()

    nel, _  = np.shape(el_coords)
    if dels is None:
        dels = np.zeros(nel)

    if wire_coords is None:
        wire_coords = (0, 5*1e-3)

    # estimate distances between transducer elements and the 'wire'
    dist = np.zeros(nel)
    for i in range(nel):
        dist[i] = np.sqrt((el_coords[i, 0]-wire_coords[0])**2
                         +(el_coords[i, 1]-wire_coords[1])**2)
    # create output array
    nsamp = np.floor((2*dist/c + dels)*fs + 1).astype(int)
    nmax = 2*np.max(nsamp)
    data = np.zeros((nel,nmax))
    for i in range(nel):
        data[i, nsamp[i]-20:nsamp[i]+20] = wire_amp

    print(nsamp.shape)
    return data
#
if __name__ == "__main__":

    data = gen_data()
    nel, nsamp = np.shape(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data.T)

    aspect = 1/np.round(nsamp/nel).astype(int)
    ax.set_aspect(aspect)
    plt.show()


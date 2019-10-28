import scipy.io as sio
import scipy.signal as scs
import matplotlib.pyplot as plt
import numpy as np

def computeDelays(nSamples, nChannels,
                  fs, c,
                  focus, pitch,
                  depth0 = 0, delay0 = 17):

    """
    Computes delay matrix for given parameters.

    :param nSamples: number of samples to consider
    :param nChannels: number of channels to consider
    :param fs: transducer's sampling frequency [Hz]
    :param c: speed of sound [m/s]
    :param focus: if float or single element list focus is y focal coordinate.
                  If two element list, its [x,y] focal coordinates.
    :param pitch: transducer's pitch [m]
    :param depth0: the starting depth [m]
    :param delay0: hardware delay
    :return: A delay matrix (shape: nSamp x nChan) [samples]
    """

    # length of transducer
    transLen = (nChannels - 1)*pitch

    # check if focus is scalar or vector
    if type(focus) == float or len(focus) == 1:
        xFoc = transLen/2
        yFoc = focus
    else:
        xFoc = focus[0]
        yFoc = focus[1]

    # The distance from the line origin along y axis. [m]
    yGrid = np.arange(0, nSamples)/fs*c/2 + depth0

    # x coordinates of transducer elements
    elementPosition = np.arange(0, nChannels)*pitch

    # Make yGrid a column vector: (n_samples, 1).
    yGrid = yGrid.reshape((-1, 1))

    # Make element position a row vector: (1, n_channels).
    elementPosition = elementPosition.reshape((1, -1))

    # distances between elements and imaged point
    txDistance = yGrid
    rxDistance = np.sqrt((xFoc - elementPosition)**2 + yGrid**2)
    totalDistance = txDistance + rxDistance

    # delay related with focusing
    if xFoc > transLen/2:
        focDelay = ((xFoc**2 + yFoc**2)**0.5 - yFoc)/c
    else:
        focDelay = (((transLen-xFoc)**2 + yFoc**2)**0.5 - yFoc)/c

    pathDelay = (totalDistance/c)
    delays = pathDelay + focDelay
    delays = delays * fs + 1
    delays = np.round(delays)
    delays += delay0
    delays = delays.astype(int)

    return delays


def lineBeamform(rf, delays):
    """
    Beamforms one line  using delay and sum algorithm.

    :param rf: input RF data of shape (nSamples, nChannels, nLines)
    :param delays: delay matrix of shape (nSamples, nChannels)
    :return: beamformed single RF line
    """

    nSamples, nChannels = delays.shape
    theLine = np.zeros((nSamples))
    for channel in range(nChannels):
        channelDelays = delays[:, channel]
        theLine += rf[channelDelays, channel]

    return theLine

def imageBeamform(rf,delays):
    """
    Beamforms image usign lineBeamform function

    :param rf: input RF data of shape (nSamples, nChannels, nLines)
    :param delays: delay matrix of shape (nSamples, nChannels)
    :return: beamformed RF image
    """

    nSamples, nChannels = delays.shape
    nLines = rf.shape[2]
    image = np.zeros((nSamples, nLines))
    for iLine in range(nLines):
        lineRf = rf[:, iLine:(iLine+32), iLine]
        lineDelays = delays
        thisLine = lineBeamform(lineRf, lineDelays)
        # thisLine = np.abs(scs.hilbert(thisLine))
        image[:,iLine] = thisLine

    return image

def rf2env(rf):
    """
    The function calculate envelope using hilbert transform
    :param rf:
    :return: envelope image
    """
    nSamples, nLines = rf.shape
    env = np.zeros((nSamples, nLines))
    for iLine in range(nLines):
        thisRfLine = rf[:,iLine]
        thisEnvLine = np.abs(scs.hilbert(thisRfLine))
        env[:,iLine] = thisEnvLine

    return env

def rfDataReformat(rf, reformatType = 'pk'):
    """
    This function is for classical beamforming data acquired from simulation no.1,
    (Piotr Karwat).
    Rf data reformat from Piotr Karwat format to more convenient.

    :param rf:
    :return: rfRfrm
    """
    if reformatType == 'pk':
        rfTr = np.transpose(rf, (1, 0, 2))
        return rfTr
    else:
        print('bad reformat type')

def pkDataLoad(path2File, verbose=1):
    """
    This function is for classical beamforming data acquired from simulation no.1,
    (Piotr Karwat).
    The function loads the data and optionally write some info about the data

    :param rf:
    :param verbose = 1:
    :return: [rf, c, fs, fn, pitch, txFoc, txAp, nElem]
    """
    matData = sio.loadmat(path2File)
    c = matData.get('c0') * 1e-3
    c = np.float(c)

    txFoc = matData.get('txFoc') * 1e-3
    txFoc = np.float(txFoc)

    fs = matData.get('fs')
    fs = np.float(fs)

    fn = matData.get('fn')
    fn = np.float(fn)

    pitch = matData.get('pitch') * 1e-3
    pitch =np.float(pitch)

    nElem = matData.get('nElem')
    nElem = np.int(nElem)

    txAp = matData.get('txAp')
    txAp = np.int(txAp)

    rf = matData.get('rfBfr')

    if verbose:
        print('imput data keys: ', matData.keys())
        print('speed of sound: ', c)
        print('pitch: ', pitch)
        print('aperture length: ', nElem)
        print('focal length: ', txFoc)
        print('subaperture length: ', txAp)

    return [rf, c, fs, fn, pitch, txFoc, txAp, nElem]

def amp2dB(img):
    mx = np.max(img)
    img = np.log10(img/mx)*20
    return img

# path do file with the data
dataFile = '/home/linuser/us4us/usgData/rfBfr.mat'

# load and reformat data
[rf, c, fs, fn, pitch, txFoc, txAp, nElem] = pkDataLoad(dataFile, 0)
rf = rfDataReformat(rf, 'pk')

# compute delays
delays = computeDelays(4000, txAp, fs, c, [15*pitch, txFoc], pitch)

# image beamforming
rfBfr = imageBeamform(rf[:, :, 15:-17], delays)
# calculate envelope
img = rf2env(rfBfr)
# convert do dB
imgDB = amp2dB(img)

# show the image
plt.imshow(imgDB, interpolation = 'bicubic', aspect='auto',cmap='gray', vmin=-40, vmax=0)
plt.show()

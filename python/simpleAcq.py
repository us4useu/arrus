#!/usr/bin/python3
#==============================================================================
# simple_acq.py - a simple Python script for ultrasound data acquisition
# with the ARIUS device.
#------------------------------------------------------------------------------
# Copyright (c) 2019 us4us Ltd. / MIT License
#==============================================================================
import os, time
import numpy
import matplotlib.pyplot as plt
import arius as ar
from matplotlib.pyplot import ion

# Define constants.
NSAMPLES = 4096     # number of samples in each RF line
NCHANNELS = 192     # number of channels

print("Detected devices:")
for device in ar.getAvailableDevices():
    print("--- %s" % (device.name))

hal = ar.getHALInstance('MOCKHAL')

# Setup the platform.
with open("PA-BFR.json", "r") as f:
    hal.configure(f.read())

# Start TX/RX.
hal.start()
# Perform data acquisition and display.
for i in range(200):
    data, metadata = hal.getData()
    if i == 0:
        imgplot = plt.imshow(image, cmap=plt.cm.gray, aspect='auto')
        ion()
        plt.show()
    else:
        imgplot.set_data(image)
        plt.draw()
        plt.pause(0.01)
    hal.sync(metadata.frameIdx)
# Stop TX/RX.
hal.stop()

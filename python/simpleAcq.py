#!/usr/bin/python3
# =============================================================================
# simpleAcq.py - a simple Python script for ultrasound data acquisition
# with the ARIUS device.
# -----------------------------------------------------------------------------
# Copyright (c) 2019 us4us Ltd. / MIT License
# =============================================================================
import arius as ar
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIUS example - simple acq.")
    parser.add_argument("--cfg", dest="cfg",
                        help="Path to the json configuration file.",
                        required=True)
    parser.add_argument("--data", dest="data",
                        help="Path to the RF data to load.",
                        required=False,
                        default=None
                        )
    args = parser.parse_args()

    plt.ion()

    print("Detected devices:")
    for device in ar.getAvailableDevices():
        print("--- %s" % (device.name))

    # Load the input data.
    data = None
    if args.data:
        data = np.load(args.data)
    hal = ar.getHALInstance('MOCKHAL', data=data)
    # Setup the platform.
    with open(args.cfg, "r") as f:
        hal.configure(f.read())
        pass

    # Start TX/RX.
    hal.start()
    # Perform data acquisition and display.

    fig, ax, line = None, None, None # matplotlib objects.
    for i in range(10):
        data, metadata = hal.getData()

        # Plot acquired i-th line.
        if i == 0:
            fig, ax = plt.subplots()
            line, = ax.plot(data[0, :, i])
        else:
            line.set_ydata(data[0, :, i])
            fig.canvas.draw()
            fig.canvas.flush_events()
        time.sleep(0.5)

        hal.sync(metadata.frameIdx)
    # Stop TX/RX.
    hal.stop()

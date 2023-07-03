==============
Examples
==============

In the following parts of this chapter we will show you how to use 
the ARRUS software to communicate with the system in order to:

* program the transmission/reception (TX/RX) sequence and reconstruction,
* acquire raw RF data,
* acquire reconstructed B-mode data,
* obtain a real-time B-mode imaging,

The source code of the ready-to-run examples can be found in
|api_language|/examples directory:

* Us4R_control_bmodePwi - for B-Mode imaging using Plane Waves,
* Us4R_control_bmodeDwi - for B-Mode imaging using Diverging Waves,
* Us4R_control_bmodeLin - for B-Mode imaging using classical Line-by-line imaging,
* Us4R_control_colorPwi - for B-Mode & Color Doppler imaging using Plane Waves.

For more information on the parameters of individual functions please refer
to section :ref:`arrus-api`.

Operations
==========

In ARRUS, you can define and run the **operations** available for the
supported hardware. Transmission/reception (TX/RX) sequences and B-mode image
reconstruction are examples of such operations.

TX/RX sequence
~~~~~~~~~~~~~~

To define a TX/RX sequence, you should create an instance of :ref:`arrus.CustomTxRxSequence` class. 
The name-value input argument pairs allow you to control various aspects of TX/RX.

.. code-block:: matlab

    seq = CustomTxRxSequence('txApertureCenter', 0, ...
                             'txApertureSize',   128, ...
                             'rxApertureCenter', 0, ...
                             'rxApertureSize',   128, ...
                             'txFocus',          inf, ...
                             'txAngle',          (-20:10:20)*pi/180, ...
                             'speedOfSound',     1540, ...
                             'txFrequency',      6.5e6, ...
                             'txNPeriods',       2, ...
                             'txVoltage',        20, ...
                             'rxDepthRange',     50e-3, ...
                             ... % Optional parameters
                             'hwDdcEnable',      true, ...
                             'decimation',       10, ...
                             'nRepetitions',     1, ...
                             'txPri',            400e-6, ...
                             'tgcStart',         14, ...
                             'tgcSlope',         200, ...
                             'txInvert',         false );

The above example shows how to create the CustomTxRxSequence object with a complete set of 
input parameters. First 11 parameters are obligatory, others are optional. 

Interchangeability of parameters
````````````````````````````````

For your convenience:

* txApertureCenter [m] can be replaced with txCenterElement [elem],
* rxApertureCenter [m] can be replaced with rxCenterElement [elem],
* rxDepthRange [m] can be replaced with rxNSamples [samp].

Scalar/vector parameters, sequence length
`````````````````````````````````````````

The following parameters can be scalars or vectors:

* txApertureCenter or txCenterElement,
* txApertureSize,
* rxApertureCenter or rxCenterElement,
* txFocus,
* txAngle,
* txFrequency,
* txNPeriods,
* txInvert.

If any of them is a vector, then its length determines the number of TXs in the sequence. 
All the other parameters must be scalars or vectors of the same length. If any parameter 
is defined as a scalar, then it is assumed to be constant over the whole sequence. 
If all parameters are scalars, then the sequence contains a single TX/RX.

All the remaining parameters must be scalars, i.e. they are constant for every TX/RX.

Typical TX/RX strategies
````````````````````````

To program the typical TX waves:

* focused wave: set txFocus to positive finite values [m],
* diverging wave: set txFocus to negative finite values [m],
* plane wave: set txFocus to inf (as in the code example above).

To program the typical scanning strategies:

* phased scanning: set txAngle to a vector of scanning angles [rad] (as in the code example above),
* scanning TX aperture: set txApertureCenter [m] or txCenterElement [elem] to a vector of TX aperture positions,
* scanning RX aperture: set rxApertureCenter [m] or rxCenterElement [elem] to a vector of RX aperture positions,

Raw data format
```````````````

The collected raw data format depends on the hwDdcEnable setting:

* set hwDdcEnable to **false** to acquire the original raw RF data, 
* set hwDdcEnable to **true** to reduce the data stream, the collected data is in complex IQ format.

For more information, see the documentation of available :ref:`arrus-api-sequences`.

Reconstruction
~~~~~~~~~~~~~~

To define how to perform B-mode image reconstruction, you should create an instance of :ref:`arrus.Reconstruction` 
class. The name-value input argument pairs allow you to control various aspects of reconstruction.

.. code-block:: matlab

    rec = Reconstruction('xGrid',            (-20:0.10:20)*1e-3, ...
                         'zGrid',            (  0:0.10:50)*1e-3, ...
                         ... % Optional parameters
                         'bmodeRxTangLim',   [-0.5 0.5], ...
                         'rxApod',           hamming(10) ...
                         );

The xGrid and zGrid inputs define the reconstruction grid and thus they are obligatory. Other inputs are optional 
and allow you to set the size of the dynamic RX aperture (bmodeRxTangLim) and the RX apodization function (rxApod). 
There are many more optional inputs for setting the raw data filtration, reconstruction mode, Color Doppler, etc. 

Running operations in the system
=================================

First, you should create a handle to the system on which you want to perform operations. For example, to communicate 
with the Us4R system, create an instance of the Us4R class. You will need to indicate a prototxt config file 
containing the information on the probe, adapter, gains, etc. It is **extremly important** to make sure that the 
**system configuration agrees with the content of the config file**.

.. code-block:: matlab

    us  = Us4R('configFile', 'us4r.prototxt');

To run the TX/RX sequence and the reconstruction (optionally), upload them onto the system:

.. code-block:: matlab

    us.upload(seq, rec);

If you only want to run the uploaded operation once (for example, to acquire a single RF frame), 
use the ``run`` function. It will return the RF data (or IQ data if the hwDdcEnable is set to true) 
and the reconstructed image data if the reconstruction was uploaded together with the TX/RX sequence.

.. code-block:: matlab

    [rf,img] = us.run;

If you want to run the uploaded operation in a loop e.g. for real-time imaging, use the ``runLoop`` function together 
with a display-dedicated object. We prepared two classes of display objects: :ref:`arrus.BModeDisplay` and 
:ref:`arrus.DuplexDisplay` (for simultaneous display of B-mode and Color Doppler).

.. code-block:: matlab

    display = BModeDisplay(rec, 'dynamicRange', [0 80]);
    us.runLoop(@display.isOpen, @display.updateImg);

.. code-block:: matlab

    display = DuplexDisplay(rec, 'dynamicRange',    [0 80], ...
                                 'powerThreshold',  20);
    us.runLoop(@display.isOpen, @display.updateImg);

See the :ref:`arrus-Us4R` docs for more information.

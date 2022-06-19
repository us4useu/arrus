% Test script
%munlock("arrus_mex_object_wrapper");
clear all;
addpath("/home/pjarosik/src/arrus/build/api/matlab/wrappers");
addpath("/home/pjarosik/src/arrus/api/matlab/");
import arrus.ops.us4r.*;
import arrus.framework.*;

pulse = arrus.ops.us4r.Pulse('centerFrequency', 5e6, "nPeriods", 3.5);

seq = TxRxSequence( ...
    "ops", [ ...
      TxRx(...
        "tx", Tx("aperture", [true false true], 'delays', [0 1 2.5], "pulse", pulse), ...
        "rx", Rx("aperture", true(1, 16), "sampleRange", [0, 4096]), ...
        "pri", 100e-6), ...
       TxRx(...
         "tx", Tx("aperture", [true false false], 'delays', [0 1e-6 0.5e-6], "pulse", pulse), ...
         "rx", Rx("aperture", true(1, 8), "sampleRange", [0, 4096]), ...
         "pri", 100e-6)])

scheme = Scheme('txRxSequence', seq);

session = arrus.session.Session("/home/pjarosik/us4r.prototxt");
%disp("scheme");
%new_scheme.workMode
%new_scheme.outputBuffer
%new_seq = new_scheme.txRxSequence

%new_seq.ops(1).tx
%new_seq.ops(1).rx
%new_seq.ops(1).pri
%new_seq.ops(2).tx
%new_seq.ops(2).rx
%new_seq.ops(2).pri



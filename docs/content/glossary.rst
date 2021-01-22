Definitions:

A **frame** is an output of a single Tx/Rx operation.
An example of a single frame is a 2-D frame which will produce a single scanline for linear scanning scheme.

A **sequence of frames** (in short: **sequence**) is an output of a sequence of Tx/Rx operations.
An example of a sequence is 3-D array with shape (frame, sample, channel), from which a single b-mode image can be reconstructed.

A **batch of sequences** (in short: a **batch**) is a collection of multiple sequences.

An example of batch is a 4-D array with shape (sequence, frame, sample, channel), which can be used in a Doppler estimation methods.
nElem = 128;
txAp = true(1,nElem);
rxAp = true(1,nElem);
txDel = zeros(1,nElem);


tx = Tx('aperture', txAp, 'delay',txDel);
rx = Rx('aperture', rxAp);
txrx = TxRx('Tx', tx, ...
            'Rx', rx, ...
            'pri', 1e-6);
            
seq = TxRxSequence([txrx]);

%%
txAngle = [-10,0,5]*pi/180;
seq = TxRxSequence();


probe.nElem = 128;
probe.pitch = 0.3048;
seq.generate('scheme', 'pwi', 'probe', probe, 'c', 1490, 'txAngle', txAngle);

figure, hold on
    for k = 1:length(seq.TxRxList)
        plot(seq.TxRxList(k).Tx.delay)
    end

%%
x = [-5:1:5]*1e-3;
y = [0:1:5]*1e-3;
params = struct;
    params.txAngle = txAngle;
rec = Reconstructor('scheme','pwi', 'probe', probe, 'c', c,'params', params);
rec.setGrid(x)

rec.grid



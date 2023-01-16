% Script for testing the voltage stabilization for AN

% Use SL1543 probe with esaote3 adapter
% Make sure a proper prototxt is indicated (line 13)
% Select proper sample position using example script for rf acquisition
% Restart (also unplug/plug the power supply cable) the Us4R-lite before the measurement

%% Parameters
nChan   = 64;
nElem   = 192;
nSamp   = 2*1024;
nRep    = 1000;

pri     = [100 200 400]*1e-6;
nPer    = [2 4 8];
txAp    = [192 96 48];
v0      = [20 40 80];
vMultip = [1 0.5];

nPri    = numel(pri);
nPls    = numel(nPer);
nAp     = numel(txAp);
nV0     = numel(v0);
nVMul   = numel(vMultip);
nMeas   = nPri*nPls*nAp*nV0*nVMul;

samp    = nan(   1,nPri,nPls,nAp,nV0,nVMul);
amp     = nan(nRep,nPri,nPls,nAp,nV0,nVMul);
time    = nan(nRep,nPri,nPls,nAp,nV0,nVMul);

centElem = 32;
iSampAprox = 1500;

%% Arrus initialization
addpath("..\..\..\install\matlab");
addpath("..");
addpath("..\arrus");

import arrus.ops.us4r.*;

arrus.initialize("clogLevel", "INFO", "logFilePath", "C:/Temp/arrus.log", "logFileLevel", "TRACE");

%% Measurement loop
measId  = randperm(nMeas);
% measId  = 1:nMeas;

wb = waitbar(0,'Progress');
for iMeas=1:nMeas

    jMeas   = measId(iMeas);

    iPri    = mod(         jMeas-1                     , nPri) + 1;
    iPls    = mod( floor( (jMeas-1)/nPri              ), nPls) + 1;
    iAp     = mod( floor( (jMeas-1)/nPri/nPls         ), nAp ) + 1;
    iV0     = mod( floor( (jMeas-1)/nPri/nPls/nAp     ), nV0 ) + 1;
    iVMul   =      floor( (jMeas-1)/nPri/nPls/nAp/nV0 )        + 1;

    session = arrus.session.Session("C:\Users\pkarwat\Documents\GitHub\arrus\api\matlab\examples\us4r.prototxt");
    us4r = session.getDevice("/Us4R:0");
    
    txMask = abs((0:(nElem-1)) - (nElem-1)/2) <= txAp(iAp)/2;
    rxMask = [false(1,64), true(1,64), false(1,64)];
    nSubAp = 1;
    
    pulse = arrus.ops.us4r.Pulse('centerFrequency', 6e6, "nPeriods", nPer(iPls));
    
    seq = TxRxSequence( "ops", TxRx("tx", Tx("aperture", txMask, 'delays', zeros(1, nElem), "pulse", pulse), ...
                                    "rx", Rx("aperture", rxMask, "sampleRange", [0, nSamp]), ...
                                    "pri", pri(iPri)), "nRepeats", nRep, "tgcCurve", 14*ones(1,numel(400:150:nSamp)));
    
    scheme = Scheme('txRxSequence', seq, 'workMode', "MANUAL");
    buffer = session.upload(scheme);
    
    if 0
        us4r.setVoltage(v0(iV0)*vMultip(iVMul));
        disp(us4r.getRDAC());
    else
        % Setting initial voltage
        for v=1:v0(iV0)
            us4r.setRDAC(uint8(v*3));
            pause(0.1);
        end
        pause(1);
        
        % Decreasing voltage
        us4r.setRDAC(uint8(v0(iV0)*vMultip(iVMul)*3));
    end
    
    % Run
    session.run();
    array = buffer.front().eval();
    
    session.close();
    
    % Reordering data
    rf = zeros(nChan, nSamp, nRep, 'int16');
    for iRep=1:nRep
        for iSubAp=1:nSubAp
            for iOem=1:2
                chNumbers = (iSubAp-1)*64+(iOem-1)*32;
                frameNumber = nRep*nSubAp*(iOem-1) + (iRep-1)*nSubAp+iSubAp-1;
                rf(chNumbers+(1:32), :, iRep) = array(:, frameNumber*nSamp + (1:nSamp));
            end
        end
    end
    rf      = permute(rf,[2 1 3]);
    
    % Extracting the useful information
    env     = abs(hilbert(squeeze(rf(:,centElem,:))));
    [~,iSamp] = max( mean(env,2) .* (abs((1:nSamp).' - iSampAprox) <= 200) );
    
    samp(1,iPri,iPls,iAp,iV0,iVMul) = iSamp;
    amp (:,iPri,iPls,iAp,iV0,iVMul) = env(iSamp,:).';
    time(:,iPri,iPls,iAp,iV0,iVMul) = (1:nRep).'*pri(iPri)*nSubAp;
    
    waitbar(iMeas/nMeas,wb);
end
close(wb);

%% Save
save('testVoltage','samp','amp','time','pri','nPer','txAp','v0','vMultip');

%% Display v = const
figure;
for iV0=1:nV0
    subplot(1,nV0,iV0);
    plot(reshape(time(:,:,:,:,iV0,1),nRep,[]), ...
         reshape( amp(:,:,:,:,iV0,1),nRep,[]));
    grid on;
    set(gca,'XLim',[0 0.1]);
    set(gca,'YLim',[0 11e3]);
end

%% Display v = var
figure;
for iV0=1:nV0
    subplot(1,nV0,iV0);
    plot(reshape(time(:,:,:,:,iV0,2),nRep,[]), ...
         reshape( amp(:,:,:,:,iV0,2),nRep,[]));
    grid on;
    set(gca,'XLim',[0 0.1]);
    set(gca,'YLim',[0 11e3]);
end


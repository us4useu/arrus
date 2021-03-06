addpath('C:\Users\pjarosik\arrus-releases\ref-115\matlab');
addpath('C:\Users\pjarosik\src\arrus\arrus\api\matlab');
arrus.setConsoleLogger('DEBUG');

rxSettings = arrus.devices.us4r.RxSettings(...
    'dtgcAttenuation', 42,...
    'pgaGain', 24, ...
    'lnaGain', 24, ...
    'tgcSamples', [14, 15, 16], ...
    'lpfCutoff', 10e6, ...
    'activeTermination', 200 ...
);

channelMapping = 0:127;
groupsMask = cat(2, ones(1, 12), zeros(1, 4));

us4oemSettings = arrus.devices.us4r.Us4OEMSettings(channelMapping, groupsMask, rxSettings);
us4RSettings = arrus.devices.us4r.Us4RSettings({us4oemSettings});
sessionSettings = arrus.session.SessionSettings(us4RSettings);

arrus.session.Session(sessionSettings);
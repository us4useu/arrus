function us4R = Us4R(nModules, probeName, voltage, logTime)
    % Returns a handle to the Us4R system configured according to the
    % provided parameters.
    %
    % The returned object is of type :mat:class:`arrus.Us4RSystem`.
    %
    % :param nModules: number of Us4OEM modules available in the us4R system
    % :param probeName: name of the probe to use, available: \
    %    Esaote: 'AL2442', 'SL1543', \
    %    Ultrasonix: 'L14-5/38'
    % :param voltage: a voltage to set, should be in range 0-90 [0.5*Vpp]
    % :param logTime: set to true if you want to display acquisition \
    %    and reconstruction time (optional)
    
    if nargin < 4
        logTime = false;
    end
    
    probe = probeParams(probeName);
    if probe.adapType == 0
        us4R = Us4REsaote(nModules, probe, voltage, logTime);
    elseif probe.adapType == 1
        us4R = Us4RUltrasonix(nModules, probe, voltage, logTime);
    else
         error("ARRUS:IllegalArgument", ...
                        ['Unhandled probe adapter type for ', probeName])
    end
end


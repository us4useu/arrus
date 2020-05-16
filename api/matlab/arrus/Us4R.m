function us4R = Us4R(nArius, probeName, voltage, logTime)
    % Returns a handle to the Us4R system configured according to the
    % provided parameters.
    %
    % The returned object is of type :mat:class:`arrus.Us4RSystem`.
    %
    % :param nArius: number of arius modules available in the us4R system
    % :param probeName: name of the probe to use, available: \
    %    Esaote: 'AL2442', 'SL1543', \
    %    Ultrasonix: 'L14-5/38'
    % :param voltage: a voltage to set, should be in range 0-90 [0.5*Vpp]
    % :param logTime: set to true if you want to display acquisition \
    %    and reconstruction time (optional)
    
    probe = probeParams(probeName);
    if probe.adapType == 0
        us4R = Us4Esaote(nArius, probe, voltage, logTime);
    elseif probe.adapType == 1
        us4R = Us4Ultrasonix(nArius, probe, voltage, logTime);
    else
         error("ARRUS:IllegalArgument", ...
                        ['Unhandled probe adapter type for ', probeName])
    end
end


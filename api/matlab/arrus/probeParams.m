function probe = probeParams(probeName,adapterType,interfaceName)

%% Probe parameters
switch probeName
    case 'AL2442'
        probe.nElem	= 192;
        probe.pitch	= 0.21e-3;
        probe.maxVpp = 100; % max safe voltage peak-to-peak [Vpp]
        % probe channels, that should be always turned off.
        % Channel numeration starts from one!
        probe.channelsMask = [];

    case 'SL1543'
        probe.nElem	= 192;
        probe.pitch	= 0.245e-3;
        probe.maxVpp = 100;
        probe.channelsMask = [];

    case 'SP2430'
        probe.nElem	= 96;
        probe.pitch	= 0.22e-3;
        probe.probeMap = [1:48, 145:192];
        probe.maxVpp = 100;
        probe.channelsMask = [];

    case 'AC2541'
        probe.nElem	= 192;
        probe.pitch	= 0.30e-3;
        probe.curvRadius = -50e-3;
        probe.maxVpp = 100;
        probe.channelsMask = [];

    case 'L14-5/38'
        probe.nElem	= 128;
        probe.pitch	= 0.3048e-3;
        probe.maxVpp = 180;
        probe.channelsMask = [];

    case 'L7-4'
        probe.nElem	= 128;
        probe.pitch	= 0.298e-3;
        probe.maxVpp = 100;
        probe.channelsMask = [];

    case 'LA/20/128' % Vermon, linear, high frequency
        probe.nElem	= 128;
        probe.pitch	= 0.1e-3;
        probe.maxVpp = 30;
        probe.channelsMask = [];

    case '5L128' % Olympus NDT, linear
        probe.nElem	= 128;
        probe.pitch	= 0.6e-3;
        probe.maxVpp = 100;
        probe.channelsMask = [];

    case '10L128' % Olympus NDT, linear
        probe.nElem	= 128;
        probe.pitch	= 0.5e-3;
        probe.maxVpp = 100;
        probe.channelsMask = [];

    case '5L64' % Olympus NDT, linear
        probe.nElem	= 64;
        probe.pitch	= 0.6e-3;
        probe.maxVpp = 100;
        probe.channelsMask = [];

    case '10L32' % Olympus NDT, linear
        probe.nElem	= 32;
        probe.pitch	= 0.31e-3;
        probe.maxVpp = 100;
        probe.channelsMask = [];
        
        case '2.25L32' % Olympus NDT, linear
            probe.nElem	= 32;
            probe.pitch	= 1.00e-3;
            probe.maxVpp = 100;
            probe.channelsMask = [];
        case '5L32' % Olympus NDT, linear
            probe.nElem	= 32;
            probe.pitch	= 0.60e-3;
            probe.maxVpp = 100;
            probe.channelsMask = [];

    otherwise
        error(['Unhandled probe model ', probeName]);
        probe = [];
        return;

end

if ~isnumeric(probe.maxVpp) ...
        || ~isscalar(probe.maxVpp) ...
        || ~isfinite(probe.maxVpp) ...
        || ~(probe.maxVpp >= 0)
    error('Invalid maxVpp value. Must be nonnegative finite scalar value.')
end

if ~isfield(probe,'probeMap')
    probe.probeMap = 1:probe.nElem;
end

if ~isfield(probe,'curvRadius')
    probe.curvRadius = nan;
end

%% Adapter type & channel mapping
switch probeName
    case {'AL2442','SL1543','SP2430','AC2541','5L128','10L128','5L64','10L32','2.25L32','5L32'}
        if strcmp(adapterType, "esaote")
            probe.adapType      = 0;
            
            probe.rxChannelMap	= [32:-1:1; 1:1:32];
            probe.rxChannelMap(1,[16 17]) = [16 17];
            
            probe.txChannelMap	= [ probe.rxChannelMap +  0, ...
                                    probe.rxChannelMap + 32, ...
                                    probe.rxChannelMap + 64, ...
                                    probe.rxChannelMap + 96];
            probe.nUs4OEM = 2;
            probe.systemType = "us4rlite";
        elseif strcmp(adapterType, "esaote2")
            probe.adapType      = -1;
            
            probe.rxChannelMap	= [ 26  27  25  23  28  22  20  21  24  18  19  15  17  16  29  13 ...
                                    11  14  30   8  12   5  10   9  31   7   3   6   0   2   4   1 ...
                                    56  55  54  53  57  52  51  49  50  48  47  46  44  45  58  42 ...
                                    43  59  40  41  60  38  61  39  62  34  37  63  36  35  32  33 ...
                                    92  93  89  91  88  90  87  85  86  84  83  82  81  80  79  77 ...
                                    78  76  95  75  74  94  73  72  70  64  71  68  65  69  67  66; ...
                                     4   3   7   5   6   2   8   9   1  11   0  10  13  12  15  14 ...
                                    16  17  19  18  20  25  21  22  23  31  24  27  30  26  28  29 ...
                                    35  34  36  38  33  37  39  40  32  41  42  43  44  45  46  47 ...
                                    49  48  50  52  51  55  53  54  58  56  59  57  62  61  60  63 ...
                                    65  67  66  69  64  68  71  70  72  74  73  75  76  77  78  79 ...
                                    80  82  81  83  85  84  87  86  88  92  89  94  90  91  95  93] + 1;
            probe.rxChannelMap	= [probe.rxChannelMap, 128*ones(2,32)]; % channel map length must be = 128
            probe.txChannelMap	= probe.rxChannelMap;
            probe.nUs4OEM = 2;
            probe.systemType = "us4rlite";
        elseif strcmp(adapterType, "esaote3")
            probe.adapType      = 1;
            
            probe.rxChannelMap	= [1:1:32; 1:1:32];
            probe.txChannelMap	= [ probe.rxChannelMap +  0, ...
                                    probe.rxChannelMap + 32, ...
                                    probe.rxChannelMap + 64, ...
                                    probe.rxChannelMap + 96];
            probe.nUs4OEM = 2;     
            probe.systemType = "us4rlite";
        elseif strcmp(adapterType, "esaote2-us4r6")
            probe.adapType      = 3;
                                        % US4OEM 0
            probe.rxChannelMap	= [ 26  27  25  23  28  22  20  21  24  18  19  15  17  16  29  13  ...
                                    11  14  30   8  12   5  10   9  31   7   3   6   0   2   4   1; ...
                                    ... % US4OEM 1
                                    24  23  22  21  25  20  19  17  18  16  15  14  12  13  26  10  ...
                                    11  27   8   9  28   6  29   7  30   2   5  31   4   3   0   1; ...
                                    ... % Us4OEM 2
                                    28  29  25  27  24  26  23  21  22  20  19  18  17  16  15  13  ...
                                    14  12  31  11  10  30   9   8   6   0   7   4   1   5   3   2; ...
                                    ... % US4OEM 3
                                     4   3   7   5   6   2   8   9   1  11   0  10  13  12  15  14  ...
                                    16  17  19  18  20  25  21  22  23  31  24  27  30  26  28  29; ...
                                    ... % US4OEM 4
                                    3   2    4   6   1   5   7   8   0  9   10  11  12  13  14  15 ...
                                    17  16  18  20  19  23  21  22  26  24  27  25  30  29  28  31; ...
                                    ... % US4OEM 5
                                    1    3   2   5   0   4   7   6  8   10   9  11  12  13  14  15 ...
                                    16  18  17  19  21  20  23  22  24  28  25  30  26  27  31  29] + 1;
                                
            probe.rxChannelMap	= [probe.rxChannelMap, 128*ones(6, 96)]; % channel map length must be = 128
            probe.txChannelMap	= probe.rxChannelMap;
            probe.nUs4OEM = 6;
            probe.systemType = "us4r";
        elseif strcmp(adapterType, "esaote2-us4r8")
            probe.adapType      = 3;
                                       % US4OEM 0
            probe.rxChannelMap	= [ 26  27  25  23  28  22  20  21  24  18  19  15  17  16  29  13  ...
                                    11  14  30   8  12   5  10   9  31   7   3   6   0   2   4   1; ...
                                    ...% US4OEM 1
                                    24  23  22  21  25  20  19  17  18  16  15  14  12  13  26  10  ...
                                    11  27   8   9  28   6  29   7  30   2   5  31   4   3   0   1; ...
                                    ...% US4OEM 2
                                    28  29  25  27  24  26  23  21  22  20  19  18  17  16  15  13  ...
                                    14  12  31  11  10  30   9   8   6   0   7   4   1   5   3   2; ...
                                    ...% US4OEM 3
                                    1:32; % doesn't matter - not used
                                    ...% US4OEM 4
                                     4   3   7   5   6   2   8   9   1  11   0  10  13  12  15  14  ...
                                    16  17  19  18  20  25  21  22  23  31  24  27  30  26  28  29; ...
                                    ...% US4OEM 5
                                    3   2    4   6   1   5   7   8   0  9   10  11  12  13  14  15  ...
                                    17  16  18  20  19  23  21  22  26  24  27  25  30  29  28  31; ...
                                    ...% US4OEM 6
                                    1    3   2   5   0   4   7   6  8   10   9  11  12  13  14  15  ...
                                    16  18  17  19  21  20  23  22  24  28  25  30  26  27  31  29; ...
                                    ...% US4OEM 7
                                    1:32 % doesn't matter - not used
                                    ] + 1;
                                
            probe.rxChannelMap	= [probe.rxChannelMap, 128*ones(8, 96)]; % channel map length must be = 128
            probe.txChannelMap	= probe.rxChannelMap;
            probe.nUs4OEM = 8;
            probe.systemType = "us4r";
        else
            error(['No adapter of type ' adapterType ' available for the ' probeName ' probe.']);
        end
    
    case 'L14-5/38'
        if strcmp(adapterType, "ultrasonix")
            probe.adapType      = 2;
            
            probe.rxChannelMap	= [1:1:32; 32:-1:1];
            
            probe.txChannelMap	= [ probe.rxChannelMap +  0, ...
                                    probe.rxChannelMap + 32, ...
                                    probe.rxChannelMap + 64, ...
                                    probe.rxChannelMap + 96];
            probe.nUs4OEM = 2;
            probe.systemType = "us4rlite";
        else
            error(['No adapter of type ' adapterType ' available for the ' probeName ' probe.']);
        end
       
    case {'L7-4', 'LA/20/128'}
        if strcmp(adapterType, "atl/philips")
            probe.adapType      = 2;
            
            probe.rxChannelMap	= [32:-1:1 ; 1:1:32];
            probe.rxChannelMap(1,[16 17]) = [16 17];
            probe.rxChannelMap(2,[16 17]) = [17 16];
            
            probe.txChannelMap	= [ probe.rxChannelMap +  0, ...
                                    probe.rxChannelMap + 32, ...
                                    probe.rxChannelMap + 64, ...
                                    probe.rxChannelMap + 96];
            probe.nUs4OEM = 2;
            probe.systemType = "us4rlite";
        else
            error(['No adapter of type ' adapterType ' available for the ' probeName ' probe.']);
        end
end

%% Interface (NDT wedge) parameters
if nargin == 2 || strcmpi(interfaceName,'none')
    probe.interfEnable = false;
else
    % interface name should be validated here!
    
    % interface parameters for each interface name should be defined here!
    probe.interfEnable = true;
    probe.interfSize = 21.0e-3;
    probe.interfAng = 35.8*pi/180;
    probe.interfSos = 2320;

end


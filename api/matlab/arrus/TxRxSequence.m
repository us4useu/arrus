classdef TxRxSequence < handle
    % Class corresponding to sequence of Tx and Rx events
    %   
    %   properties: 
    %       TxRxList - object array of class TxRx
    %
    %   methods: 
    %       TxRxSequence() - constructor. 
    %       The input should be TxRx object array (TxRxList), or nothing.
    %       Example: TxRxSeqence() creates TxRxSequence with single TxRx
    %                event of empty Tx and Rx events.
    
    properties
        TxRxList {mustBeTxRx}
        scheme = 'custom';
        params = [];

        
    end
    
    methods
        function obj = TxRxSequence(varargin)                
            if nargin == 1
                obj.TxRxList = varargin{1};
                
            elseif nargin > 1 
                error('Too many arguments for TxRxSequence constructor.')

            end 

        end
        
        function obj = generate(obj, varargin)
        % The method generate sequence corresponding to given scheme.
        % The name-value convention is used. 
        % The followinf arguments should be given:
        %   'scheme' - the scheme name, 'pwi', 'sta', 'classic'
        %   'probe' - the structure with probe.nElem and probe.pitch fields
        %   'c' or 'sos' or 'speedOfSound' - the speed of sound [m/s]
        %
        % If the scheme is 'pwi' the following specific arguments should be given:
        %   'txAngles' - a vector of transmit angles in [rad].
        %   
            if nargin == 1
                disp('nargin == 1, should generate dafault scheme (todo)')
            end
            
            
            if nargin > 1
                
                scheme = findArg('scheme', varargin);
                probe = findArg('probe', varargin);
                c = findArg({'speedOfSound','c', 'sos'}, varargin);
                excitation = findArg('excitation', varargin);
                
                % checking if there are some nans in outputs of findArg()
                if isnumeric(c) && isscalar(c) && isnan(c)
                    error('The speed of sound is unknown.')
                elseif isnumeric(probe) && isscalar(probe) && isnan(probe)
                    error('The probe is unknown.')
%                 elseif isnumeric(excitation) && isscalar(excitation) && isnan(excitation)
%                     error('The excitation is unknown')
                end
                    
                    
                    
                switch scheme
                    case 'pwi'
                        disp('The scheme is: pwi')
                        disp('The probe is:'), disp(probe)
                        disp(['The assumed speed of sound is: ', num2str(c)])
                        
                        txAngle = findArg('txAngle', varargin);
                        obj.TxRxList = pwiKernel(txAngle, probe, c);
                        obj.scheme = scheme;
                        obj.params = struct;
                        obj.params.probe = probe;
                        obj.params.c = c;
                        
                    otherwise
                        disp('Unknown scheme.')
                        
                end
                        
                
                
                
            end
            
        end
        
    end
end

 
function mustBeTxRx(TxRxList)
    if ~isa(TxRxList,'TxRx') && ~isempty(TxRxList)
        error(['Bad TxRxSequence constructor input. ' ...
               'TxRxList must by the object from TxRx class.' ...
               ])
    end 
end


function argValue = findArg(argName, args)
% The function returns a value of searched argument in varargin cell assuming 
% name-value convention. It is possible to give argName as a cell with list
% of names - then if any of the names is found, the corresponding 
% argValue will be returned.

    vlen = length(args);
    if iscell(argName)
        nlen = length(argName);
    else
        nlen = 1;
        argName = {argName};
    end
    argValue = NaN;
    for k = 1:vlen
        for n = 1:nlen
            if isequal(args{k}, argName{n}) && vlen >= k+1
               argValue = args{k+1};
               return
            end            
            
        end

    end
end


function txrxlst = pwiKernel(txAngle, probe, c)
% The kernel function which generate list of TxRx objects for the
% TxRxSequence class. 
% This specific kernel generate TxRx objects for PWI scheme 
    aperture = true(1,probe.nElem);
    txrxlst = TxRx();

    for iAngle = 1:length(txAngle)
        angle = txAngle(iAngle);
        txdel = txDelay(angle, probe.nElem, probe.pitch, c);
        tx = Tx('aperture', aperture, 'delay', txdel);
        rx = Rx('aperture', aperture);
        txrx = TxRx('Tx', tx, 'Rx', rx);
        txrxlst(iAngle) = txrx;
    end

end

function txdel = txDelay(angle, nElem, pitch, c)
% The function enumearte tx delays for pwiKernel.
    txdel = zeros(1, nElem);

    for i=1:nElem
        if angle>=0
            n = 1;
        else
            n = nElem;
        end
        txdel(i) = (i-n)*pitch*tan(angle)/c;
    end

end
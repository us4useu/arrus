classdef TxRxScheme
    % The class describes the transmit-receive scheme (probably everything
    % will be implemented into TxRxSequence class (generate() method)
    
    properties
        name (1,:) 
        probe
        c (1,1)
        
        
    end
    
    methods
        function obj = TxRxScheme(varargin)

            
            name = findArg('name', varargin);
            probe = findArg('probe', varargin);
            c = findArg({'speedOfSound','c', 'sos'}, varargin);
            
            nameVld = @(x) ismember(x,{'pwi', 'sta', 'classic'});
            probeVld
            cVld
            
            if isnumeric(c) && isscalar(c) && isnan(c)
                error('The speed of sound is unknown.')
                
            elseif isnumeric(probe) && isscalar(probe) && isnan(probe)
                error('The probe is unknown.')
                
            elseif ~nameVld(name)
                error('Unknown scheme.')
                
            end            
            
            obj.name = name;
            obj.probe = probe;
            obj.c = c;
            
            switch name
                case 'pwi'
                    
                case 'sta'
                    
                case 'classic'
                    
                otherwise
                    
            end
                    

            
        end
        
        
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

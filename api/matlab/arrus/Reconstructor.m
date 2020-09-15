classdef Reconstructor < handle
    % Class for us image reconstruction
    %   
    %   properties: 
    %       scheme - scheme name, 'pwi', 'classic', 'sta', 'custom'
    %       probe - the structure with probe.nElem and probe.pitch fields
    %       c - speed of sound [m/s]
    %       params - structure with parameters specific for given scheme 
    %               (e.g. params.txAngle)
    %       grid - describes grid. The structure with following fields: 
    %               q1, q2, system ('cartesian', 'polar', 'custom')
    %               
    %
    %   methods: 
    %       Reconstructor() - constructor
    %       setGrid() - creates grid
    %
    
    properties
        scheme = [];
        probe = [];
        c = [];
        params = [];
        grid = [];

        
    end
    
    methods
        function obj = Reconstructor(varargin)
            p = inputParser;
            cVld = @(x) isnumeric(x) ...
                     && isscalar(x) ...
                     && isfinite(x) ...
                     && isreal(x) ...
                     && x > 0 ...
                     ;
            probeVld = @(x) isstruct(x) ...
                         && isfield(x,'nElem') ...
                         && isfield(x,'pitch') ...
                         ;
            schemeVld = @(x) ismember(x,{'pwi', 'sta', 'classic', 'custom'});
            paramsVld = @(x) isstruct(x);
            gridVld = @(x) isstruct(x) ...
                        && isfield(x,'system') ...
                        && isfield(x, 'q1') ...
                        && isfield(x, 'q2') ...
                        ;
            addParameter(p, 'scheme',[], schemeVld)
            addParameter(p, 'probe',[], probeVld)
            addParameter(p, 'c',[], cVld)
            addParameter(p, 'params',[], paramsVld)
            addParameter(p, 'grid',[], gridVld)
            
            parse(p,varargin{:})
            obj.scheme = p.Results.scheme;
            obj.probe  = p.Results.probe;
            obj.c      = p.Results.c;
            obj.params = p.Results.params;
            obj.grid   = p.Results.grid;
            
        end
        
        function obj = setGrid(obj,varargin)
            % The method for creating image grid.
            
            if nargin == 1
                q1 = [];
                q2 = [];
                system = 'cartesian';
            end
            
           if nargin == 2
                q1 = varargin{1};
                q2 = varargin{1};
                system = 'cartesian';
           end

             
           if nargin == 3
                q1 = varargin{1};
                q2 = varargin{2};
                system = 'cartesian';
           end
            
           if nargin == 4
                q1 = varargin{1};
                q2 = varargin{2};
                system = varargin{3};
           end
           
           grid.q1 = q1;
           grid.q2 = q2;
           grid.system = system;
           obj.grid = grid;
            
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
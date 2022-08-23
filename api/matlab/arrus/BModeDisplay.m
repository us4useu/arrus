classdef BModeDisplay < handle
    % A B-mode image display.
    %
    % Currently is implemented as a simple MATLAB figure with dynamically
    % updated content.
    % 
    % :param reconstructionObject: object of class "Reconstruction" \
    %   (obligatory input unless old syntax is used)
    % :param dynamicRange: two-element vector [min, max], dynamic range \
    %   limits to apply (optional name-value argument)
    % :param cineLoopLength: positive scalar, number of frames stored in \
    %   cineloop (optional name-value argument)
    % :param persistence: persistence filtration weights (vector) or \
    %   length (scalar) (optional name-value argument)
    % :param bmodeTgc: applies linear TGC from 0 at top to bmodeTgc[dB] \
    %   at bottom of the image (scalar) (optional name-value argument)
    % :param bmodeAutoTgcResp: applies linear TGC that adapts in time to \
    %   the imaging conditions, defines the responsiveness of the \
    %   adaptation (0-no adaptation, 1-instant adaptation), \
    %   <0,1> (scalar) (optional name-value argument)
    % 
    % Old syntax (won't be supported in future releases):
    % :param xGrid: (1, width) vector, x-coordinates of the image pixels [m]
    % :param zGrid: (1, depth) vector z-coordinates of the image pixels [m]
    % :param dynamicRange: two-element vector [min, max], value lims to apply
    % 
    % BModeDisplay class includes a cineloop buffer. The content of the \
    % buffer can be accessed through the getCineLoop method.
    
    properties(Access = private)
        hFig
        hImg
        xGrid
        zGrid
        cineLoop
        cineLoopLength = 1
        cineLoopIndex
        persistence = 1
        bmodeTgc = 0
        bmodeAutoTgcResp = 0
    end
    
    methods 
        function obj = BModeDisplay(varargin)
            
            % Input parser
            if ~isa(varargin{1},'Reconstruction')
                obj.xGrid = varargin{1};
                obj.zGrid = varargin{2};
                if numel(varargin) < 3
                    dynamicRange = [20 80];
                else
                    dynamicRange = varargin{3};
                    if ~all(size(dynamicRange) == [1 2])
                        error("ARRUS:IllegalArgument", ...
                            "Invalid dimensions of dynamic range vector, should be: [min max]")
                    end
                end
            else
                dispParParser = inputParser;
                
                addRequired(dispParParser, 'reconstructionObject', ...
                    @(x) assert(isscalar(x) && isa(x,'Reconstruction'), ...
                    "reconstructionObject is an obligatory scalar input of class Reconstruction."));
                
                addParameter(dispParParser, 'dynamicRange', [0 80], ...
                    @(x) assert(isnumeric(x) && all(size(x) == [1 2]), ...
                    "dynamicRange must be a 2-element numerical vector: [min max]."));
                
                addParameter(dispParParser, 'cineLoopLength', 1, ...
                    @(x) assert(isnumeric(x) && isscalar(x) && mod(x,1)==0 && x>=1, ...
                    "cineLoopLength must be a positive integer scalar."));
                
                addParameter(dispParParser, 'persistence', 1, ...
                    @(x) assert(isnumeric(x) && ((isvector(x) && numel(x)>1) || ...
                    (isscalar(x) && mod(x,1)==0 && x>=1)), ...
                    "persistence must be a positive integer scalar or numerical vector."));
                
                addParameter(dispParParser, 'bmodeTgc', 0, ...
                    @(x) assert(isnumeric(x) && isscalar(x), ...
                    "bmodeTgc must be a numerical scalar."));
                
                addParameter(dispParParser, 'bmodeAutoTgcResp', 0, ...
                    @(x) assert(isnumeric(x) && isscalar(x) && x>=0 && x<=1, ...
                    "bmodeAutoTgcResp must be a numerical scalar in <0,1> range."));
                
                parse(dispParParser, varargin{:});
                
                proc             = dispParParser.Results.reconstructionObject;
                dynamicRange     = dispParParser.Results.dynamicRange;
                cineLoopLength   = dispParParser.Results.cineLoopLength; %#ok<*PROP>
                persistence      = dispParParser.Results.persistence;
                bmodeTgc         = dispParParser.Results.bmodeTgc;
                bmodeAutoTgcResp = dispParParser.Results.bmodeAutoTgcResp;
                
                if isscalar(persistence)
                    persistence = ones(1,persistence);
                end
                
                if numel(persistence)>cineLoopLength
                    warning("cineLoopLength increased to fit the persistence.");
                    cineLoopLength = numel(persistence);
                end
                
                obj.xGrid = proc.xGrid;
                obj.zGrid = proc.zGrid;
                obj.cineLoopLength = cineLoopLength;
                obj.persistence = reshape(persistence,1,1,[]) / sum(persistence);
                obj.bmodeTgc = bmodeTgc * obj.zGrid(:) / obj.zGrid(end);
                obj.bmodeAutoTgcResp = bmodeAutoTgcResp;
            end
            
            % Prepare cineLoop
            obj.cineLoop = nan(numel(obj.zGrid), numel(obj.xGrid), obj.cineLoopLength);
            obj.cineLoopIndex = 0;
            
            % Create figure.
            obj.hFig = figure();
            obj.hImg = imagesc(obj.xGrid*1e3, obj.zGrid*1e3, []);
            xlabel('x [mm]');
            ylabel('z [mm]');
            daspect([1 1 1]);
            set(gca, 'CLim', dynamicRange);
            set(gca, 'XLim', obj.xGrid([1 end])*1e3);
            set(gca, 'YLim', obj.zGrid([1 end])*1e3);
            colormap(gray);
            colorbar;
        end
        
        function state = isOpen(obj)
            % Checks if the window was not already closed.
            %
            % :return: true when the windows with b-mode image was not closed,
            %          false otherwise
            state = ishghandle(obj.hFig);
        end
        
        function updateImg(obj, data)
            % Updates currently displayed image.
            %
            % :param data: bmode data to display
            try
                % update cineLoop
                obj.cineLoopIndex = mod(obj.cineLoopIndex, obj.cineLoopLength) + 1;
                obj.cineLoop(:,:,obj.cineLoopIndex) = data;
                
                % persistence
                index = mod(obj.cineLoopIndex - (0:(numel(obj.persistence) - 1)) - 1, obj.cineLoopLength) + 1;
                bmode = sum(obj.cineLoop(:,:,index) .* obj.persistence, 3, 'omitnan');
                
                % time gain compensation
                if obj.bmodeAutoTgcResp ~= 0
                    % linear regression of bmode average brightness profile
                    n = numel(obj.zGrid);
                    x = obj.zGrid(:);
                    y = mean(bmode,2,'omitnan');
                    a = (n*sum(x.*y) - sum(x)*sum(y)) / (n*sum(x.^2) - sum(x)^2); % [dB/m]
                    
                    obj.bmodeTgc = obj.bmodeTgc * (1 - obj.bmodeAutoTgcResp) + ...
                                  (-a * obj.zGrid(:)) * obj.bmodeAutoTgcResp;
                end
                bmode = bmode + obj.bmodeTgc;
                
                set(obj.hImg, 'CData', bmode);
                
                % TODO removed below pause
                % Applied the pause to make the figure window more
                % responsive. Removing this pause may introduce some
                % issues when closing the figure - e.g. a long delay
                % between pressing the window close button and the 
                % reaction to that close.
                % That was an issue on MATLAB 2018b, testenv2.
                pause(0.01);
                
            catch ME
                if(strcmp(ME.identifier, 'MATLAB:class:InvalidHandle'))
                    disp('Display was closed.');
                else
                    rethrow(ME);
                end
            end
        end
        
        function cineLoop = getCineLoop(obj)
            if obj.cineLoopLength > 0
                cineLoop = circshift(obj.cineLoop, -obj.cineLoopIndex, 3);
            else
                cineLoop = [];
            end
        end
        
    end    
end


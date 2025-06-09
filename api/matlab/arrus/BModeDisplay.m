classdef BModeDisplay < handle
    % B-mode image display class.
    % 
    % Currently implemented as a simple MATLAB figure with dynamically \
    % updated content.
    % 
    % Includes a cineloop buffer. The content of the buffer can be \
    % accessed through the getCineLoop method.
    
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
            % Creates a BModeDisplay object.
            % 
            % Syntax:
            % obj = BModeDisplay(reconstructionObject, name, value, ..., name, value)
            % 
            % First input is obligatory, further inputs (name-value pairs) \
            % are optional.
            % 
            % :param reconstructionObject: Object of class "Reconstruction". \
            %   Obligatory input.
            % :param dynamicRange: Dynamic range limits [dB]. \
            %   Two-element vector [min, max]. Optional name-value argument, \
            %   default = [0 80].
            % :param cineLoopLength: Cineloop buffer size (number of frames \
            %   stored in the buffer). Positive scalar. Optional name-value \
            %   argument, default = 1.
            % :param persistence: If given, enables persistence. If scalar, \
            %   it defines the persistence filter length. If vector, it \
            %   defines the persistence filter coefficients. \
            %   Optional name-value argument, default = 1 (no persistence).
            % :param bmodeTgc: If given, enables linear TGC from 0 dB \
            %   at the top to the given value [dB] at the bottom of the \
            %   image. Scalar. Optional name-value argument, default = 0.
            % :param bmodeAutoTgcResp: If given, enables linear TGC \
            %   that adapts in time to the imaging conditions. The \
            %   responsiveness of the adaptation: 0-no adaptation, \
            %   1-instant adaptation. Scalar, in the range <0,1>. \
            %   Optional name-value argument, default = 0.
            % 
            % :return: BModeDisplay object.
            
            % Input parser
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
            % Checks if the display window is open.
            %
            % :return: True if the display window was not closed, \
            %          false otherwise.

            state = ishghandle(obj.hFig);
        end
        
        function updateImg(obj, data)
            % Updates currently displayed image.
            %
            % :param data: B-mode data to be displayed.

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
                drawnow limitrate;
                
            catch ME
                if(strcmp(ME.identifier, 'MATLAB:class:InvalidHandle'))
                    disp('Display was closed.');
                else
                    rethrow(ME);
                end
            end
        end
        
        function cineLoop = getCineLoop(obj)
            % Returns the cineloop buffer.
            % 
            % :return: Cineloop buffer.
            
            if obj.cineLoopLength > 0
                cineLoop = circshift(obj.cineLoop, -obj.cineLoopIndex, 3);
            else
                cineLoop = [];
            end
        end
        
    end    
end


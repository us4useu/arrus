classdef DuplexDisplay < handle
    % Duplex image display class.
    % 
    % Currently implemented as a simple MATLAB figure with dynamically \
    % updated content.
    % 
    % Includes a cineloop buffer. The content of the buffer can be \
    % accessed through the getCineLoop method.
    
    properties(Access = private)
        hFig
        hAx
        hImg
        vector
        xGrid
        zGrid
        colorEnable
        vectorEnable
        dynamicRange
        powerThreshold
        turbuThreshold
        stdevThreshold
        smoothKernel
        cineLoop
        cineLoopLength
        cineLoopIndex
        persistence
        bmodeTgc
        bmodeAutoTgcResp
    end
    
    methods 
        function obj = DuplexDisplay(varargin)
            % Creates a DuplexDisplay object.
            % 
            % Syntax:
            % obj = DuplexDisplay(reconstructionObject, name, value, ..., name, value)
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
            % :param powerThreshold: Power limit [dB] BELOW which the color \
            %   data is not displayed. Scalar. Optional name-value argument, \
            %   default = -inf (no thresholding).
            % :param turbuThreshold: Turbulence limit ABOVE which the color \
            %   data is not displayed. Scalar. Optional name-value argument, \
            %   default = 1 (no thresholding, as turbulence is limited to \
            %   0-1 range by definition).
            % :param stdevThreshold: Limit of standard deviation of color \
            %   data ABOVE which the color data is not displayed. Scalar. \
            %   Optional name-value argument, default = inf (no thresholding).
            % :param thresholdSmooth: Size [pix] of the circular smoothing \
            %   kernel applied to the data that are used for thresholding, \
            %   i.e. power, turbulence, and st. deviation. Scalar. \
            %   Optional name-value argument, default = 0 (no smoothing).
            % :param subplotEnable: Enables separate display of Duplex, \
            %   B-mode, power, and turbulence images. Logical scalar. \
            %   Optional name-value argument, default = false.
            % 
            % :return: DuplexDisplay object.
            
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
            
            addParameter(dispParParser, 'powerThreshold', -inf, ...
                        @(x) assert(isnumeric(x) && isscalar(x), ...
                        "powerThreshold must be a numerical scalar."));
            
            addParameter(dispParParser, 'turbuThreshold', 1, ...
                        @(x) assert(isnumeric(x) && isscalar(x), ...
                        "turbuThreshold must be a numerical scalar."));
            
            addParameter(dispParParser, 'stdevThreshold', inf, ...
                        @(x) assert(isnumeric(x) && isscalar(x), ...
                        "stdevThreshold must be a numerical scalar."));
            
            addParameter(dispParParser, 'thresholdSmooth', 0, ...
                        @(x) assert(isnumeric(x) && isscalar(x), ...
                        "thresholdSmooth must be a numerical scalar."));
            
            addParameter(dispParParser, 'subplotEnable', false, ...
                        @(x) assert(islogical(x) && isscalar(x), ...
                        "subplotEnable must be a logical scalar."));
            
            parse(dispParParser, varargin{:});
            
            proc             = dispParParser.Results.reconstructionObject;
            dynamicRange     = dispParParser.Results.dynamicRange;
            cineLoopLength   = dispParParser.Results.cineLoopLength;
            persistence      = dispParParser.Results.persistence;
            bmodeTgc         = dispParParser.Results.bmodeTgc;
            bmodeAutoTgcResp = dispParParser.Results.bmodeAutoTgcResp;
            powerThreshold   = dispParParser.Results.powerThreshold;
            turbuThreshold   = dispParParser.Results.turbuThreshold;
            stdevThreshold   = dispParParser.Results.stdevThreshold;
            thresholdSmooth  = dispParParser.Results.thresholdSmooth;
            subplotEnable    = dispParParser.Results.subplotEnable;
            
            if isscalar(persistence)
                persistence = ones(1,persistence);
            end
            
            if numel(persistence)>cineLoopLength
                warning("cineLoopLength increased to fit the persistence.");
                cineLoopLength = numel(persistence);
            end
            
            if thresholdSmooth>0
                aux = linspace(-1,1,thresholdSmooth).^2;
                smoothKernel = double(sqrt(aux.' + aux) <= 1);
            else
                smoothKernel = [];
            end
            
            obj.xGrid = proc.xGrid;
            obj.zGrid = proc.zGrid;
            obj.colorEnable = proc.colorEnable;
            obj.vectorEnable = proc.vectorEnable;
            obj.dynamicRange = dynamicRange;
            obj.powerThreshold = powerThreshold;
            obj.turbuThreshold = turbuThreshold;
            obj.stdevThreshold = stdevThreshold;
            obj.smoothKernel = smoothKernel;
            obj.cineLoopLength = cineLoopLength;
            obj.persistence = reshape(persistence,1,1,[]) / sum(persistence);
            obj.bmodeTgc = bmodeTgc * obj.zGrid(:) / obj.zGrid(end);
            obj.bmodeAutoTgcResp = bmodeAutoTgcResp;
            
            % Prepare cineLoop
            cineLoopLayersNumber = 1 + 3*double(obj.colorEnable && ~obj.vectorEnable) + 4*double(obj.vectorEnable);
            obj.cineLoop = nan(numel(obj.zGrid), numel(obj.xGrid), obj.cineLoopLength, cineLoopLayersNumber);
            obj.cineLoopIndex = 0;
            
            % Create figure.
            obj.hFig = figure();
            if subplotEnable && (obj.colorEnable || obj.vectorEnable)
                subplotColorMaps = cat(3, gray, gray, hot, hot);
                subplotDynRange = [dynamicRange; dynamicRange; dynamicRange; [0 1]];
                
                for iAx=1:4
                    obj.hAx(iAx) = subplot(2,2,iAx);
                    obj.hImg(iAx) = imagesc(obj.xGrid*1e3, obj.zGrid*1e3, []);
                    colormap(gca,subplotColorMaps(:,:,iAx));
                    colorbar;
                    xlabel('x [mm]');
                    ylabel('z [mm]');
                    daspect([1 1 1]);
                    set(gca, 'CLim', subplotDynRange(iAx,:));
                    set(gca, 'XLim', obj.xGrid([1 end])*1e3);
                    set(gca, 'YLim', obj.zGrid([1 end])*1e3);
                end
            else
                obj.hAx = axes();
                obj.hImg = imagesc(obj.xGrid*1e3, obj.zGrid*1e3, []);
                colormap(gca,gray);
                colorbar;
                xlabel('x [mm]');
                ylabel('z [mm]');
                daspect([1 1 1]);
                set(gca, 'CLim', dynamicRange);
                set(gca, 'XLim', obj.xGrid([1 end])*1e3);
                set(gca, 'YLim', obj.zGrid([1 end])*1e3);
            end
            
            if obj.vectorEnable
                vecDec	= 10;

                obj.vector.xSel = vecDec:vecDec:(numel(obj.xGrid)-vecDec+1);
                obj.vector.zSel = vecDec:vecDec:(numel(obj.zGrid)-vecDec+1);
                obj.vector.scale = vecDec*diff(obj.xGrid(1:2)*1e3)/pi/2;

                axes(obj.hAx(1)), hold on;
                set(gca,'YDir','reverse');
                set(gca,'XLim',obj.xGrid([1 end])*1e3);
                set(gca,'YLim',obj.zGrid([1 end])*1e3);
                
                obj.vector.hQvr = quiver(obj.xGrid(obj.vector.xSel)*1e3, ...
                                         obj.zGrid(obj.vector.zSel)*1e3, ...
                                         zeros(numel(obj.vector.zSel),numel(obj.vector.xSel)), ...
                                         zeros(numel(obj.vector.zSel),numel(obj.vector.xSel)), 0, 'Color', 'm', 'LineWidth', 1.5);
            end
            
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
            % :param data: Duplex data to be displayed.

            try
                [nZPix,nXPix,~,~] = size(data);
                
                % update cineLoop
                obj.cineLoopIndex = mod(obj.cineLoopIndex, obj.cineLoopLength) + 1;
                obj.cineLoop(:,:,obj.cineLoopIndex,:) = data;
                
                % persistence
                index = mod(obj.cineLoopIndex - (0:(numel(obj.persistence) - 1)) - 1, obj.cineLoopLength) + 1;
                bmode = sum(obj.cineLoop(:,:,index,1) .* obj.persistence, 3, 'omitnan');
                
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
                
                % conversion to RGB
                bmode(isnan(bmode)) = -inf;
                bmodeRGB = (bmode - obj.dynamicRange(1)) / diff(obj.dynamicRange);
                bmodeRGB = max(0,min(1,bmodeRGB));
                bmodeRGB = bmodeRGB .* ones(1,1,3);   % colormap = gray
                
                if obj.colorEnable || obj.vectorEnable
                    power = data(:,:,:,2);
                    turbu = data(:,:,:,3);
                    powerMap = hot(128);
                    turbuMap = hot(128);
                    if obj.colorEnable
                        color = data(:,:,:,4);
                        colorMap = jet(128);
                        colorOffset = 0.5;
                    elseif obj.vectorEnable
                        colorX = data(:,:,:,4);
                        colorZ = data(:,:,:,5);
                        color = sqrt(colorX.^2 + colorZ.^2);
                        colorMap = hot(128);
                        colorOffset = 0;
                    end

                    msk = ~isnan(color);
                    
                    color(~msk) = 0;
                    power(~msk) = -realmax(class(power));
                    turbu(~msk) = 1;
                    
                    colorRGB = color/2/pi + colorOffset;
                    colorRGB = max(0,min(1,colorRGB));
                    colorRGB = reshape(colorMap(1+round(colorRGB*127),:),nZPix,nXPix,3);
                    
                    powerRGB = (power - obj.dynamicRange(1)) / diff(obj.dynamicRange);
                    powerRGB = max(0,min(1,powerRGB));
                    powerRGB = reshape(powerMap(1+round(powerRGB*127),:),nZPix,nXPix,3);
                    
                    turbuRGB = max(0,min(1,turbu));
                    turbuRGB = reshape(turbuMap(1+round(turbuRGB*127),:),nZPix,nXPix,3);
                    
                    %% Duplex
                    if ~isempty(obj.smoothKernel)
                        N   = conv2(msk,obj.smoothKernel,'same');

                        power = conv2(power.*msk,obj.smoothKernel,'same') ./ N;
                        turbu = conv2(turbu.*msk,obj.smoothKernel,'same') ./ N;
                        stdev = sqrt((conv2(color.^2,obj.smoothKernel,'same') - ...
                                      conv2(color,   obj.smoothKernel,'same').^2./N)./(N-1));
                        
                        power(~msk) = nan;
                        turbu(~msk) = nan;
                        stdev(~msk) = nan;
                    else
                        stdev = zeros(size(color));
                    end
                    duplexMask = (power >= obj.powerThreshold) ...
                               & (turbu <= obj.turbuThreshold) ...
                               & (stdev <= obj.stdevThreshold);
                    
                    imageRGB = bmodeRGB.*~duplexMask + colorRGB.*duplexMask;
                else
                    imageRGB = bmodeRGB;
                end
                
                
                set(obj.hImg(1), 'CData', imageRGB);
                if numel(obj.hImg)==4
                    set(obj.hImg(2), 'CData', bmodeRGB);
                    set(obj.hImg(3), 'CData', powerRGB);
                    set(obj.hImg(4), 'CData', turbuRGB);
                end
                
                if obj.vectorEnable
                    colorX(~duplexMask) = nan;
                    colorZ(~duplexMask) = nan;
                    
                    obj.vector.hQvr.UData = obj.vector.scale*colorX(obj.vector.zSel,obj.vector.xSel);
                    obj.vector.hQvr.VData = obj.vector.scale*colorZ(obj.vector.zSel,obj.vector.xSel);
                end
                
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


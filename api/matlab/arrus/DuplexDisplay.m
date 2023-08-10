classdef DuplexDisplay < handle
    % A Duplex display.
    %
    % Currently is implemented as a simple MATLAB figure with dynamically
    % updated content.
    % 
    % :param reconstructionObject: object of class "Reconstruction" \
    %   (obligatory input unless old syntax is used)
    % :param dynamicRange: two-element vector [min, max], dynamic range \
    %   limits to apply (optional name-value argument)
    % :param powerThreshold: power limit [dB] below which the color data \
    %   is not displayed (scalar) (optional name-value argument)
    % :param turbuThreshold: turbulence limit over which the color data \
    %   is not displayed (scalar) (optional name-value argument)
    % :param stdevThreshold: limit of standard deviation of color data \
    %   over which the color data is not displayed \
    %   (scalar) (optional name-value argument)
    % :param thresholdSmoothe: size [pix] of the smoothing kernel applied \
    %   to the images that are used for thresholding, e.g. power, turbu, \
    %   stdev (scalar) (optional name-value argument)
    % :param subplotEnable: enables separate display of Duplex, bmode, \
    %   power, and turbulence (logical scalar) (optional name-value argument)
    % :param cineLoopLength: positive scalar, number of frames stored in \
    %   cineloop (optional name-value argument)
    % :param persistence: persistence filtration weights (vector) or \
    %   length (scalar) (optional name-value argument)
    % :param bmodeTgc: applies linear TGC equal to bmodeTgc[dB]/depth \
    %   (scalar) (optional name-value argument)
    % :param bmodeAutoTgcResp: applies linear TGC that adapts in time to \
    %   the imaging conditions, defines the responsiveness of the \
    %   adaptation, <0,1> (scalar) (optional name-value argument)
    
    properties(Access = private)
        hFig
        hAx
        hImg
        hQvr
        xGrid
        zGrid
        colorEnable
        vectorEnable
        dynamicRange
        powerThreshold
        turbuThreshold
        stdevThreshold
        smootheKernel
        cineLoop
        cineLoopLength
        cineLoopIndex
        persistence
        bmodeTgc
        bmodeAutoTgcResp
    end
    
    methods 
        function obj = DuplexDisplay(varargin)
            
            % Input parser
            dispParParser = inputParser;
            
            addRequired(dispParParser, 'reconstructionObject', ...
                        @(x) assert(isscalar(x) && isa(x,'Reconstruction'), ...
                        "reconstructionObject is an obligatory scalar input of class Reconstruction."));
            
            addParameter(dispParParser, 'dynamicRange', [0 80], ...
                        @(x) assert(isnumeric(x) && all(size(x) == [1 2]), ...
                        "dynamicRange must be a 2-element numerical vector: [min max]."));
            
            addParameter(dispParParser, 'powerThreshold', -inf, ...
                        @(x) assert(isnumeric(x) && isscalar(x), ...
                        "powerThreshold must be a numerical scalar."));
            
            addParameter(dispParParser, 'turbuThreshold', 1, ...
                        @(x) assert(isnumeric(x) && isscalar(x), ...
                        "turbuThreshold must be a numerical scalar."));
            
            addParameter(dispParParser, 'stdevThreshold', 0, ...
                        @(x) assert(isnumeric(x) && isscalar(x), ...
                        "stdevThreshold must be a numerical scalar."));
            
            addParameter(dispParParser, 'thresholdSmoothe', 0, ...
                        @(x) assert(isnumeric(x) && isscalar(x), ...
                        "thresholdSmoothe must be a numerical scalar."));

            addParameter(dispParParser, 'subplotEnable', false, ...
                        @(x) assert(islogical(x) && isscalar(x), ...
                        "subplotEnable must be a logical scalar."));
            
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
            powerThreshold   = dispParParser.Results.powerThreshold;
            turbuThreshold   = dispParParser.Results.turbuThreshold;
            stdevThreshold   = dispParParser.Results.stdevThreshold;
            thresholdSmoothe = dispParParser.Results.thresholdSmoothe;
            subplotEnable    = dispParParser.Results.subplotEnable;
            cineLoopLength   = dispParParser.Results.cineLoopLength;
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
            
            if thresholdSmoothe>0
                aux = linspace(-1,1,thresholdSmoothe).^2;
                smootheKernel = sqrt(aux.' + aux);
            else
                smootheKernel = [];
            end
            
            obj.xGrid = proc.xGrid;
            obj.zGrid = proc.zGrid;
            obj.colorEnable = proc.colorEnable;
            obj.vectorEnable = proc.vectorEnable;
            obj.dynamicRange = dynamicRange;
            obj.powerThreshold = powerThreshold;
            obj.turbuThreshold = turbuThreshold;
            obj.stdevThreshold = stdevThreshold;
            obj.smootheKernel = smootheKernel;
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
            
            obj.hQvr = nan;
            
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
            % :param data: duplex data to display
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
                    if ~isempty(obj.smootheKernel)
                        power = conv2(power.*msk,obj.smootheKernel,'same') ./ conv2(msk,obj.smootheKernel,'same');
                        turbu = conv2(turbu.*msk,obj.smootheKernel,'same') ./ conv2(msk,obj.smootheKernel,'same');
                        
                        % local std of color map
                        N   = conv2(msk,obj.smootheKernel,'same');
                        avg = conv2(color,obj.smootheKernel,'same')./N;
                        stdev = sqrt((conv2(color.^2,obj.smootheKernel,'same')-N.*avg.^2)./(N-1));
                        
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
                    vecDec	= 20;
                    
                    vXSel	= vecDec:vecDec:(nXPix-vecDec+1);
                    vZSel	= vecDec:vecDec:(nZPix-vecDec+1);
                    vMultip	= vecDec*diff(obj.xGrid(1:2)*1e3)/pi  /2;
                    
                    colorX(~duplexMask) = nan;
                    colorZ(~duplexMask) = nan;
                    
                    if ishandle(obj.hQvr)
                        delete(obj.hQvr);
                    end
                    
                    axes(obj.hAx), hold on;
                    obj.hQvr = quiver(	obj.xGrid(vXSel)*1e3, ...
                                        obj.zGrid(vZSel)*1e3, ...
                                        vMultip*colorX(vZSel,vXSel), ...
                                        vMultip*colorZ(vZSel,vXSel),0,'Color','k');
                    obj.hQvr.Head.LineWidth = 1.5;
                end
                
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


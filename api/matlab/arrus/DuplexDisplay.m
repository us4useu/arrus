classdef DuplexDisplay < handle

    properties(Access = private)
        hFig
        hAx
        hImg
        hQvr
        xGrid
        zGrid
        colorEnable
        vectorEnable
        showTimes
        dynamicRange
        powerThreshold
    end
    
    methods 
        function obj = DuplexDisplay(proc, dynamicRange, powerThreshold, subplotEnable)
            
            if nargin < 4
                subplotEnable = false;
            end
            
            if nargin < 3
                powerThreshold = -inf;
            end
            
            if nargin < 2
                dynamicRange = [0 60];
            else 
                if ~all(size(dynamicRange) == [1 2])
                    error("ARRUS:IllegalArgument", ...
                        "Invalid dimensions of dynamic range vector, should be: [min max]")
                end
            end
            
            obj.xGrid = proc.xGrid;
            obj.zGrid = proc.zGrid;
            obj.colorEnable = proc.colorEnable;
            obj.vectorEnable = proc.vectorEnable;
            obj.dynamicRange = dynamicRange;
            obj.powerThreshold = powerThreshold;
            
            % Create figure.
            obj.hFig = figure();
            if subplotEnable && (obj.colorEnable || obj.vectorEnable)
                if obj.colorEnable
                    subplotColorMaps = cat(3, gray, gray, jet, hot);
                    subplotDynRange = [dynamicRange; dynamicRange; [-pi pi]; dynamicRange];
                elseif obj.vectorEnable
                    subplotColorMaps = cat(3, gray, gray, hot, hot);
                    subplotDynRange = [dynamicRange; dynamicRange; [0 2*pi]; dynamicRange];
                end
                
                for iAx=1:4
                    obj.hAx(iAx) = subplot(2,2,iAx);
                    obj.hImg(iAx) = image(obj.xGrid*1e3, obj.zGrid*1e3, []);
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
                obj.hImg = image(obj.xGrid*1e3, obj.zGrid*1e3, []);
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
                
                bmode = data(:,:,:,1);
                bmode(isnan(bmode)) = -inf;
                
                bmodeRGB = (bmode - obj.dynamicRange(1)) / diff(obj.dynamicRange);
                bmodeRGB = max(0,min(1,bmodeRGB));
                bmodeRGB = bmodeRGB .* ones(1,1,3);   % colormap = gray
                
                if obj.colorEnable || obj.vectorEnable
                    power = data(:,:,:,2);
                    powerMap = hot(128);
                    if obj.colorEnable
                        color = data(:,:,:,3);
                        colorMap = jet(128);
                        colorOffset = 0.5;
                    elseif obj.vectorEnable
                        colorX = data(:,:,:,3);
                        colorZ = data(:,:,:,4);
                        color = sqrt(colorX.^2 + colorZ.^2);
                        colorMap = hot(128);
                        colorOffset = 0;
                    end
                    power(isnan(power)) = -inf;
                    color(isnan(color)) = 0;
                    
                    colorRGB = color/2/pi + colorOffset;
                    colorRGB = max(0,min(1,colorRGB));
                    colorRGB = reshape(colorMap(1+round(colorRGB*127),:),nZPix,nXPix,3);
                    
                    powerRGB = (power - obj.dynamicRange(1)) / diff(obj.dynamicRange);
                    powerRGB = max(0,min(1,powerRGB));
                    powerRGB = reshape(powerMap(1+round(powerRGB*127),:),nZPix,nXPix,3);
                    
                    duplexMask = (power >= obj.powerThreshold);
                    imageRGB = bmodeRGB.*~duplexMask + colorRGB.*duplexMask;
                else
                    imageRGB = bmodeRGB;
                end
                
                
                set(obj.hImg(1), 'CData', imageRGB);
                if numel(obj.hImg)==4
                    set(obj.hImg(2), 'CData', bmodeRGB);
                    set(obj.hImg(3), 'CData', colorRGB);
                    set(obj.hImg(4), 'CData', powerRGB);
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
    end    
end


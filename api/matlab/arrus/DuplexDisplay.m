classdef DuplexDisplay < handle

    properties(Access = private)
        hFig
        hAxDuplex
        hImg
        hQvr
        xGrid
        zGrid
        showTimes
        dynamicRange
        powerThreshold
    end
    
    methods 
        function obj = DuplexDisplay(xGrid, zGrid, dynamicRange, powerThreshold)
            if nargin < 3
                dynamicRange = [0 60];
            else 
                if ~all(size(dynamicRange) == [1 2])
                    error("ARRUS:IllegalArgument", ...
                        "Invalid dimensions of dynamic range vector, should be: [min max]")
                end
            end
            
            obj.xGrid = xGrid;
            obj.zGrid = zGrid;
            obj.dynamicRange = dynamicRange;
            obj.powerThreshold = powerThreshold;
            
            % Create figure.
            obj.hFig = figure();
            for iAx=1:4
                
                if iAx==2
                    obj.hAxDuplex = subplot(2,2,iAx);
                    obj.hImg(iAx) = image  (xGrid*1e3, zGrid*1e3,[]);
                else
                    subplot(2,2,iAx);
                    obj.hImg(iAx) = imagesc(xGrid*1e3, zGrid*1e3,[]);
                end
                
                if iAx==1
                    colormap(gca,gray);
                    set(gca,'CLim',dynamicRange);
                elseif iAx==3
                    colormap(gca,hot);
                    set(gca,'CLim',dynamicRange);
                elseif iAx==2 || iAx==4
                    % will require change for vector doppler
                    colormap(gca,jet);
                    set(gca,'CLim',[-pi pi]);
                end
                colorbar;
                xlabel('x [mm]'), ylabel('z [mm]'), daspect([1 1 1]);
                set(gca, 'XLim', xGrid([1 end])*1e3, 'YLim', zGrid([1 end])*1e3);
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
            % :param img: an image to display
            try
                [nZPix,nXPix,~,nComp] = size(data);
                
                bmode = data(:,:,:,1);
                
                if nComp==3
                    vectorEnable = false;
                    power = data(:,:,:,2);
                    color = data(:,:,:,3);
                    colorMap = jet(128);
                    colorOffset = 0.5;
                elseif nComp==5
                    vectorEnable = true;
                    power = data(:,:,:,2:3);
                    color = data(:,:,:,4:5);
                    colorMap = hot(128);
                    colorOffset = 0;
                end
                
                bmode(isnan(bmode)) = -inf;
                power(isnan(power)) = -inf;
                color(isnan(color)) = 0;
                
                duplexMask = (power >= obj.powerThreshold) & all(power > -inf, 4);
                
                if vectorEnable
                    % fixed values of txrxAng!!!
                    txrxAng = (0 + 20*pi/180 * [-1;1]) / 2;   % [rad] (1 x 2)
                    txrxAng	= reshape(txrxAng,1,1,1,[]);
                    colorX	=   diff(-color.*duplexMask./cos(txrxAng),                    [],4) / diff(tan(txrxAng));
                    colorZ	= - diff(-color.*duplexMask./cos(txrxAng).*tan(flip(txrxAng)),[],4) / diff(tan(txrxAng));
                    color	= sqrt(colorX.^2 + colorZ.^2);
                    power	= mean(power,4);
                    duplexMask = color ~= 0;
                end
                
                duplexBMode = (bmode - obj.dynamicRange(1)) / diff(obj.dynamicRange) .* ones(1,1,3);
                duplexBMode = max(0,min(1,duplexBMode));
                duplexColor = reshape(colorMap(1+round(max(0,min(1, color/2/pi + colorOffset))*127),:),nZPix,nXPix,3);
                
                duplex = duplexBMode.*~duplexMask + duplexColor.*duplexMask;
                
                set(obj.hImg(1), 'CData', bmode);
                set(obj.hImg(2), 'CData', duplex);
                set(obj.hImg(3), 'CData', power);
                set(obj.hImg(4), 'CData', color);
                
                if vectorEnable
                    % fixed sparsing coefficient!!!
                    vecDec	= 20;
                    
                    vXSel	= vecDec:vecDec:nXPix;
                    vZSel	= vecDec:vecDec:nZPix;
                    vMultip	= vecDec*diff(obj.xGrid(1:2)*1e3)/pi  /2;
                    
                    if ishandle(obj.hQvr)
                        delete(obj.hQvr);
                    end
                    
                    axes(obj.hAxDuplex), hold on;
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


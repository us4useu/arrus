classdef BModeDisplay < handle
    properties(Access = private)
        hFig
        hImg
        showTimes
    end
    
    methods
        function obj = Us4RDisplay(xGrid, zGrid)
            % Create figure.
            obj.hFig = figure();
            obj.hImg = imagesc(xGrid*1e3, zGrid*1e3,[]);
            xlabel('x [mm]');
            ylabel('z [mm]');
            daspect([1 1 1]);
            set(gca, 'XLim', xGrid([1 end])*1e3);
            set(gca, 'YLim', zGrid([1 end])*1e3);
            set(gca, 'CLim', [-20 80]);
            colormap(gray);
            colorbar;    
        end
        
        function state = isOpen(obj)
            state = ishghandle(obj.hFig);
        end
        
        function updateImg(obj, img)
            set(obj.hImg, 'CData', img);
            drawnow;
        end
    end    
end


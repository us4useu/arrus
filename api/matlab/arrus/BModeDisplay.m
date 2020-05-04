classdef BModeDisplay < handle
    % A B-mode image display.
    %
    % Currently is implemented as a simple MATLAB figure with dynamically
    % updated content.
    %
    % :param xGrid: (1, width) vector, x-coordinates of the image pixels [m]
    % :param zGrid: (1, depth) vector z-coordinates of the image pixels [m]

    properties(Access = private)
        hFig
        hImg
        showTimes
    end
    
    methods
        function obj = BModeDisplay(xGrid, zGrid)
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
            % Checks if the window was not already closed.
            %
            % :return: true when the windows with b-mode image was not closed,
            %          false otherwise
            state = ishghandle(obj.hFig);
        end
        
        function updateImg(obj, img)
            % Updates currently displayed image.
            %
            % :param img: an image to display
            set(obj.hImg, 'CData', img);
            drawnow;
        end
    end    
end


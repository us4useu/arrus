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
            set(gca, 'CLim', [20 80]);
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
            try
                set(obj.hImg, 'CData', img);
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


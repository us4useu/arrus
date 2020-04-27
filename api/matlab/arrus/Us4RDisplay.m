classdef Us4RDisplay < handle

    properties(Access = private)
        rec

    end

    methods
        function setRecParams(obj,varargin)
            %% Set reconstruction parameters
            % Reconstruction parameters names mapping
            %                    public name         private name
            recParamMapping = { 'filterEnable',     'filtEnable'; ...
                                'filterACoeff',     'filtA'; ...
                                'filterBCoeff',     'filtB'; ...
                                'filterDelay',      'filtDel'; ...
                                'iqEnable',         'iqEnable'; ...
                                'cicOrder',         'cicOrd'; ...
                                'decimation',       'dec'; ...
                                'xGrid',            'xGrid'; ...
                                'zGrid',            'zGrid'};

            if mod(length(varargin),2) == 1
                % Throw exception
            end

            for iPar=1:size(recParamMapping,1)
%                 eval(['obj.seq.' recParamMapping{iPar,2} ' = [];']);
                eval(['obj.rec.' recParamMapping{iPar,2} ' = [];']);
            end

            nPar = length(varargin)/2;
            for iPar=1:nPar
                idPar = find(strcmpi(varargin{iPar*2-1},recParamMapping(:,1)));

                if isempty(idPar)
                    % Throw exception
                end

                if ~isnumeric(varargin{iPar*2})
                    % Throw exception
                end

                eval(['obj.rec.' recParamMapping{idPar,2} ' = reshape(varargin{iPar*2},1,[]);']);
            end

            %% Resulting parameters
            obj.rec.zSize	= length(obj.rec.zGrid);
            obj.rec.xSize	= length(obj.rec.xGrid);

            %% Fixed parameters
            obj.rec.gpuEnable	= license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('distcomp'));
        end

        function img = execReconstr(obj,rfRaw)

            %% Move data to GPU if possible
            if obj.rec.gpuEnable
                rfRaw = gpuArray(rfRaw);
            end

            %% Preprocessing
            % Raw rf data filtration
            if obj.rec.filtEnable
                rfRaw = filter(obj.rec.filtB,obj.rec.filtA,rfRaw);
            end

            % Digital Down Conversion
            rfRaw = downConversion(rfRaw,obj.seq,obj.rec);

            % warning: both filtration and decimation introduce phase delay!
            % rfRaw = preProc(rfRaw,obj.seq,obj.rec);

            %% Reconstruction
            if strcmp(obj.seq.type,'lin')
                rfBfr = reconstructRfLin(rfRaw,obj.sys,obj.seq,obj.rec);
            else
                rfBfr = reconstructRfImg(rfRaw,obj.sys,obj.seq,obj.rec);
            end

            %% Postprocessing
            % Obtain complex signal (if it isn't complex already)
            if ~obj.rec.iqEnable
                nanMask = isnan(rfBfr);
                rfBfr(nanMask) = 0;
                rfBfr = hilbert(rfBfr);
                rfBfr(nanMask) = nan;
            end

            % Scan conversion (for 'lin' mode)
            if strcmp(obj.seq.type,'lin')
                rfBfr = scanConversion(rfBfr,obj.sys,obj.seq,obj.rec);
            end

            % Envelope detection
            envImg = abs(rfBfr);

            % Compression
            img = 20*log10(envImg);

            % Gather data from GPU
            if obj.rec.gpuEnable
                img = gather(img);
            end


        end



        function [] = runOnce(obj)

            %% TX/RX sequence
            obj.openSequence;
            rf	= obj.execSequence;
            obj.closeSequence;

            %% Reconstruction
            img	= obj.execReconstr(rf);

            %% Display
            if obj.rec.filtEnable
                rf = filter(obj.rec.filtB,obj.rec.filtA,rf);
            end

            nTx     = obj.seq.nTx;
            nTx     = min(3,nTx);   % limits the number of displayed data sets

            figure;
            for iTx=1:nTx
                subplot(1,nTx,iTx);
                imagesc(rf(:,:,iTx));
                xlabel('Chan #');
                ylabel('Samp #');
                colormap(jet);
                colorbar;
                set(gca,'CLim',[-1 1]*1e2);
            end
            set(gcf,'Position',get(gcf,'Position') + [560 0 0 0]);

            figure;
            imagesc(obj.rec.xGrid*1e3,obj.rec.zGrid*1e3,img);
            xlabel('x [mm]');
            ylabel('z [mm]');
            daspect([1 1 1]);
%             set(gca,'CLim',[40 80]);
            colormap(gray);
            colorbar;

        end



        function [] = runLoop(obj,showTimes)

            if nargin<2
                showTimes = false;
            end

            %% Prepare the display
            hFig	= figure;
            hImg	= imagesc(obj.rec.xGrid*1e3,obj.rec.zGrid*1e3,[]);
            xlabel('x [mm]');
            ylabel('z [mm]');
            daspect([1 1 1]);
            set(gca,'XLim',obj.rec.xGrid([1 end])*1e3);
            set(gca,'YLim',obj.rec.zGrid([1 end])*1e3);
            set(gca,'CLim',[-20 80]);
            colormap(gray);
            colorbar;

            %% TX/RX / Reconstruction / Display
            obj.openSequence;
            iFrame = 0;
            while(ishghandle(hFig))
                iFrame = iFrame + 1;

                % TX/RX sequence
                tic;
                rf	= obj.execSequence;
                tSeq = toc;

                % Reconstruction
                tic;
                img	= obj.execReconstr(rf);
                tRec = toc;

                % Display
                set(hImg, 'CData', img);
                drawnow;

                % Show times
                if showTimes
                    disp(['Frame no. ' num2str(iFrame)]);
                    disp(['Acq.  time = ' num2str(tSeq,         '%5.3f') ' s']);
                    disp(['Rec.  time = ' num2str(tRec,         '%5.3f') ' s']);
                    disp(['Frame rate = ' num2str(1/(tSeq+tRec),'%5.1f') ' fps']);
                    disp('--------------------');
                end
            end
            obj.closeSequence;

        end

    end

end


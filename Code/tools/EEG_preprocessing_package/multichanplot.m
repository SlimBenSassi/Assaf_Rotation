function varargout = multichanplot( data, varargin )
% MULTICHANPLOT plots scrollable, markable multi-channel data. 
%
% Press 'h' or F1 to open a help window with a list of commands. Press left and right arrow to jump 
% forward and back. Add shift to move by 1/10th of a window. "Home"/"End" jump to start/end of the 
% data set. Press up and down to change the y-axis scale by 10%. Press 'c' to set the plotted channels. 
% Press 'y' to set the y-axis range. Press 'l' to set the window duration. Press 'p' to set a new current 
% position. Press 'm' to toggle whether marked intervals are displayed. Press 'x' to delete all manually-
% marked intervals and reset pre-loaded intervals. 
% NOTE: Key-press commands will not be captured by multichanplot if any figure tool is selected (zoom, 
% data cursor, etc.). 
%
% Marking intervals: Press the left mouse button to start marking an interval, and the right mouse 
% button to finish marking an interval. Clicking the left mouse button within a marked interval will 
% cancel it. Ending a marked interval within an already marked interval using the right mouse button 
% will concatenate the two intervals. Intervals can be set across different display windows (press 
% left mouse button, then left or right keys, then right mouse button). 
%
% Marked intervals which have been pre-loaded into the function will appear in a different color,
% but behave identically to manually marked intervals. 
%
% Syntax:
% MULTICHANPLOT(data):                       Displays a multiple-channel plot, where "data" is a time x channels
%                                            matrix. As a default, a 1000-sample/1-second window is displayed. 
% MULTICHANPLOT(...,winLen,...):             Where winLen is a scalar, defines the number of samples/seconds 
%                                            displayed in each window. 
% MULTICHANPLOT(...,'ylim',[ymin ymax]):     Sets the y-scale of each channel, where YL is a 2-element
%                                            vector. Default value is the minimum and maximum values of the 
%                                            input dataset.
% MULTICHANPLOT(...,'channels',c):           Selects which channels will be plotted. 
% MULTICHANPLOT(...,'channelnames',C)        Provides the names of the channels to be displayed, where C is a cell
%                                            array of strings, with the same length as the number of columns in the 
%                                            input data. 
% MULTICHANPLOT(...,'srate',sr):             Sets the sampling rate of the input data. The "winLen"
%                                            variable will be interpreted as seconds. Default value is 1. 
% MULTICHANPLOT(...,'markdata',m):           Provides an initial set of marked intervals, where m is a N x 1 
%                                            vector with the same length as the input data, with non-zero
%                                            samples corresponding to marked intervals. 
% MULTICHANPLOT(...,'markIndivChan',ma):     Provides an initial set of marked intervals in individual channels,
%                                            where ma is a N x C matrix with same size as the input data, with 1/0
%                                            corresponding to marked/non-marked intervals in specific channels
% MULTICHANPLOT(...,handle,...):             Where handle is the handle of an existing figure or axes object, 
%                                            will create the multichanplot display within that figure or axes. 
%                                            For example: multichanplot(...,subplot(2,1,1)) will create the 
%                                            display in a sub-plot axes. NOTE: Bugs may occur when multiple 
%                                            multichanplot displays are opened in the same figure. 
% MULTICHANPLOT(...,'capturekeys',b):        Where b is a boolean value, sets whether keyboard key-presses will 
%                                            be captured by the GUI. Since key presses are captured by the figure, 
%                                            set this to false if you have multiple multi-chan-plots in a single 
%                                            figure to allow only one multi-chan-plot to respond to key presses. 
%                                            Default value is true. 
% 
% mask = MULTICHANPLOT(...):                 Setting an output variable will cause the marked segments to
%                                            be exported in the form of a Nx1 boolean vector. 
% 
% Written by Edden M. Gerber, lab of Leon Y. Deouell, Sep. 2013. Inspired by EEGlab's eegplot function.
% Edited by Assaf Breska


%% Parameters
MARK_COLOR_INITIAL = 'c';
MARK_COLOR_MANUAL = 'y';
MARK_COLOR_ELEC = 'r';

%%  Handle input

% Turn row vector into column
if size(data,1) == 1
    data = data'; 
end
% Check if the matrix should be transposed
if size(data,2) > size(data,1)
    warning('multichanplot: warning: number of channel exceeds number of time points - expected matrix dimensions are time x channels');
end

% initialize optional arguments
winLen = 1000;
defaultWinLen = true;
yLim = []; 
channels = 1:size(data,2);
channelNames = 1:size(data,2);
srate = 1;
markdata = [];
artMat = [];
capturekeys = true;
% read optional arguments
narg = size(varargin,2);
arg  = 1;
while arg <= narg
    if ischar(varargin{arg})
        switch varargin{arg}
            case 'ylim'
                if narg > arg && isvector(varargin{arg+1}) && isnumeric(varargin{arg+1})
                    yLim = varargin{arg+1};
                    arg = arg + 2;
                else 
                    error('multichanplot: ''ylim'' argument should be followed by a [-y y] vector');
                end
            case 'channels'
                if narg > arg && isvector(varargin{arg+1}) && isnumeric(varargin{arg+1})
                    channels = varargin{arg+1};
                    arg = arg + 2;
                else 
                    error('multichanplot: ''channels'' argument should be followed by a channel vector');
                end
            case 'channelnames'
                if narg > arg && iscell(varargin{arg+1}) && length(varargin{arg+1}) == size(data,2)
                    channelNames = varargin{arg+1};
                    arg = arg + 2;
                else 
                    error('multichanplot: ''channelnames'' argument should be followed by a cell array of the same length as the number of data columns');
                end
            case 'srate'
                if narg > arg && isscalar(varargin{arg+1}) && isnumeric(varargin{arg+1})
                    srate = varargin{arg+1};
                    if defaultWinLen
                        winLen = 1; % default
                    end
                    arg = arg + 2;
                else 
                    error('multichanplot: ''srate'' argument should be followed by a scalar');
                end
            case 'markdata'
                if narg > arg && ismatrix(varargin{arg+1}) && (isnumeric(varargin{arg+1}) || islogical(varargin{arg+1}))
                    markdata = varargin{arg+1};
                    arg = arg + 2;
                else 
                    error('multichanplot: ''markdata'' argument should be followed by a vector or matrix');
                end
            case 'capturekeys'
                if narg > arg && isscalar(varargin{arg+1}) && (islogical(varargin{arg+1}) || isnumeric(varargin{arg+1}))
                    capturekeys = varargin{arg+1};
                    arg = arg + 2;
                else 
                    error('multichanplot: ''capturekeys'' argument should be followed by true or false');
                end
            case 'markIndivChan'
                if narg > arg && ismatrix(varargin{arg+1}) && ((isnumeric(varargin{arg+1}) || islogical(varargin{arg+1}))) && sum(size(data)~=size(varargin{arg+1}))==0
                    artMat = varargin{arg+1};
                    arg = arg + 2;
                else 
                    error('multichanplot: ''markIndivChan'' argument should be followed by a matrix with the same size as first input argument');
                end
            otherwise
                error(['Unknown argument: ' varargin{arg}]);
        end
    elseif ishandle(varargin{arg})
       in_handle = varargin{arg};
       arg = arg + 1;
    elseif isnumeric(varargin{arg}) && isscalar(varargin{arg})
        winLen = varargin{arg};
        defaultWinLen = false;
       arg = arg + 1;
    end
end

%% Initialize

if ~exist('in_handle','var')
    h_fig = figure;
    h_axes = axes;
else
    switch get(in_handle,'type')
        case 'figure'
            h_fig = in_handle;
            h_axes = axes;
        case 'axes'
            h_fig = get(in_handle,'parent');
            h_axes = in_handle;
        otherwise
            error('Multichanplot: Input handle should be to figure or axis object.');
    end
end

axes(h_axes); % Give focus;

if capturekeys
    set(h_fig,'KeyPressFcn',@f_KeyPress);
    set(h_axes,'tag','multichanplot');     % This is to identify this axes as the multichanplot axis 
                                           % when it is part of a larger GUI
end

L = size(data,1);
loc = 1;
yInterv = [];
yTotal = [];
nChan = [];
winLen = round(winLen * srate);
if winLen > L
  warning(['Window size set to more than the length of the data: setting it to maximum window size (' num2str(L/srate) ').']);
  winLen = L;
end
T = (1:L) / srate;

if isempty(yLim)
    % yLim = [min(data(:)) max(data(:))]; The min-max range is too often too large. 
    yLim = [-5*nanmean(std(data(1:winLen,:))) 5*nanmean(std(data(1:winLen,:)))]; % instead use 5 standard deviations as the default range. 
end

show_marks = true;
select_on = false;
curr_select = 0;
mask = zeros(L,1);
artMatMask=[];

if ~isempty(markdata)
    mask(~~markdata) = 2;    
end

if ~isempty(artMat) 
    artMatMask=artMat;
    artMatMask(artMatMask==0)=nan;
end


helpStr = {'Keyboard commands: ',...
                 '',...
                 'Left/right arrow              Jump one time window',...
                 'Shift+left/right arrow      Jump 1/10 time window',...
                 'Home/End                    Jump to start/end',...
                 'Up/down arrow             Change y-axis scale by 10%',...
                 '''c''                                   Select channels to plot',...
                 '''y''                                   Set y-axis range',...
                 '''l''                                    Set window size',...
                 '''p''                                   Set new current position',...
                 '''m''                                   Toggle display of marked intervals',...
                 '''x''                                   Delete all marked intervals',...
                 '''r''                                   Reset initial marked intervals',...
                 '''h''                                   Display command help',...
                 '',...
                 '',...
                 'Marking intervals: ',...
                 '',...
                 strcat('Press the left mouse button to start marking an interval, and the right mouse button ',...
                 ' to finish marking an interval. Clicking the left mouse button within a marked interval will cancel it. Ending ', ...
                 ' a marked interval within an already  marked interval will concatenate the two intervals. Intervals can be set ',...
                 ' across different display windows (press left mouse button, then left or right keys, then right mouse button). ',...
                 ' Marked intervals which have been pre-loaded into the function will appear in a different color, but behave ',...
                 ' identically to manually marked intervals. '),...
                 '',...
                 '',...
                 'Optional function input parameters:',...
                 '',...
                 '''ylim''                               Set y-axis for each channel',...
                 '''channels''                      Select subset of channels to display',...
                 '''srate''                             Set sampling rate',...
                 '''markdata''                     Supply a vector of pre-set intervals',...
                 '',...
                 '',...
                 '',...
                 '',...
                 '',...
                 '',...
                 '',...
                 '',...
                 };

%%  Run

update_data;

plotfig;

if nargout > 0
    waitfor(h_fig);
end

%% Output
if nargout > 0
    mask = mask(1:L);
    mask = ~~mask;
   varargout{1} = mask;
end

%% Nested functions

    function update_data
        yInterv = yLim(2)-yLim(1);
        nChan = length(channels);
        yTotal = yInterv * nChan;
        plotfig;
    end

    function plotfig
        
        % FUTURE OPTIMIZATIONS: 
        % don't redraw the whole axis; just delete and redraw the plot
        % handles. 
        % Instead of multiple calls to plot, use the multiple "triplets"
        % input method. 
        %
        dataWin = data(loc:(loc+winLen-1),channels);
        dataWin = bsxfun(@minus,dataWin,dataWin(1,:));
        yStart = (nChan:-1:1)*yInterv-(yInterv/2);
        dataWin = bsxfun(@plus,dataWin,yStart);
        tWin = T(loc:(loc+winLen-1));
        maskWin1 = dataWin;
        maskWin2 = dataWin;
        maskWin1(~(mask(loc:(loc+winLen-1))==1),:) = nan;
        maskWin2(~(mask(loc:(loc+winLen-1))==2),:) = nan;
        
        if ~isempty(artMat)
            artMatMaskWin=dataWin.*artMatMask(loc:(loc+winLen-1),channels);
        end

        hold off
        h_1 = plot(h_axes,tWin,dataWin);
        hold all
        if show_marks
            h_2 = plot(h_axes,tWin,maskWin1,'color',MARK_COLOR_MANUAL,'linewidth',1.5);
            h_3 = plot(h_axes,tWin,maskWin2,'color',MARK_COLOR_INITIAL,'linewidth',1.5);
            if ~isempty(artMat) 
                h_4 = plot(h_axes,tWin,artMatMaskWin,'color',MARK_COLOR_ELEC,'linewidth',1.5);
                line_handles = [h_1 ; h_2 ; h_3 ; h_4];
            else
                line_handles = [h_1 ; h_2 ; h_3 ];
            end
        else
            line_handles = [h_1];
        end
        set(gca,'ytick',yStart(end:-1:1),'yticklabel',channelNames(channels(end:-1:1)));
        set(h_axes,'tag','multichanplot');  % This is to identify this axes
                                            % as a multichanplot axis when 
                                            % it is part of a larger GUI. 
                                            % This needs to be set after 
                                            % each plot as it resets the 
                                            % axes properties. 
        ylim([-yInterv yTotal+yInterv]);
        xlim([tWin(1) tWin(end)]);
        
        % Add "progress bar"
        bar_width = 4;
        bar_clr = [0.8 0.8 1];
        segment_clr = [0.5 0.5 1];
        seg_position = [floor(loc / L * length(tWin)) ceil((loc+winLen) / L * length(tWin))];
        if seg_position(1) < 1; seg_position(1) = 1; end
        if seg_position(2) > length(tWin); seg_position(2) = length(tWin); end;
        line([tWin(1) tWin(end)],[-yInterv -yInterv],'color',bar_clr,'linewidth',bar_width);
        line([tWin(seg_position(1)) tWin(seg_position(2))],[-yInterv -yInterv],'color',segment_clr,'linewidth',bar_width);
        
        if 0 % This slows down the scrolling and is therefore disabled:
            if srate==1
                xlabel('Samples');
            else
                xlabel('Time (sec)');
            end
            ylabel('Channels');
        end
        
        set(gca,'ButtonDownFcn',@f_ButtonDown);
        set(line_handles,'hittest','off'); % this is so that a click on a line will be captured 
                                           % by the axis object behind it. 
    end

    function f_KeyPress( hObject, eventdata, handles )
        % first focus on multichanplot axis object 
        mcpHandle = findobj('tag','multichanplot');
        if isempty(mcpHandle)
            return;
        else
            axes(mcpHandle(1));
        end
        switch eventdata.Key
            case 'rightarrow'
                if ~isempty(eventdata.Modifier) && strcmp(eventdata.Modifier{1},'shift');
                    loc = loc + ceil(winLen/10);
                else
                    loc = loc + winLen;
                end
                if loc > L-winLen+1
                    loc = L-winLen+1;
                end
                plotfig;
            case 'leftarrow'
                if ~isempty(eventdata.Modifier) && strcmp(eventdata.Modifier{1},'shift');
                    loc = loc - ceil(winLen/10);
                else
                    loc = loc - winLen;
                end
                if loc < 1
                    loc = 1;
                end
                plotfig;
            case 'downarrow'
                Y = yLim(2)-yLim(1);
                yLim(1) = yLim(1) - Y*0.05;
                yLim(2) = yLim(2) + Y*0.05;
                update_data;
            case 'uparrow'
                Y = yLim(2)-yLim(1);
                yLim(1) = yLim(1) + Y*0.05;
                yLim(2) = yLim(2) - Y*0.05;
                update_data;
            case 'home'
                loc = 1;
                update_data;
            case 'end'
                loc = L - winLen + 1;
                update_data;
            case 'y'
                in = inputdlg('enter new y limits (ymin ymax): ','y limits',1,{num2str(yLim)});
                if ~isempty(in)
                    try
                        eval(['yLim = [' in{1} '];']);
                    end
                    update_data;
                end
            case 'c'
                in = inputdlg('enter new channel list vector (type ":" for all channels): ','channels');
                if ~isempty(in)
                    if strcmp(in{1},':')
                        channels = 1:size(data,2);
                    else
                        channels = str2num(in{1});
                    end
                    update_data;
                end
            case 'l'
                in = inputdlg('enter new window length (in seconds if s.r. is set): ','window length',1,{num2str(winLen/srate)});
                if ~isempty(in)
                    winLen = str2double(in{1}) * srate;
                    winLen  = round(winLen);
                    if winLen < 2
                        winLen = 2;
                    end
                    if winLen > L
                      warning(['Window size set to more than the length of the data: setting it to maximum window size (' num2str(L/srate) ').']);
                      winLen = L;
                    end
                    data = data(1:L,:);
                    mask = mask(1:L);
                    L = size(data,1);
                    T = (1:L) / srate;
                    if loc > L-winLen+1; loc = L-winLen+1; end
                    if loc < 1; loc = 1; end;
                    update_data;
                end
            case 'p'
                in = inputdlg(['enter new current position (0 - ' num2str(L/srate) '): '],'New position',1,{num2str(loc/srate)});
                if ~isempty(in)
                    loc = str2double(in{1}) * srate;
                    loc  = round(loc);
                    if loc < 1; loc = 1;end;
                    if loc > (L - winLen); loc = L - winLen + 1; end;
                    update_data;
                end
            case 'm'
                show_marks = xor(show_marks,1);
                update_data;
            case 'x'
                button = questdlg('Delete all marked intervals?','','Ok','Cancel','Ok');
                if strcmp(button,'Ok')
                    mask = zeros(L,1);
                    update_data;
                end
            case 'r'
                button = questdlg('Reset marked intervals?','','Ok','Cancel','Ok');
                if strcmp(button,'Ok')
                    mask = zeros(L,1);
                    if ~isempty(markdata)
                        mask(~~markdata) = 2;                        
                    end
                    if ~isepmty(artMat)
                        artMatMask=artMat;
                        artMatMask(artMatMask==0)=nan;
                    end
                    update_data;
                end
            case 'h'
                msgbox(helpStr,'MultiChanPlot Help','modal');
            case 'f1'
                msgbox(helpStr,'MultiChanPlot Help','modal');
        end
    end

    function f_ButtonDown(varargin )
        currp = get(gca,'CurrentPoint');
        x = currp(1,1);
        x = round(x * srate);
        if x < 1; x = 1; end
        if x > L; x = L; end
        switch get(h_fig, 'SelectionType')
            case 'normal'
                mark_data(x,1);
            case 'alt'
                mark_data(x,2);
        end
        
    end

    function mark_data(x,button)
        if button==1
            if mask(x)
                x1 = find(~mask(x:-1:1),1,'first') - 2;
                if isempty(x1); x1 = x - 1; end;
                x2 = find(~mask(x:end),1,'first') - 2;
                if isempty(x2); x2 = L-x; end;
                mask((x-x1):(x+x2)) = false;
                artMatMask((x-x1):(x+x2),:) = nan;
                update_data;
            else
                curr_select = x;
                select_on = true;
            end
        else
            if select_on
                mask(min(curr_select,x):max(curr_select,x)) = 1;
                select_on = false;
                update_data;
            end
        end
    end

end


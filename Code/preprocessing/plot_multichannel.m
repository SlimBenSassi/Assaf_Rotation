function plot_multichannel(data, Fs, title_str)
% PLOT_MULTICHANNEL plots all EEG channels with a vertical offset.
%
% INPUTS:
%   data: [Time Points x Channels] matrix (required format for this function)
%   Fs: Sampling Rate (Hz)
%   title_str: Title for the plot.

    % --- 1. Basic Setup ---
    [n_time, n_chan] = size(data);
    
    if n_chan > 100 % Avoid plotting too many channels to prevent clutter
        warning('Plotting aborted: Too many channels to visualize cleanly.');
        return;
    end
    
    % --- 2. Determine Vertical Scaling and Offset ---
    % Calculate the max range of the signal to define the spacing.
    % We use the global standard deviation of the signal for a normalized offset.
    std_val = std(data(:));
    offset_spacing = std_val * 10; % Vertical space between channels (6 st.dev is usually clean)
    
    % Create an offset vector: [0, offset, 2*offset, 3*offset, ...]
    offset_vector = (0 : n_chan - 1) * offset_spacing;
    
    % Reshape offset_vector to be applied across all time points
    offset_matrix = repmat(offset_vector, n_time, 1);
    
    % --- 3. Apply Offset ---
    % Add the offset to each channel's data
    plot_data = data + offset_matrix;
    
   % --- 2. Create the Time Vector ---
    % Create the X-axis vector in units of seconds, starting at 0
    time_points = (0 : n_time - 1) / Fs;

    % --- 3. Plotting ---
    figure;
    
    % *** CRITICAL FIX: Plot against the time vector ***
    plot(time_points, plot_data, 'LineWidth', 0.4); 
    
    % --- 4. Aesthetics ---
    % Set Y-Axis Limits 
    min_val = min(plot_data(:));
    max_val = max(plot_data(:));
    padding = (max_val - min_val) * 0.05;
    ylim([min_val - padding, max_val + padding]); 
    
    % Set X-axis Ticks: Use the actual time points for labels
    % We use simple arithmetic to get 5 major ticks spread evenly across the time points
    num_ticks = 10;
    
    % Calculate tick POSITIONS (in seconds)
    tick_positions_sec = linspace(0, time_points(end), num_ticks);
    
    % Set the XTick values to the actual seconds
    set(gca, 'XTick', tick_positions_sec);
    
    % Set the XTickLabels to format those seconds nicely
    set(gca, 'XTickLabel', num2str(tick_positions_sec', '%.1f')); 
    
    % Set Y-axis ticks to show channel numbers/labels
    set(gca, 'YTick', offset_vector);
    set(gca, 'YTickLabel', flipud(num2str((1:n_chan)')), 'FontSize', 6); 
    
    %set(gca, 'YDir', 'reverse'); % Invert Y-axis 
    
    % Clean up axes
    xlabel(['Time (s) - Total length: ' num2str(n_time/Fs) 's']);
    ylabel('Channel Number (Offset)');
    title(title_str);
    grid on;

end
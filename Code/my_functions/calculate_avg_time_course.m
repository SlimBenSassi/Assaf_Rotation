function avg_tc = calculate_avg_time_course(time_series_cells, start_idx, end_idx)
    % Averages across the specified frequency band and then averages across trials.
    
    if isempty(time_series_cells)
        % Handle cases where there are no trials for a given outcome
        % Returns a row vector of zeros matching the time axis length
        % You may need to define the length based on your time_axis_ms
        if exist('time_axis_ms', 'var')
            avg_tc = zeros(1, length(time_axis_ms));
        else
            % Fallback: use the length of the first non-empty cell if time_axis_ms is unavailable
            first_non_empty = find(~cellfun(@isempty, time_series_cells), 1, 'first');
            if ~isempty(first_non_empty)
                nTime = size(time_series_cells{first_non_empty}, 1); % Assuming time is in rows
                avg_tc = zeros(1, nTime);
            else
                avg_tc = 0; % Default if absolutely no data exists
            end
        end
        return;
    end
    
    % 1. Average across the specified frequency band (columns) for each trial
    time_course_data = cellfun(@(x) mean(x(:, start_idx:end_idx), 2).', ...
        time_series_cells, 'UniformOutput', false);

    % 2. Stack into matrix: nTrials x nTime
    M = cell2mat(time_course_data);
    
    % 3. Average across trials (dimension 1)
    avg_tc = mean(M, 1); 
end

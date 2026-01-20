function DATA = load_power_futures_data(csvFile, opts)
%LOAD_POWER_FUTURES_DATA Loads futures data, builds row filter, drops specific columns.
%
% Outputs in DATA:
%   .T_raw, .T_clean
%   .dates
%   .seriesNames
%   .legendLabels
%   .toRemove
%   .F (numeric matrix for plotting)
%   .F_all (numeric before dropping columns, optional)

    if nargin < 2, opts = struct(); end
    if ~isfield(opts, "doPlots"), opts.doPlots = false; end

    T = readtable(csvFile);

    dates = T.date;

    % Row filter based on NaN blocks (computed on full numeric matrix)
    X_all = T{:,2:end};
    toRemove = compute_row_filter_nan_blocks(X_all);

    % Drop the two columns (as in your snippet)
    dropNames = {'TRDEBYc3','TRFRBYc3'};
    existing = intersect(dropNames, T.Properties.VariableNames, 'stable');
    T_clean = T;
    if ~isempty(existing)
        T_clean(:, existing) = [];
    end

    seriesNames = T_clean.Properties.VariableNames(2:end);
    F = T_clean{:,2:end};

    % Legend labels (if you want fixed order labels)
    legendLabels = { ...
        'TRDEBMc1','TRDEBMc2','TRDEBMc3','TRDEBMc4', ...
        'TRDEBQc1','TRDEBQc2','TRDEBQc3','TRDEBQc4', ...
        'TRDEBYc1','TRDEBYc2', ...
        'TRFRBMc1','TRFRBMc2','TRFRBMc3','TRFRBMc4', ...
        'TRFRBQc1','TRFRBQc2','TRFRBQc3','TRFRBQc4', ...
        'TRFRBYc1','TRFRBYc2' ...
    };

    % Optional quick plot of all series
    if opts.doPlots
        f = figure('WindowStyle','normal');
        plot(dates, F, 'LineWidth', 1.5);
        grid on;
        title('Time Series Plot');
        xlabel('Date');
        ylabel('Price');
        legend(legendLabels, 'Location', 'bestoutside', 'NumColumns', 2);
    end

    DATA = struct();
    DATA.T_raw = T;
    DATA.T_clean = T_clean;
    DATA.dates = dates;
    DATA.seriesNames = seriesNames;
    DATA.legendLabels = legendLabels;
    DATA.toRemove = toRemove;
    DATA.F_all = X_all;
    DATA.F = F;
end

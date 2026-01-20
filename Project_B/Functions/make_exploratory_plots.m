function FIG = make_exploratory_plots(DATA, DIAG_contracts, CURVES, opts)
%MAKE_EXPLORATORY_PLOTS Centralized plots (contracts + reconstructed monthly forwards).

    if ~isfield(opts, "maxLag"),  opts.maxLag  = 20; end
    if ~isfield(opts, "nMonths"), opts.nMonths = 24; end

    FIG = struct();

    dates = DATA.dates;
    F = DATA.F;
    legend_labels = DATA.legendLabels;
    contractNames = string(DATA.seriesNames);

    X_DE = CURVES.X_DE;
    X_FR = CURVES.X_FR;

    % -------------------------
    % 1) Separate DE/FR price series (gradients)
    % -------------------------
    colors_de = [linspace(0.5, 1, 10)', linspace(0, 0.4, 10)', linspace(0, 0.4, 10)'];
    colors_fr = [linspace(0, 0.3, 10)', linspace(0, 0.3, 10)', linspace(0.5, 1, 10)'];

    FIG.de_prices = figure('WindowStyle','normal'); hold on; grid on;
    for i = 1:10
        plot(dates, F(:,i), 'Color', colors_de(i,:), 'LineWidth', 1.5);
    end
    title('German Power Prices (DE) - Red Gradient');
    xlabel('Date'); ylabel('Price (€/MWh)');
    legend(legend_labels(1:10), 'Location', 'bestoutside', 'NumColumns', 1);

    FIG.fr_prices = figure('WindowStyle','normal'); hold on; grid on;
    for i = 1:10
        plot(dates, F(:,10+i), 'Color', colors_fr(i,:), 'LineWidth', 1.5);
    end
    title('French Power Prices (FR) - Blue Gradient');
    xlabel('Date'); ylabel('Price (€/MWh)');
    legend(legend_labels(11:20), 'Location', 'bestoutside', 'NumColumns', 1);

    % -------------------------
    % 2) Normality examples (contracts)
    % -------------------------
    R_contracts = DIAG_contracts.R;
    if isempty(R_contracts)
        % Fallback if DIAG_contracts does not store returns
        % (you can also pass returns in DIAG_contracts if you want)
    end

    % Use the same idx you used before
    idx = [3, 5, 9, 13, 15, 19];
    idx = idx(idx <= size(R_contracts,2));

    FIG.norm_contracts = local_plot_normality_examples(R_contracts, contractNames, idx, ...
        'Normality examples – Contract returns');

    % -------------------------
    % 3) ACF (contracts) split 10+10 like your original
    % -------------------------
    FIG.acf_contracts_1 = local_plot_acf_grid(R_contracts, contractNames, opts.maxLag, 1, min(10,size(R_contracts,2)), ...
        'Autocorrelation of Returns – First 10 Contracts');
    if size(R_contracts,2) > 10
        FIG.acf_contracts_2 = local_plot_acf_grid(R_contracts, contractNames, opts.maxLag, 11, min(20,size(R_contracts,2)), ...
            'Autocorrelation of Returns – Remaining 10 Contracts');
    end

    % -------------------------
    % 4) Samuelson + correlation surface (contracts)
    % -------------------------
    FIG.samuelson_contracts = figure;
    bar(DIAG_contracts.stats.Std);
    set(gca,'XTickLabel', contractNames, 'XTickLabelRotation',45);
    ylabel('Volatility');
    title('Samuelson effect: Vol goes down with delivery');
    grid on;

    FIG.corr_contracts = figure;
    imagesc(DIAG_contracts.C); colorbar;
    xticks(1:length(contractNames)); yticks(1:length(contractNames));
    xticklabels(contractNames); yticklabels(contractNames);
    xtickangle(45);
    title('Correlation surface between contract log-returns');

    % -------------------------
    % 5) Monthly forward returns DE/FR (normality + ACF + Samuelson + corr)
    % -------------------------
    R_X_DE = diff(log(X_DE)); R_X_DE(~isfinite(R_X_DE)) = NaN;
    R_X_FR = diff(log(X_FR)); R_X_FR(~isfinite(R_X_FR)) = NaN;

    names_DE = "DE_M" + string(1:size(R_X_DE,2));
    names_FR = "FR_M" + string(1:size(R_X_FR,2));

    idx_DE = unique(round(linspace(1, size(R_X_DE,2), min(6,size(R_X_DE,2)))));
    idx_FR = unique(round(linspace(1, size(R_X_FR,2), min(6,size(R_X_FR,2)))));

    FIG.norm_X_DE = local_plot_normality_examples(R_X_DE, names_DE, idx_DE, ...
        'Normality examples – Monthly forward returns (Germany)');
    FIG.norm_X_FR = local_plot_normality_examples(R_X_FR, names_FR, idx_FR, ...
        'Normality examples – Monthly forward returns (France)');

    % ACF monthly forwards split in halves like your original
    FIG.acf_X_DE_1 = local_plot_acf_grid(R_X_DE, names_DE, opts.maxLag, 1, ceil(size(R_X_DE,2)/2), ...
        'Autocorrelation – Monthly Forward Returns (Germany, First Half)');
    FIG.acf_X_DE_2 = local_plot_acf_grid(R_X_DE, names_DE, opts.maxLag, ceil(size(R_X_DE,2)/2)+1, size(R_X_DE,2), ...
        'Autocorrelation – Monthly Forward Returns (Germany, Second Half)');

    FIG.acf_X_FR_1 = local_plot_acf_grid(R_X_FR, names_FR, opts.maxLag, 1, ceil(size(R_X_FR,2)/2), ...
        'Autocorrelation – Monthly Forward Returns (France, First Half)');
    FIG.acf_X_FR_2 = local_plot_acf_grid(R_X_FR, names_FR, opts.maxLag, ceil(size(R_X_FR,2)/2)+1, size(R_X_FR,2), ...
        'Autocorrelation – Monthly Forward Returns (France, Second Half)');

    FIG.samuelson_X_DE = figure;
    bar(std(R_X_DE, 0, 1, 'omitnan'));
    xlabel('Delivery month'); ylabel('Volatility');
    title('Volatility term structure – Monthly forwards (Germany)');
    grid on;

    FIG.samuelson_X_FR = figure;
    bar(std(R_X_FR, 0, 1, 'omitnan'));
    xlabel('Delivery month'); ylabel('Volatility');
    title('Volatility term structure – Monthly forwards (France)');
    grid on;

    FIG.corr_X_DE = figure;
    imagesc(corr(R_X_DE, 'Rows','pairwise')); colorbar;
    xlabel('Delivery month'); ylabel('Delivery month');
    title('Correlation surface of monthly forward returns – Germany');

    FIG.corr_X_FR = figure;
    imagesc(corr(R_X_FR, 'Rows','pairwise')); colorbar;
    xlabel('Delivery month'); ylabel('Delivery month');
    title('Correlation surface of monthly forward returns – France');

    % -------------------------
    % 6) Monthly forward curves snapshots
    % -------------------------
    FIG.curves_FR = figure; hold on; grid on;
    idxDates = unique(round(linspace(1, size(X_FR,1), 6)));
    for i = idxDates
        plot(1:size(X_FR,2), X_FR(i,:), 'LineWidth', 1.5);
    end
    xlabel('Delivery Month'); ylabel('Price (€/MWh)');
    title('Monthly Forward Curves – France (Selected Dates)');

    FIG.curves_DE = figure; hold on; grid on;
    idxDates = unique(round(linspace(1, size(X_DE,1), 6)));
    for i = idxDates
        plot(1:size(X_DE,2), X_DE(i,:), 'LineWidth', 1.5);
    end
    xlabel('Delivery Month'); ylabel('Price (€/MWh)');
    title('Monthly Forward Curves – Germany (Selected Dates)');
end

% =========================
% Local helpers
% =========================
function fig = local_plot_normality_examples(R, names, idx, figTitle)
    fig = figure;
    tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

    for k = 1:min(6,numel(idx))
        j = idx(k);
        x = R(:,j);
        x = x(isfinite(x));
        if numel(x) < 30 || std(x) == 0, continue; end

        mu = mean(x);
        sigma = std(x);

        nexttile;
        histogram(x,'Normalization','pdf'); hold on;
        xx = linspace(min(x), max(x), 200);
        plot(xx, normpdf(xx, mu, sigma), 'r', 'LineWidth', 2);
        grid on;
        title(names(j), 'Interpreter','none');
    end
    sgtitle(figTitle);
end

function fig = local_plot_acf_grid(R, names, maxLag, jStart, jEnd, figTitle)
    nCols = 3;
    nPlots = max(0, jEnd - jStart + 1);
    nRows = max(1, ceil(nPlots / nCols));

    fig = figure;
    tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','compact');

    for j = jStart:jEnd
        x = R(:,j);
        x = x(isfinite(x));
        if numel(x) < 30 || std(x) == 0, continue; end

        [acf, lags] = xcorr(x - mean(x), maxLag, 'coeff');
        conf = 1.96 / sqrt(length(x));

        nexttile;
        idxPos = lags >= 0;
        stem(lags(idxPos), acf(idxPos), 'filled'); hold on;
        yline(conf,  'r--', 'LineWidth', 1);
        yline(-conf, 'r--', 'LineWidth', 1);
        yline(0, 'k-', 'LineWidth', 0.8);
        grid on;
        title(names(j), 'Interpreter','none');
        xlabel('Lag'); ylabel('ACF');
    end
    sgtitle(figTitle);
end

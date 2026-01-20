function fig = p2_make_plots(dates, F, legendLabels, T, P2, opts)
%P2_MAKE_PLOTS Generates all Point 2 plots (optional).
%
% Output:
%   fig struct with figure handles (so you can reuse/export later)

    fig = struct();

    % -------------------------
    % A) Separate DE/FR price series
    % -------------------------
    colors_de = [linspace(0.5, 1, 10)', linspace(0, 0.4, 10)', linspace(0, 0.4, 10)'];
    colors_fr = [linspace(0, 0.3, 10)', linspace(0, 0.3, 10)', linspace(0.5, 1, 10)'];

    fig.de_prices = figure('WindowStyle','normal'); hold on; grid on;
    for i = 1:10
        plot(dates, F(:,i), 'Color', colors_de(i,:), 'LineWidth', 1.5);
    end
    title('German Power Prices (DE) - Red Gradient');
    xlabel('Date'); ylabel('Price (€/MWh)');
    legend(legendLabels(1:10), 'Location', 'bestoutside', 'NumColumns', 1);

    fig.fr_prices = figure('WindowStyle','normal'); hold on; grid on;
    for i = 1:10
        plot(dates, F(:,10+i), 'Color', colors_fr(i,:), 'LineWidth', 1.5);
    end
    title('French Power Prices (FR) - Blue Gradient');
    xlabel('Date'); ylabel('Price (€/MWh)');
    legend(legendLabels(11:20), 'Location', 'bestoutside', 'NumColumns', 1);

    % -------------------------
    % B) Normality examples (contracts)
    % -------------------------
    R = P2.R_contracts;
    stats = P2.diag_contracts.stats;
    idx = [3, 5, 9, 13, 15, 19];
    idx = idx(idx <= size(R,2));

    fig.norm_contracts = figure;
    tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

    for k = 1:numel(idx)
        j = idx(k);
        x = R(:,j); x = x(isfinite(x));
        if numel(x) < 30 || std(x) == 0, continue; end

        mu = mean(x); sigma = std(x);

        nexttile;
        histogram(x, 'Normalization', 'pdf'); hold on;
        xx = linspace(min(x), max(x), 200);
        plot(xx, normpdf(xx, mu, sigma), 'r', 'LineWidth', 2);
        grid on;
        title(string(stats.Contract(j)), 'Interpreter','none');
    end

    % -------------------------
    % C) ACF plots (contracts)
    % -------------------------
    maxLag = opts.maxLag;
    fig.acf_contracts = local_plot_acf_panels(R, string(P2.contractNames), maxLag, ...
        'Autocorrelation of Returns – Contracts');

    % -------------------------
    % D) Samuelson effect (contracts)
    % -------------------------
    fig.samuelson_contracts = figure;
    bar(P2.diag_contracts.stats.Std);
    set(gca,'XTickLabel', T.Properties.VariableNames(2:end), 'XTickLabelRotation',45);
    ylabel('Volatility');
    title('Samuelson effect: Vol goes down with delivery');
    grid on;

    % -------------------------
    % E) Correlation surface (contracts)
    % -------------------------
    fig.corr_contracts = figure;
    imagesc(P2.diag_contracts.C); colorbar;
    xticks(1:length(P2.contractNames)); yticks(1:length(P2.contractNames));
    xticklabels(P2.contractNames); yticklabels(P2.contractNames);
    xtickangle(45);
    title('Correlation surface between contract log-returns');

    % -------------------------
    % F) Monthly forward diagnostics plots (DE/FR)
    % -------------------------
    X_DE = P2.X_DE;  X_FR = P2.X_FR;

    R_X_DE = diff(log(X_DE)); R_X_DE(~isfinite(R_X_DE)) = NaN;
    R_X_FR = diff(log(X_FR)); R_X_FR(~isfinite(R_X_FR)) = NaN;

    % ACF monthly forwards
    fig.acf_X_DE = local_plot_acf_panels(R_X_DE, "DE_M"+string(1:size(R_X_DE,2)), maxLag, ...
        'Autocorrelation – Monthly Forward Returns (Germany)');
    fig.acf_X_FR = local_plot_acf_panels(R_X_FR, "FR_M"+string(1:size(R_X_FR,2)), maxLag, ...
        'Autocorrelation – Monthly Forward Returns (France)');

    % Samuelson-type effect monthly forwards
    fig.samuelson_X_DE = figure;
    bar(std(R_X_DE,0,1,'omitnan'));
    xlabel('Delivery month'); ylabel('Volatility');
    title('Volatility term structure – Monthly forwards (Germany)');
    grid on;

    fig.samuelson_X_FR = figure;
    bar(std(R_X_FR,0,1,'omitnan'));
    xlabel('Delivery month'); ylabel('Volatility');
    title('Volatility term structure – Monthly forwards (France)');
    grid on;

    % Correlation surfaces monthly forwards
    fig.corr_X_DE = figure;
    imagesc(corr(R_X_DE,'Rows','pairwise')); colorbar;
    xlabel('Delivery month'); ylabel('Delivery month');
    title('Correlation surface – Monthly forward returns (Germany)');

    fig.corr_X_FR = figure;
    imagesc(corr(R_X_FR,'Rows','pairwise')); colorbar;
    xlabel('Delivery month'); ylabel('Delivery month');
    title('Correlation surface – Monthly forward returns (France)');

    % -------------------------
    % G) Monthly forward curves snapshots
    % -------------------------
    fig.curves_FR = figure; hold on; grid on;
    idxDates = unique(round(linspace(1, size(X_FR,1), 6)));
    for i = idxDates
        plot(1:size(X_FR,2), X_FR(i,:), 'LineWidth', 1.5);
    end
    xlabel('Delivery Month'); ylabel('Price (€/MWh)');
    title('Monthly Forward Curves – France (Selected Dates)');

    fig.curves_DE = figure; hold on; grid on;
    idxDates = unique(round(linspace(1, size(X_DE,1), 6)));
    for i = idxDates
        plot(1:size(X_DE,2), X_DE(i,:), 'LineWidth', 1.5);
    end
    xlabel('Delivery Month'); ylabel('Price (€/MWh)');
    title('Monthly Forward Curves – Germany (Selected Dates)');
end

% =====================================================================
% Local helper (kept inside the same file to avoid extra Functions files)
% =====================================================================
function figHandle = local_plot_acf_panels(R, names, maxLag, superTitle)
    nComp = size(R,2);
    nCols = 3;

    % First half
    nFirst = min(ceil(nComp/2), nComp);
    nRows1 = ceil(nFirst / nCols);

    figHandle(1) = figure;
    tiledlayout(nRows1, nCols, 'TileSpacing','compact', 'Padding','compact');

    for j = 1:nFirst
        x = R(:,j); x = x(isfinite(x));
        if numel(x) < 30 || std(x) == 0, continue; end

        [acf, lags] = xcorr(x - mean(x), maxLag, 'coeff');
        conf = 1.96 / sqrt(length(x));

        nexttile;
        idxPos = lags >= 0;
        stem(lags(idxPos), acf(idxPos), 'filled'); hold on;
        yline(conf,'r--','LineWidth',1);
        yline(-conf,'r--','LineWidth',1);
        yline(0,'k-','LineWidth',0.8);
        grid on;
        title(names(j), 'Interpreter','none');
        xlabel('Lag'); ylabel('ACF');
    end
    sgtitle(superTitle + " (First Half)");

    % Second half
    idxStart = nFirst + 1;
    idxEnd   = nComp;
    nSecond  = idxEnd - idxStart + 1;
    nRows2   = ceil(nSecond / nCols);

    figHandle(2) = figure;
    tiledlayout(nRows2, nCols, 'TileSpacing','compact', 'Padding','compact');

    for j = idxStart:idxEnd
        x = R(:,j); x = x(isfinite(x));
        if numel(x) < 30 || std(x) == 0, continue; end

        [acf, lags] = xcorr(x - mean(x), maxLag, 'coeff');
        conf = 1.96 / sqrt(length(x));

        nexttile;
        idxPos = lags >= 0;
        stem(lags(idxPos), acf(idxPos), 'filled'); hold on;
        yline(conf,'r--','LineWidth',1);
        yline(-conf,'r--','LineWidth',1);
        yline(0,'k-','LineWidth',0.8);
        grid on;
        title(names(j), 'Interpreter','none');
        xlabel('Lag'); ylabel('ACF');
    end
    sgtitle(superTitle + " (Second Half)");
end

clc
clear all
close all

%%
addpath("Functions");
addpath("PCA");
addpath("Pricing Chooser");

%% futures curves
T = readtable('power_DE_FR.csv');

% estraggo solo i dati numerici (escludo la colonna date)
X = T{:, 2:end};

% condizioni sui NaN per ciascun blocco
cond1 = sum(isnan(X(:, 1:5)),  2) > 2;
cond2 = sum(isnan(X(:, 6:9)),  2) > 2;
cond3 = sum(isnan(X(:,10:11)), 2) > 1;
cond4 = sum(isnan(X(:,12:15)), 2) > 2;
cond5 = sum(isnan(X(:,16:19)), 2) > 2;
cond6 = sum(isnan(X(:,20:21)), 2) > 1;

% righe da eliminare
toRemove = cond1 | cond2 | cond3 | cond4 | cond5 | cond6;

dates = T.date;
contractNames = T.Properties.VariableNames(2:end);
T(:, {'TRDEBYc3','TRFRBYc3'}) = [];
F = T{:, 2:end};   % tutte le colonne prezzo rimaste
f = figure();
set(f, 'WindowStyle', 'normal'); 

% PLOT DIRETTO
plot(dates, F, 'LineWidth', 1.5); 

grid on
title('Time Series Plot')
xlabel('Date')
ylabel('Price')

% Definizione delle etichette della legenda
legend_labels = { ...
    'TRDEBMc1', 'TRDEBMc2', 'TRDEBMc3', 'TRDEBMc4', ... % Germany Monthly
    'TRDEBQc1', 'TRDEBQc2', 'TRDEBQc3', 'TRDEBQc4', ... % Germany Quarterly
    'TRDEBYc1', 'TRDEBYc2', ...                         % Germany Yearly
    'TRFRBMc1', 'TRFRBMc2', 'TRFRBMc3', 'TRFRBMc4', ... % France Monthly
    'TRFRBQc1', 'TRFRBQc2', 'TRFRBQc3', 'TRFRBQc4', ... % France Quarterly
    'TRFRBYc1', 'TRFRBYc2'                              % France Yearly
};

% Creazione della legenda
legend(legend_labels, 'Location', 'bestoutside', 'NumColumns', 2);

%% Plot Separati: Germania e Francia
colors_de = [linspace(0.5, 1, 10)', linspace(0, 0.4, 10)', linspace(0, 0.4, 10)'];
colors_fr = [linspace(0, 0.3, 10)', linspace(0, 0.3, 10)', linspace(0.5, 1, 10)'];

% --- PLOT GERMANIA (Sfumature Rosse) ---
f1 = figure(); 
set(f1, 'WindowStyle', 'normal');
hold on; grid on;

% Ciclo per plottare ogni linea col suo colore specifico
for i = 1:10
    plot(dates, F(:, i), 'Color', colors_de(i,:), 'LineWidth', 1.5);
end

title('German Power Prices (DE) - Red Gradient')
xlabel('Date'); ylabel('Price (€/MWh)')
% Legenda Germania
legend(legend_labels(1:10), 'Location', 'bestoutside', 'NumColumns', 1);

% --- PLOT FRANCIA ---
f2 = figure(); 
set(f2, 'WindowStyle', 'normal'); 
hold on; grid on;
for i = 1:10
    plot(dates, F(:, 10+i), 'Color', colors_fr(i,:), 'LineWidth', 1.5);
end

title('French Power Prices (FR) - Blue Gradient')
xlabel('Date'); ylabel('Price (€/MWh)')
% Legenda Francia
legend(legend_labels(11:20), 'Location', 'bestoutside', 'NumColumns', 1);%% Handling Nan

% Should we care about missing data?

nCols = size(F,2);

fprintf('=== Missing Data Analysis ===\n');
for j = 1:nCols
   
    x = F(:,j);
    isn = isnan(x);
    
    % Total number of NaN
    totalNan = sum(isn);

    fprintf('%-10s | NaN : %4d |\n ', ...
        T.Properties.VariableNames{j+1}, totalNan);
end
figure()
spy(isnan(F))
xlabel('Series');
ylabel('Time');
title('NaN map');
pbaspect([1 1 1]); %just for visualization

% IMPORTANT: These data are "gentle", in reality it is quite common to
% have they harmful. Manage them is part of the job. You should choose a
% technique for dealing with them according to some criterion based on your
% situation and goal. This holds true both for missing data and other
% problems such as outliers and so on.

% for j = 2:width(T)
%     T{:,j} = fillmissing(T{:,j}, 'linear');
%     T{:,j} = fillmissing(T{:,j}, 'previous'); % Alternative
%     T{:,j} = fillmissing(T{:,j}, 'next'); % Alternative
% end
As=zeros(10,36,12);
for i = 1:12
    
    As(:,:,i)=buildContractMonthWeights(i);
end

%Take log-returns
R = diff(log(T{:,2:end}));

%%
disc_table=readtable("discount_factors.xlsx");
dates=table2array(disc_table(1,:));
datesDisc=datetime(dates, 'ConvertFrom', 'excel');
discounts=table2array(disc_table(2,:));
T_DE = T(:,1:11);
T_FR = [T(:,1) T(:,12:end)];

%%
lambda = 1e-3;               
nRows = height(T_DE);
nMonths = 36;
tic
X_DE = NaN(nRows, nMonths);   % output finale

for i = 1:nRows
    % --- data della riga ---
    d = T_DE.date(i);
    % --- prezzi (1x10) ---
    b = T_DE{i, 2:end};   % Mc1..Mc4 Qc1..Qc4 Yc1 Yc2
    % --- costruisco curva mensile ---
    mu = build_mu_fixed_columns(d, b);
    % --- salvo ---
    X_DE(i, :) = mu.';
end

X_DE=X_DE(:,1:24);
X_FR = NaN(nRows, nMonths);   % output finale

for i = 1:nRows
    % --- data della riga ---
    d = T_FR.date(i);
    % --- prezzi (1x10) ---
    b = T_FR{i, 2:end};   % Mc1..Mc4 Qc1..Qc4 Yc1 Yc2
    % --- costruisco curva mensile ---
    mu = build_mu_fixed_columns(d, b);
    % --- salvo ---
    X_FR(i, :) = mu.';
end

X_FR=X_FR(:,1:24);
toc

%% Distribution: Empirical fit vs theoretical gaussian
% =========================
% Momenti primi e secondi
% =========================
meanR = mean(R, 1, 'omitnan');
stdR  = std(R, 0, 1, 'omitnan');   % 0 = stima corretta (finanza)

% =========================
% Skewness e Kurtosis
% (version-proof, NaN-safe)
% =========================
nComp = size(R,2);

skewR = NaN(1, nComp);
kurtR = NaN(1, nComp);

for j = 1:nComp
    x = R(:,j);
    x = x(~isnan(x));   % rimuove NaN

    if numel(x) > 2     % evita warning inutili
        skewR(j) = skewness(x);
        kurtR(j) = kurtosis(x);
    end
end

% =========================
% Tabella descrittiva
% =========================
contracts = T.Properties.VariableNames(2:end)';

stats = table( ...
    contracts, ...
    meanR', ...
    stdR', ...
    skewR', ...
    kurtR', ...
    'VariableNames', {'Contract','Mean','Std','Skew','Kurtosis'} );

disp(stats)

% With gaussian returns I have no reason to introduce Levy drivers. Let's 
% give a preliminary Look at the skewness and kurtosis values... 

% Stationarity: test ADF
for i = 1:size(R,2)
    [h,p] = adftest(R(:,i)); %h=1-> reject null hp, p is p-value
    fprintf('%s: adftest h=%d, p=%.3f\n', T.Properties.VariableNames{i}, h, p); %h=1->stationary
end

% Gaussianity: Jarque-Bera test
for i = 1:size(R,2)
    [h,p] = jbtest(R(:,i));
    fprintf('%s: jbtest h=%d, p=%.3f\n', T.Properties.VariableNames{i}, h, p);%h=1->not gaussian
end

% Examples of Plots of Normality
idx = [3, 5, 9, 13, 15, 19];

figure;
tiledlayout(2,3, 'TileSpacing','compact', 'Padding','compact');

for k = 1:length(idx)
    j = idx(k);

    x = R(:,j);
    x = x(~isnan(x));

    mu = mean(x);
    sigma = std(x);

    nexttile;
    histogram(x, 'Normalization', 'pdf');
    hold on;

    xx = linspace(min(x), max(x), 200);
    plot(xx, normpdf(xx, mu, sigma), 'r', 'LineWidth', 2);

    grid on;
    title(stats.Contract{j}, 'Interpreter','none');
end

%% Autocorrelation analysis of returns R
nComp  = size(R,2);
maxLag = 20;          % numero di lag
nCols  = 3;           % colonne della griglia

% =========================
% FIGURA 1: primi 10
% =========================
nFirst = min(10, nComp);
nRows1 = ceil(nFirst / nCols);

figure;
tiledlayout(nRows1, nCols, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for j = 1:nFirst

    x = R(:,j);
    x = x(~isnan(x));   % rimuove NaN

    if numel(x) < 30
        continue
    end

    [acf, lags] = xcorr(x - mean(x), maxLag, 'coeff');

    Tlen = length(x);
    conf = 1.96 / sqrt(Tlen);

    nexttile;
    idxPos = lags >= 0;

    stem(lags(idxPos), acf(idxPos), 'filled'); 
    hold on;
    yline(conf,  'r--', 'LineWidth', 1);
    yline(-conf, 'r--', 'LineWidth', 1);
    yline(0, 'k-', 'LineWidth', 0.8);

    grid on;
    title(contractNames{j}, 'Interpreter','none');
    xlabel('Lag');
    ylabel('ACF');
end

sgtitle('Autocorrelation of Returns – First 10 Contracts');

% =========================
% FIGURA 2: restanti 10
% =========================
idxStart = 11;
idxEnd   = min(20, nComp);
nSecond  = idxEnd - idxStart + 1;
nRows2   = ceil(nSecond / nCols);

figure;
tiledlayout(nRows2, nCols, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for j = idxStart:idxEnd

    x = R(:,j);
    x = x(~isnan(x));

    if numel(x) < 30
        continue
    end

    [acf, lags] = xcorr(x - mean(x), maxLag, 'coeff');

    Tlen = length(x);
    conf = 1.96 / sqrt(Tlen);

    nexttile;
    idxPos = lags >= 0;

    stem(lags(idxPos), acf(idxPos), 'filled'); 
    hold on;
    yline(conf,  'r--', 'LineWidth', 1);
    yline(-conf, 'r--', 'LineWidth', 1);
    yline(0, 'k-', 'LineWidth', 0.8);

    grid on;
    title(contractNames{j}, 'Interpreter','none');
    xlabel('Lag');
    ylabel('ACF');
end

sgtitle('Autocorrelation of Returns – Remaining 10 Contracts');

%% Samuelson effect
vols = stdR;

figure();
bar(vols);
set(gca,'XTickLabel', T.Properties.VariableNames(2:end), 'XTickLabelRotation',45);
ylabel('Volatility');
title('Samuelson effect: Vol goes down with delivery');
grid on;

%% Correlation surface 
C = corr(R, 'Rows','pairwise');
figure;
imagesc(C); colorbar;
xticks(1:length(C)); yticks(1:length(C));
xticklabels(T.Properties.VariableNames(2:end));
yticklabels(T.Properties.VariableNames(2:end));
xtickangle(45);
title('Superficie di correlazione tra log-returns');

%% Correlation on longer horizon
R = diff(log(T{:,2:end}));
datesR = T.date(2:end);
TT = array2timetable(R, 'RowTimes', datesR, ...
    'VariableNames', T.Properties.VariableNames(2:end));
R_w = retime(TT, 'weekly', 'mean');
R_m = retime(TT, 'monthly', 'mean');
corr_w = corr(R_w.Variables,'Rows','pairwise');
corr_m = corr(R_m.Variables,'Rows','pairwise');

%% POINT 2 — Distribution: Empirical fit vs theoretical Gaussian
% GERMANY - Monthly forward returns
% =========================
% Log-returns (robusti)
% =========================
R_X_DE = diff(log(X_DE));
R_X_DE(~isfinite(R_X_DE)) = NaN;

nComp_DE = size(R_X_DE,2);

% =========================
% First and second moments
% =========================
meanR_DE = mean(R_X_DE, 1, 'omitnan');
stdR_DE  = std(R_X_DE, 0, 1, 'omitnan');

% =========================
% Skewness and Kurtosis
% =========================
skewR_DE = NaN(1, nComp_DE);
kurtR_DE = NaN(1, nComp_DE);

for j = 1:nComp_DE
    x = R_X_DE(:,j);
    x = x(isfinite(x));

    if numel(x) > 20 && std(x) > 0
        skewR_DE(j) = skewness(x);
        kurtR_DE(j) = kurtosis(x);
    end
end

% =========================
% Summary table
% =========================
contracts_DE = strcat("DE_M", string(1:nComp_DE))';

stats_X_DE = table( ...
    contracts_DE, ...
    meanR_DE', ...
    stdR_DE', ...
    skewR_DE', ...
    kurtR_DE', ...
    'VariableNames', {'Contract','Mean','Std','Skew','Kurtosis'} );

disp(stats_X_DE)

% =========================
% Stationarity: ADF test
% =========================
for j = 1:nComp_DE
    x = R_X_DE(:,j);
    x = x(isfinite(x));

    if numel(x) > 30 && std(x) > 0
        [h,p] = adftest(x);
        fprintf('%s: adftest h=%d, p=%.3f\n', contracts_DE(j), h, p);
    end
end

% =========================
% Gaussianity: Jarque-Bera
% =========================
for j = 1:nComp_DE
    x = R_X_DE(:,j);
    x = x(isfinite(x));

    if numel(x) > 30 && std(x) > 0
        [h,p] = jbtest(x);
        fprintf('%s: jbtest h=%d, p=%.3f\n', contracts_DE(j), h, p);
    end
end

% =========================
% Examples of Normality plots
% =========================
idx_DE = unique(round(linspace(1, nComp_DE, min(6,nComp_DE))));

figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

for k = 1:length(idx_DE)
    j = idx_DE(k);

    x = R_X_DE(:,j);
    x = x(isfinite(x));

    if numel(x) < 30 || std(x) == 0
        continue
    end

    mu = mean(x);
    sigma = std(x);

    nexttile;
    histogram(x,'Normalization','pdf');
    hold on;

    xx = linspace(min(x), max(x), 200);
    plot(xx, normpdf(xx, mu, sigma),'r','LineWidth',2);

    grid on;
    title(contracts_DE(j),'Interpreter','none');
end

%% POINT 2 — Distribution: Empirical fit vs theoretical Gaussian
% FRANCE - Monthly forward returns
% =========================
% Log-returns (robusti)
% =========================
R_X_FR = diff(log(X_FR));
R_X_FR(~isfinite(R_X_FR)) = NaN;

nComp_FR = size(R_X_FR,2);

% =========================
% First and second moments
% =========================
meanR_FR = mean(R_X_FR, 1, 'omitnan');
stdR_FR  = std(R_X_FR, 0, 1, 'omitnan');

% =========================
% Skewness and Kurtosis
% =========================
skewR_FR = NaN(1, nComp_FR);
kurtR_FR = NaN(1, nComp_FR);

for j = 1:nComp_FR
    x = R_X_FR(:,j);
    x = x(isfinite(x));

    if numel(x) > 20 && std(x) > 0
        skewR_FR(j) = skewness(x);
        kurtR_FR(j) = kurtosis(x);
    end
end

% =========================
% Summary table
% =========================
contracts_FR = strcat("FR_M", string(1:nComp_FR))';

stats_X_FR = table( ...
    contracts_FR, ...
    meanR_FR', ...
    stdR_FR', ...
    skewR_FR', ...
    kurtR_FR', ...
    'VariableNames', {'Contract','Mean','Std','Skew','Kurtosis'} );

disp(stats_X_FR)

% =========================
% Stationarity: ADF test
% =========================
for j = 1:nComp_FR
    x = R_X_FR(:,j);
    x = x(isfinite(x));

    if numel(x) > 30 && std(x) > 0
        [h,p] = adftest(x);
        fprintf('%s: adftest h=%d, p=%.3f\n', contracts_FR(j), h, p);
    end
end

% =========================
% Gaussianity: Jarque-Bera
% =========================
for j = 1:nComp_FR
    x = R_X_FR(:,j);
    x = x(isfinite(x));

    if numel(x) > 30 && std(x) > 0
        [h,p] = jbtest(x);
        fprintf('%s: jbtest h=%d, p=%.3f\n', contracts_FR(j), h, p);
    end
end

% =========================
% Examples of Normality plots
% =========================
idx_FR = unique(round(linspace(1, nComp_FR, min(6,nComp_FR))));

figure;
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

for k = 1:length(idx_FR)
    j = idx_FR(k);

    x = R_X_FR(:,j);
    x = x(isfinite(x));

    if numel(x) < 30 || std(x) == 0
        continue
    end

    mu = mean(x);
    sigma = std(x);

    nexttile;
    histogram(x,'Normalization','pdf');
    hold on;

    xx = linspace(min(x), max(x), 200);
    plot(xx, normpdf(xx, mu, sigma),'r','LineWidth',2);

    grid on;
    title(contracts_FR(j),'Interpreter','none');
end

%% POINT 2 — Autocorrelation analysis
% GERMANY - Monthly forward returns

R_X_DE = diff(log(X_DE));
R_X_DE(~isfinite(R_X_DE)) = NaN;

nComp  = size(R_X_DE,2);
maxLag = 20;
nCols  = 3;

% =========================
% FIGURE 1: first half
% =========================
nFirst = min(ceil(nComp/2), nComp);
nRows1 = ceil(nFirst / nCols);

figure;
tiledlayout(nRows1, nCols, 'TileSpacing','compact','Padding','compact');

for j = 1:nFirst

    x = R_X_DE(:,j);
    x = x(isfinite(x));

    if numel(x) < 30 || std(x)==0
        continue
    end

    [acf, lags] = xcorr(x-mean(x), maxLag, 'coeff');
    conf = 1.96 / sqrt(length(x));

    nexttile;
    idxPos = lags >= 0;

    stem(lags(idxPos), acf(idxPos), 'filled');
    hold on;
    yline(conf,'r--','LineWidth',1);
    yline(-conf,'r--','LineWidth',1);
    yline(0,'k-','LineWidth',0.8);

    grid on;
    title(['DE\_M' num2str(j)], 'Interpreter','none');
    xlabel('Lag');
    ylabel('ACF');
end

sgtitle('Autocorrelation of Monthly Forward Returns – Germany (First Half)');

% =========================
% FIGURE 2: second half
% =========================
idxStart = nFirst + 1;
idxEnd   = nComp;
nSecond  = idxEnd - idxStart + 1;
nRows2   = ceil(nSecond / nCols);

figure;
tiledlayout(nRows2, nCols, 'TileSpacing','compact','Padding','compact');

for j = idxStart:idxEnd

    x = R_X_DE(:,j);
    x = x(isfinite(x));

    if numel(x) < 30 || std(x)==0
        continue
    end

    [acf, lags] = xcorr(x-mean(x), maxLag, 'coeff');
    conf = 1.96 / sqrt(length(x));

    nexttile;
    idxPos = lags >= 0;

    stem(lags(idxPos), acf(idxPos), 'filled');
    hold on;
    yline(conf,'r--','LineWidth',1);
    yline(-conf,'r--','LineWidth',1);
    yline(0,'k-','LineWidth',0.8);

    grid on;
    title(['DE\_M' num2str(j)], 'Interpreter','none');
    xlabel('Lag');
    ylabel('ACF');
end

sgtitle('Autocorrelation of Monthly Forward Returns – Germany (Second Half)');

%% POINT 2 — Autocorrelation analysis
% FRANCE - Monthly forward returns

R_X_FR = diff(log(X_FR));
R_X_FR(~isfinite(R_X_FR)) = NaN;

nComp  = size(R_X_FR,2);
maxLag = 20;
nCols  = 3;

% =========================
% FIGURE 1: first half
% =========================
nFirst = min(ceil(nComp/2), nComp);
nRows1 = ceil(nFirst / nCols);

figure;
tiledlayout(nRows1, nCols, 'TileSpacing','compact','Padding','compact');

for j = 1:nFirst

    x = R_X_FR(:,j);
    x = x(isfinite(x));

    if numel(x) < 30 || std(x)==0
        continue
    end

    [acf, lags] = xcorr(x-mean(x), maxLag, 'coeff');
    conf = 1.96 / sqrt(length(x));

    nexttile;
    idxPos = lags >= 0;

    stem(lags(idxPos), acf(idxPos), 'filled');
    hold on;
    yline(conf,'r--','LineWidth',1);
    yline(-conf,'r--','LineWidth',1);
    yline(0,'k-','LineWidth',0.8);

    grid on;
    title(['FR\_M' num2str(j)], 'Interpreter','none');
    xlabel('Lag');
    ylabel('ACF');
end

sgtitle('Autocorrelation of Monthly Forward Returns – France (First Half)');

% =========================
% FIGURE 2: second half
% =========================
idxStart = nFirst + 1;
idxEnd   = nComp;
nSecond  = idxEnd - idxStart + 1;
nRows2   = ceil(nSecond / nCols);

figure;
tiledlayout(nRows2, nCols, 'TileSpacing','compact','Padding','compact');

for j = idxStart:idxEnd

    x = R_X_FR(:,j);
    x = x(isfinite(x));

    if numel(x) < 30 || std(x)==0
        continue
    end

    [acf, lags] = xcorr(x-mean(x), maxLag, 'coeff');
    conf = 1.96 / sqrt(length(x));

    nexttile;
    idxPos = lags >= 0;

    stem(lags(idxPos), acf(idxPos), 'filled');
    hold on;
    yline(conf,'r--','LineWidth',1);
    yline(-conf,'r--','LineWidth',1);
    yline(0,'k-','LineWidth',0.8);

    grid on;
    title(['FR\_M' num2str(j)], 'Interpreter','none');
    xlabel('Lag');
    ylabel('ACF');
end

sgtitle('Autocorrelation of Monthly Forward Returns – France (Second Half)');

%% Samuelson-type effect — Germany (monthly forwards)
vols_DE = std(R_X_DE, 0, 1, 'omitnan');

figure;
bar(vols_DE);
xlabel('Delivery month');
ylabel('Volatility');
title('Volatility term structure – Monthly forwards (Germany)');
grid on;
% Correlation surface — Germany (monthly forwards)

C_DE = corr(R_X_DE, 'Rows','pairwise');

figure;
imagesc(C_DE);
colorbar;
xlabel('Delivery month');
ylabel('Delivery month');
title('Correlation surface of monthly forward returns – Germany');

%% Samuelson-type effect — France (monthly forwards)
vols_FR = std(R_X_FR, 0, 1, 'omitnan');

figure;
bar(vols_FR);
xlabel('Delivery month');
ylabel('Volatility');
title('Volatility term structure – Monthly forwards (France)');
grid on;

% Correlation surface — France (monthly forwards)

C_FR = corr(R_X_FR, 'Rows','pairwise');

figure;
imagesc(C_FR);
colorbar;
xlabel('Delivery month');
ylabel('Delivery month');
title('Correlation surface of monthly forward returns – France');

%% Monthly forward curves
idxDates = round(linspace(1, size(X_FR,1), 6));  % 6 date rappresentative

figure; hold on; grid on;
for i = idxDates
    plot(1:24, X_FR(i,:), 'LineWidth', 1.5);
end

xlabel('Delivery Month');
ylabel('Price (€/MWh)');
title('Monthly Forward Curves – France (Selected Dates)');


idxDates = round(linspace(1, size(X_DE,1), 6));  % 6 date rappresentative

figure; hold on; grid on;
for i = idxDates
    plot(1:24, X_DE(i,:), 'LineWidth', 1.5);
end

xlabel('Delivery Month');
ylabel('Price (€/MWh)');
title('Monthly Forward Curves – Germany (Selected Dates)');
% The figure shows selected snapshots of the reconstructed monthly forward curves.
% The curves are stable across maturities and over time, with a piecewise structure reflecting the contract aggregation.
% No irregular patterns are observed, supporting the robustness of the reconstruction.

%% Exercise 3 - Principal Component Analysis
X=[X_DE X_FR];

Ret = buildReturnsPCA(X, toRemove, T);
plotHistWithNormalFit(Ret);
Ret_pca = removeOutliersMahalanobis(Ret);
[eigenvectors, score, eigenvalues, tsq, explained, mu_ret] = pca(Ret_pca, 'Centered', true);
threshold = 90; 
[~, nComp] = plotPCAExplained(explained, eigenvalues, threshold);

%% Exercise 4 - Price chooser option
% Pricing closed form
gamma = diag(eigenvalues(1:nComp));
C = eigenvectors(:,1:nComp);
dt = 1/252;
sigma_sim = C*sqrt(gamma)*dt^(-0.5);
F1_0 = X_DE(end,17);
F2_0 = X_FR(end, 17);
sigma1 = sigma_sim(17,:);
sigma2 = sigma_sim(17+24,:);
t0 = datetime(2025,11,4);
T1 = datetime(2027,2,5);
ttm = yearfrac(t0, T1, 3);
discPricing = discount_interp(datesDisc,discounts,datetime(2027,2,5),t0);

price_closedform = priceClosedForm(F1_0,F2_0,sigma1,sigma2,ttm,discPricing);

% Pricing with Monte Carlo
rng(42);
nSim = 100000;
[price_MC, F1_end, F2_end] = priceMC(F1_0,F2_0,sigma1,sigma2,ttm,discPricing,nSim,nComp);

% Margrabe option pricing (price of max(F1,F2))
price_margrabe = priceMargrabe(F1_0, F2_0, sigma1, sigma2, ttm, discPricing);

% Lower & Upper bounds (da simulazioni MC)
[lower_bound, upper_bound] = boundsLowerUpper(F1_end, F2_end, discPricing);

fprintf('Closed form  : %.10f\n', price_closedform);
fprintf('Monte Carlo  : %.10f\n', price_MC);
fprintf('Margrabe max : %.10f\n', price_margrabe);
fprintf('Lower bound  : %.10f\n', lower_bound);
fprintf('Upper bound  : %.10f\n', upper_bound);

%% Exercise 5 - Print prices and checks
startDate = datetime(2027,11,1);
endDate   = datetime(2027,11,30);

datesVec = (startDate:endDate).';
datesVec = businessdayoffset(datesVec);
n_days = length(datesVec);

K = 40;
F0 = X_DE(end, 24);
sigma_DE = sigma_sim(24,:);

% Swing with N = 15 (assignment)
N15 = 15;
price_swing_15 = price_swing_option(t0, datesDisc, discounts, startDate, endDate, F0, sigma_DE, N15, K);

% Swing with N = number of business days (upper bound)
Nmax = n_days;
price_swing_Nmax = price_swing_option(t0, datesDisc, discounts, startDate, endDate, F0, sigma_DE, Nmax, K);

% Strip of daily European calls (upper-bound check)
price_strip = price_sum_calls(t0, datesDisc, discounts, startDate, endDate, F0, sigma_DE, K);

% Print
fprintf('\nSwing price (N = %d): %.10f\n', N15, price_swing_15);
fprintf('Swing price (N = %d): %.10f\n', Nmax, price_swing_Nmax);
fprintf('Sum of daily European calls (strip): %.10f\n\n', price_strip);
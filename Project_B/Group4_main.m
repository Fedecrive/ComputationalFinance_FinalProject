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

%% Exercise 2
opts = struct();
opts.doPlots = true;
opts.nMonths = 24;
opts.maxLag  = 20;

P2 = p2_run_point2(T, F, dates, legend_labels, "discount_factors.xlsx", opts);

% Variabili che ti servono dopo (come da tuo Exercise 3-5)
X_DE      = P2.X_DE;
X_FR      = P2.X_FR;
X         = P2.X;              % [X_DE X_FR]
datesDisc = P2.datesDisc;
discounts = P2.discounts;

% Se ti servono anche returns/diag:
R         = P2.R_contracts;
stats     = P2.diag_contracts.stats;

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

%% Plot Swing price with respect to N (centered + highlighted N=15 and N=22)
N_min = 1;
N_max = 50;

K40 = 40;
plot_swing_price_vs_N(N_min, N_max, t0, datesDisc, discounts, startDate, endDate, F0, sigma_DE, K40)

K90 = 90;
plot_swing_price_vs_N(N_min, N_max, t0, datesDisc, discounts, startDate, endDate, F0, sigma_DE, K90)

K100 = 100;
plot_swing_price_vs_N(N_min, N_max, t0, datesDisc, discounts, startDate, endDate, F0, sigma_DE, K100)
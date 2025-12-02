clear all
close all
clc

%% ==============================
%  1. DATA LOADING
% ==============================

addpath(genpath('Data'));
addpath(genpath('Exercise 3'));

% Price file (16 assets) and mapping table (Cyclical / Neutral / Defensive)
path_data   = '';   % leave empty if .csv files are in the same folder
price_file  = 'asset_prices.csv';
map_file    = 'mapping_table.csv';

% Prices 
tbl_prices  = readtable(fullfile(path_data, price_file));

dates_all   = tbl_prices{:,1};        % first column = dates
prices_all  = tbl_prices{:,2:end};    % columns 2..17 = prices
assetNames  = tbl_prices.Properties.VariableNames(2:end);

% Convert to timetable
myPrice_dt  = array2timetable(prices_all, 'RowTimes', dates_all, ...
                              'VariableNames', assetNames);

% Macro-group mapping
map_tbl = readtable(fullfile(path_data, map_file));

asset_id_map = map_tbl{:,1};          % asset code / name
group_label  = map_tbl{:,2};          % 'Cyclical' / 'Neutral' / 'Defensive'

% Align mapping order to the price column order
[~, idx_map] = ismember(assetNames, asset_id_map);
group_label  = group_label(idx_map);

% Logical indices for each macro-group
isCyclical = strcmpi(group_label, 'Cyclical');
isNeutral  = strcmpi(group_label, 'Neutral');
isDef      = strcmpi(group_label, 'Defensive');

%% ==============================
%  2. IN-SAMPLE PERIOD SELECTION
% ==============================

start_dt = datetime('01/01/2018', 'InputFormat', 'dd/MM/yyyy');
end_dt   = datetime('31/12/2022', 'InputFormat', 'dd/MM/yyyy');

rng_ = timerange(start_dt, end_dt, 'closed');
subsample = myPrice_dt(rng_, :);

prices = subsample.Variables;
dates  = subsample.Time;

%% ==============================
%  3. RETURNS & MOMENTS
% ==============================

% Daily log returns
LogRet = tick2ret(prices, 'Method', 'Continuous');    % T x N
ExpRet = mean(LogRet);                                % 1 x N
Sigma  = cov(LogRet);                                 % N x N
N      = size(LogRet, 2);

vol_i  = sqrt(diag(Sigma));                           % individual asset volatilities

% Annualization (252 trading days)
mu_ann   = ExpRet * 252;             % 1 x N
Sigma_ann = Sigma * 252;

rf = 0.00;   % risk-free rate for Sharpe ratio

%% ===========================================
%  4. COMMON CONSTRAINTS (G, H, EW)
% ============================================

% 0 ≤ wi ≤ 0.2
lb = zeros(N,1);
ub = 0.2 * ones(N,1);

% Sum of weights = 1
Aeq = ones(1,N);
beq = 1;

% Group constraints: each macro-group ≥ 15%
% sum_{i in group} w_i ≥ 0.15   ↔  -sum_{i in group} w_i ≤ -0.15
A = [];
b = [];

if any(isCyclical)
    rowC = zeros(1,N);
    rowC(isCyclical) = -1;
    A = [A; rowC];
    b = [b; -0.15];
end

if any(isNeutral)
    rowN = zeros(1,N);
    rowN(isNeutral) = -1;
    A = [A; rowN];
    b = [b; -0.15];
end

if any(isDef)
    rowD = zeros(1,N);
    rowD(isDef) = -1;
    A = [A; rowD];
    b = [b; -0.15];
end

% Initial guess
w0 = ones(N,1) / N;

opts = optimoptions('fmincon','Display','off', ...
                    'Algorithm','sqp','MaxIterations',1e8);

%% ===========================================
%  5. PORTFOLIO G: MAX DIVERSIFICATION RATIO
% ============================================

% Diversification ratio:
% DR(w) = [(w' * vol_i) / sqrt(w' * Sigma * w)]
fun_DR = @(w) - (w' * vol_i) / sqrt(w' * Sigma * w);   
[w_G, fval_G] = fmincon(fun_DR, w0, A, b, Aeq, beq, lb, ub, [], opts);

DR = -fval_G;   % maximum DR value

%% ===========================================
%  6. PORTFOLIO H: EQUAL RISK CONTRIBUTION
% ============================================

% Objective: equal risk contribution
% Minimize the dispersion of risk contributions
fun_RPC = @(w) riskParityObjective(w, Sigma);

opts_minmax = optimoptions('fminimax', ...
    'Display', 'off', ...
    'MaxIterations', 1e4, ...
    'ConstraintTolerance', 1e-8, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-10, ...
    'UseParallel', false);

[w_H, fval_H] = fminimax(fun_RPC, w0, A, b, Aeq, beq, lb, ub, [], opts_minmax);

%% ===========================================
%  7. BENCHMARK EQUALLY-WEIGHTED
% ============================================

w_EW = ones(N,1) / N;

%% ===========================================
%  8. METRICS: DR, VOL, SHARPE, N_eff
% ============================================

% Diversification Ratio
DR_G = getDiversificationRatio(w_G, LogRet);
DR_H = getDiversificationRatio(w_H, LogRet);
DR_EW = getDiversificationRatio(w_EW, LogRet);

% Annualized volatility
vol_G  = sqrt(w_G' * Sigma_ann * w_G);
vol_H  = sqrt(w_H' * Sigma_ann * w_H);
vol_EW = sqrt(w_EW' * Sigma_ann * w_EW);

% Annualized expected return
ret_G  = mu_ann * w_G;
ret_H  = mu_ann * w_H;
ret_EW = mu_ann * w_EW;

% Sharpe ratio
Sharpe_G  = (ret_G  - rf) / vol_G;
Sharpe_H  = (ret_H  - rf) / vol_H;
Sharpe_EW = (ret_EW - rf) / vol_EW;

% Effective number of assets (Herfindahl index)
Neff_G  = 1 / sum(w_G.^2);
Neff_H  = 1 / sum(w_H.^2);
Neff_EW = 1 / sum(w_EW.^2);

% Summary table
Results = table(...
    [DR_G; DR_H; DR_EW], ...
    [vol_G;  vol_H;  vol_EW], ...
    [Sharpe_G; Sharpe_H; Sharpe_EW], ...
    [Neff_G; Neff_H; Neff_EW], ...
    'VariableNames', {'DivRatio','Vol_Ann','Sharpe','N_eff'}, ...
    'RowNames', {'G_MaxDR','H_RiskParity','EW_Benchmark'});

disp('===== Exercise 3 Results (in-sample) =====')
disp(Results)

%% ===========================================
%  9. PLOT OF WEIGHTS
% ============================================

figure;
bar([w_G, w_H, w_EW])
legend({'G: Max DR','H: Risk Parity','EW'},'Location','bestoutside')
xlabel('Asset')
ylabel('Weight')
title('Comparison of G, H and Benchmark Portfolio Weights')
grid on

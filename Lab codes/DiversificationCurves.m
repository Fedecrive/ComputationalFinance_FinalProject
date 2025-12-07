clear all
close all
clc

% Beyond Mean–Variance: Exploring Diversification Frontiers
%% Read Prices
load myPrice_dt
load array_prices

%% Selection of a subset of Dates
start_dt = datetime('01/06/2020', 'InputFormat', 'dd/MM/yyyy'); 
end_dt   = datetime('01/01/2022', 'InputFormat', 'dd/MM/yyyy');

rng = timerange(start_dt, end_dt,'closed');
subsample = myPrice_dt(rng,:);

prices_val = subsample.Variables;
dates_ = subsample.Time;
%% Calculate returns and Covariance Matrix
LogRet = tick2ret(prices_val, 'Method', 'Continuous');
ExpRet = mean(LogRet);
V = cov(LogRet);
NumAssets = size(LogRet, 2);
%% Entropy frontier → How evenly the portfolio is allocated
% Portfolio Optimization
p = Portfolio('AssetList', assetNames);
p = setDefaultConstraints(p);
P = setAssetMoments(p, ExpRet, V);
pwgt = estimateFrontier(P,100);
[pf_Risk, pf_Retn] = estimatePortMoments(P,pwgt);

% Compute Entropy for each portfolio
% Entropy is -sum(w*log(w)) over positive weights
N = 100;
EntropyFrontier = zeros(1,N);
for i =1:N
    w = pwgt(:,i);
    w = w / sum(w);     % Normalize weights just in case
    w_pos = w(w > 0);   % Only consider strictly positive weights
    EntropyFrontier(i) = -sum(w_pos .* log(w_pos));
end

% Plot
plot(pf_Risk, EntropyFrontier, '-o', 'LineWidth', 4)
xlabel('Volatility')
ylabel('Diversification')

%% Compute max available entropy for each risk level
fun = @(x) x'*log(x); % Objective function (negentropy)
x0 = rand(NumAssets,1); % Initial guess
x0 = x0./sum(x0);

lb = zeros(1,NumAssets);
ub = ones(1,NumAssets);
MaxEntropyFrontier = zeros(1, N);

for i = 1:length(pf_Risk)
    vol_i = pf_Risk(i);  % Target volatility
    Aeq = ones(1,NumAssets);
    beq = 1;
    [w_opt, fval] = fmincon(fun, x0, [],[],Aeq, beq, lb, ub, @(x) nonlinConstr(x, V, vol_i));
    MaxEntropyFrontier(i) = -fval; % Store max entropy achievable at that risk
end 

% Plot Entropy Frontier vs Maximum Entropy
h = figure();
scatter(pf_Risk, EntropyFrontier, 'filled', 'g', 'LineWidth', 6)
hold on
scatter(pf_Risk, MaxEntropyFrontier, 'filled', 'm', 'LineWidth', 6)
grid on
legend('Entropy for Efficient Portfolios', 'Max Possible Entropy', 'fontsize', 16)
ylabel('Diversification', 'fontsize', 14)
xlabel('Volatility', 'fontsize', 14)

%% Diversification–Risk Frontier → How effectively risk is diversified
Sigma = cov(LogRet);
sigma_i = sqrt(diag(Sigma));

% Define target volatilities (grid)
targetVol = linspace(0.005, 0.05, 100); 
DR_vals = zeros(size(targetVol));
weights_DR_frontier = zeros(NumAssets, length(targetVol));

for k = 1:length(targetVol)
    sigma_target = targetVol(k);

    % Optimization: maximize Diversification Ratio with volatility constraint
    fun = @(w) - (w'*sigma_i) / sqrt(w'*Sigma*w);
    Aeq = ones(1,NumAssets); beq = 1;
    lb = zeros(NumAssets,1); ub = ones(NumAssets,1);

    % Nonlinear constraint: portfolio volatility <= target
    nonlcon = @(w) deal([], sqrt(w'*Sigma*w) - sigma_target);

    w0 = ones(NumAssets,1)/NumAssets;
    w_opt = fmincon(fun, w0, [], [], Aeq, beq, lb, ub, nonlcon);
    weights_DR_frontier(:,k) = w_opt;

    DR_vals(k) = getDiversificationRatio(w_opt, LogRet);
end

% Plot Diversification–Risk Frontier
figure;
plot(targetVol, DR_vals, 'LineWidth', 2)
xlabel('Portfolio Volatility')
ylabel('Diversification Ratio')
title('Diversification–Risk Frontier')
grid on

%% Compare MaxDivRatio with Mean–Variance Frontier
p = Portfolio('AssetList', assetNames);
p = estimateAssetMoments(p, LogRet);
p = setDefaultConstraints(p);
weights_MV = estimateFrontier(p, 100);
[risk_MV, ret_MV] = estimatePortMoments(p, weights_MV);

DR_MV = zeros(1, size(weights_MV,2));
for j = 1:size(weights_MV,2)
    DR_MV(j) = getDiversificationRatio(weights_MV(:,j), LogRet);
end

% Plot comparison
figure;
plot(risk_MV, DR_MV, 'r-', 'LineWidth', 2)
hold on
plot(targetVol, DR_vals, 'b--', 'LineWidth', 2)
xlabel('Portfolio Volatility')
ylabel('Diversification Ratio')
legend('Mean–Variance Frontier', 'Diversification–Risk Frontier', 'Location', 'best')
title('Diversification Ratio Comparison')
grid on

% Final Plot Comparison Entropy Frontier vs Div-Ratio Frontier
figure;
plot(pf_Risk, EntropyFrontier, 'g-', 'LineWidth', 2)
hold on
plot(targetVol, DR_vals, 'b--', 'LineWidth', 2)
legend('Entropy–Risk Frontier', 'Diversification–Risk Frontier', 'Location', 'best')
xlabel('Portfolio Volatility')
ylabel('Diversification Measure')
title('Comparison of Diversification Frontiers')
grid on


% Interpretation:
% The Entropy frontier measures diversification in terms of weight uniformity.
% The Diversification-Risk frontier measures diversification in terms of
% effective risk reduction.
% Both describe how diversification potential changes with portfolio risk.
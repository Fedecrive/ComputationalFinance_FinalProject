clear all
close all
clc

addpath('./Functions')
addpath('./Data')

%% Read Prices
table_prices = readtable('asset_prices.csv');
mapping_table = readtable('mapping_table.csv');
capitalization_weights = readtable('capitalization_weights.csv');

%% Transform prices from table to timetable
dt = table_prices(:,1).Variables;
values = table_prices(:,2:end).Variables;
nm = table_prices.Properties.VariableNames(2:end);

myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm); 

%% Selection of a subset of Dates
[prices_val, dates] = selectPriceRange(myPrice_dt, '01/01/2018', '31/12/2022');

%% Calculate returns
daysPerYear = 252;

% Daily log-returns
LogRet = log(prices_val(2:end,:) ./ prices_val(1:end-1,:));   % (T-1 x N)

% Daily moments
ExpRet_daily = mean(LogRet);                % 1 x N
CovMatrix_daily    = cov(LogRet);           % N x N
Std_daily    = std(LogRet);                 % 1 x N
numAssets    = size(LogRet, 2);

% Annualized moments
ExpRet_ann = ExpRet_daily * daysPerYear;
CovMatrix_ann    = CovMatrix_daily * daysPerYear;
Std_ann    = Std_daily * sqrt(daysPerYear);

% individual asset volatilities
vol_i  = sqrt(diag(CovMatrix_ann));                           

%% === Exercise 1 – Constrained and Robust Efficient Frontier ===

%% Map assets to macro groups
[tf, loc] = ismember(nm', mapping_table.Asset);
if ~all(tf)
    error('Some assets in asset_prices.csv are not present in mapping_table.csv');
end

groups = mapping_table.MacroGroup(loc);   % 'Cyclical','Neutral','Defensive'

isCyc = strcmp(groups,'Cyclical');
isDef = strcmp(groups,'Defensive');
isNeu = strcmp(groups,'Neutral');

%% Generate N random portfolio
N = 100000;
RetPtfs = zeros(1,N);
VolaPtfs = zeros(1,N);
SharpePtfs = zeros(1,N);

for n = 1:N
    w = rand(1,numAssets);
    w = w./sum(w); % normalize weights

    RetPtfs(n) = w*ExpRet_ann';
    VolaPtfs(n) = sqrt(w * CovMatrix_ann * w');
    SharpePtfs(n) = RetPtfs(n)/VolaPtfs(n);
end

%% Plot: random portfolios
figure;
scatter(VolaPtfs, RetPtfs, [], SharpePtfs, 'filled')
colorbar
title('Random Portfolios: Expected return vs volatility')
xlabel('Volatility (annualized)')
ylabel('Expected return (annualized)')

%% compute efficient frontier con estimateFrontier
p = Portfolio('AssetList', nm);
ret_range = linspace(min(RetPtfs), max(RetPtfs), 100);

nPort = 100;
lb = zeros(numAssets,1);
ub = 0.25*ones(numAssets,1);
p = setBounds(p, lb, ub);

% Maschere logiche (le hai già pronte)
isCyc = isCyc(:)';   
isDef = isDef(:)';

maskDef    = double(isDef);            % somma pesi Defensive
maskCycDef = double(isCyc | isDef);    % Cyclical + Defensive

% Vincoli lineari: A*w <= b
A = [
       maskDef;            % sum_Def <= 0.40
       maskCycDef;         % sum_CycDef <= 0.75
      -maskCycDef          % -sum_CycDef <= -0.45  -> sum_CycDef >= 0.45
    ];
b = [0.40; 0.75; -0.45];

p = setInequality(p, A, b);
p = setBudget(p, 1, 1);
P= setAssetMoments(p, ExpRet_ann, CovMatrix_ann);
W_OPT_estimate= estimateFrontier(P, nPort);

% Momenti dei portafogli sulla frontiera
[portRisk, portExpRet] = estimatePortMoments(P, W_OPT_estimate);
% portRisk    : [1 x nPort] std dev
% portExpRet  : [1 x nPort] expected return

%% Portafoglio a minima varianza
[~, idxMinVar] = min(portRisk);
wMinVar     = W_OPT_estimate(:, idxMinVar);
retMinVar   = portExpRet(idxMinVar);
riskMinVar  = portRisk(idxMinVar);

%% Portafoglio a massimo Sharpe
rf = 0.00;   % metti il tuo risk-free (es. 0.02 per 2%)
excessRet = portExpRet - rf;
Sharpe    = excessRet ./ portRisk;

[~, idxMaxSharpe] = max(Sharpe);
wMaxSharpe     = W_OPT_estimate(:, idxMaxSharpe);
retMaxSharpe   = portExpRet(idxMaxSharpe);
riskMaxSharpe  = portRisk(idxMaxSharpe);
maxSharpeVal   = Sharpe(idxMaxSharpe);

%% reseampling approach fatto a mano seguendo slide
tic
p = Portfolio('AssetList', nm);

nPort = 100;
lb = zeros(numAssets,1);
ub = 0.25*ones(numAssets,1);
p  = setBounds(p, lb, ub);

% Maschere logiche (le hai già pronte)
isCyc = isCyc(:)';   
isDef = isDef(:)';

maskDef    = double(isDef);          % somma pesi Defensive
maskCycDef = double(isCyc | isDef);  % Cyclical + Defensive

% Vincoli lineari: A*w <= b
A = [
       maskDef;            % sum_Def    <= 0.40
       maskCycDef;         % sum_CycDef <= 0.75
      -maskCycDef          % -sum_CycDef <= -0.45  -> sum_CycDef >= 0.45
    ];
b = [0.40; 0.75; -0.45];

p = setInequality(p, A, b);

% FULL INVESTMENT
p = setBudget(p, 1, 1);

% RESAMPLING
nResampling = 100;
T       = size(LogRet,1);   % numero di osservazioni storiche
rf = 0;                     % risk-free per Sharpe

% Preallocazioni in stile "codice 2"
Weights   = zeros(numAssets, nPort, nResampling);
RiskSim   = zeros(nPort, nResampling);
RetSim    = zeros(nPort, nResampling);
SharpeSim = zeros(nPort, nResampling);

rng(42)
for i = 1:nResampling
    
    % 1) Simula una serie storica di T osservazioni
    R_sim = mvnrnd(ExpRet_ann, CovMatrix_ann, T);       % T x numAssets
    
    % 2) Stima momenti dal dataset simulato
    ExpRet_i = mean(R_sim);             % 1 x numAssets
    Cov_i      = cov(R_sim);            % numAssets x numAssets
    
    % 3) Imposta momenti nel Portfolio simulato
    P_sim = setAssetMoments(p, ExpRet_i, Cov_i);
    
    % 4) Calcolo la frontiera con i vincoli
    W_OPT_i = estimateFrontier(P_sim, nPort);           % numAssets x nPort
    Weights(:,:,i) = W_OPT_i;
    
    % 5) Rischio/rendimento dei portafogli simulati
    [pf_risk, pf_ret] = estimatePortMoments(P_sim, W_OPT_i);
    % pf_risk, pf_ret: 1 x nPort
    
    RiskSim(:, i) = pf_risk.';      % nPort x 1
    RetSim(:,  i) = pf_ret.';       % nPort x 1
    
    % 6) Sharpe per ogni punto di frontiera in questa simulazione
    SharpeSim(:, i) = (pf_ret.' - rf) ./ pf_risk.';  % nPort x 1
end

% FRONTIERA / PORTAFOGLI
% Media dei pesi, dei rischi, dei rendimenti e degli Sharpe
meanWeights = mean(Weights, 3);   % numAssets x nPort  (== "W_OPT_robust" di prima)
meanRisk    = mean(RiskSim, 2);   % nPort x 1
meanRet     = mean(RetSim,  2);   % nPort x 1
meanSharpe  = mean(SharpeSim, 2); % nPort x 1

% (Se vuoi tenere il nome vecchio)
W_OPT_robust = meanWeights;

% ---- Portafoglio ROBUSTO a minima varianza (Portfolio C) ----
[~, idxMVP_res] = min(meanRisk);
wMinVar_rob     = meanWeights(:, idxMVP_res);
retMinVar_rob   = meanRet(idxMVP_res);
riskMinVar_rob  = meanRisk(idxMVP_res);

% per coerenza con il secondo codice
Portfolio_C        = wMinVar_rob;
mvp_risk_resampled = riskMinVar_rob;
mvp_ret_resampled  = retMinVar_rob;

% ---- Portafoglio ROBUSTO a massimo Sharpe (Portfolio D) ----
[~, idxMSRP_res] = max(meanSharpe);
wMaxSharpe_rob   = meanWeights(:, idxMSRP_res);
maxSharpe_risk   = meanRisk(idxMSRP_res);
maxSharpe_ret    = meanRet(idxMSRP_res);

Portfolio_D      = wMaxSharpe_rob;
maxSharpeVal_rob = meanSharpe(idxMSRP_res);
toc


%% === Exercise 2 – Black–Litterman Model ===

%% Calculate returns and Covariance Matrix
daysPerYear = 252;
Ret         = tick2ret(prices_val);           % log returns (T-1 x N)
numAssets   = size(Ret, 2);
CovMatrix   = cov(Ret);

CovMatrix_ann = CovMatrix * daysPerYear;  

ExpRet      = mean(Ret);                      % vettore 1 x N (giornalieri)
ExpRet_ann  = ExpRet * daysPerYear;           % annualizzati

%% b) Introduce the views 
Cap         = readtable("Data\capitalization_weights.csv");
assetNames  = string(Cap.Asset);
MacroGroup  = string(Cap.MacroGroup);
wMKT        = Cap.MarketWeight;

v   = 3;                           % numero di views
tau = 1/length(Ret);  

P      = zeros(v, numAssets); 
q      = zeros(v, 1);        
Omega  = zeros(v);

% View 1: Cyclical assets expected to outperform Defensive ones by 2% annualized.  
Cyclical_idx  = find(MacroGroup == "Cyclical");
Defensive_idx = find(MacroGroup == "Defensive");

P(1, Cyclical_idx)  =  1 / length(Cyclical_idx);
P(1, Defensive_idx) = -1 / length(Defensive_idx);
q(1) = 0.02; 

% View 2: Asset 3 is expected to outperform Asset 11 by +1% annualized
P(2,3)  =  1;
P(2,11) = -1;
q(2)    = 0.01; 

% View 3: Asset 7 is expected to outperform the average Defensive group by +0.5% annualized
P(3,7)               =  1; 
P(3, Defensive_idx)  = -1 / length(Defensive_idx);
q(3)                 = 0.005; 

% Compute Omega as tau*P*Cov*P' (diagonal approximation)
for i = 1:v
    Omega(i,i) = tau * P(i,:) * CovMatrix_ann * P(i,:)';
end

%%% Plot views distribution
X_views = mvnrnd(q, Omega, 200);  % 200 random samples from N(q, Omega)
figure;
hold on
for i = 1:v
    histogram(X_views(:,i), 'DisplayName', ['View ' num2str(i)], 'FaceAlpha',0.5);
end
legend
title('Distribution of Views')
hold off

%% a) Compute equilibrium returns 
rf = 0;
Rm_log = Ret * wMKT;  % serie dei ritorni di mercato (giornalieri)

lambda = ((mean(Rm_log)*daysPerYear) - rf) / ...
         (var(Rm_log)*daysPerYear);

mu_mkt = lambda * CovMatrix_ann * wMKT;   % equilibrium returns (annual)
C      = tau * CovMatrix_ann;             % scaled prior covariance

fprintf('\n(a) Equilibrium returns annual:\n');
Tab_Equil = table(assetNames, mu_mkt, ...
    'VariableNames', {'Asset','EquilibriumRet_ann'});
disp(Tab_Equil);

% Plot prior distribution (tutti gli asset insieme)
X_prior = mvnrnd(mu_mkt, C, 200);   % 200 scenari di rendimento di equilibrio
figure;
histogram(X_prior);   % istogramma aggregato
title('Prior Distribution of Returns (Equilibrium)');
xlabel('Annual Return')
ylabel('Frequency')

%% c) Obtain posterior expected returns and compute the efficient frontier under standard
%     constraints (i.e. full investment & no short-selling ) using the in-sample data.
muBL = (C\eye(numAssets) + P'/Omega*P) \ (P'/Omega*q + C\mu_mkt);
covBL = inv(P'/Omega*P + inv(C));

% Compare prior vs BL
TBL = table(assetNames, mu_mkt, muBL, ...
    'VariableNames', ["Asset","PriorReturnAnnual","BLReturnAnnual"]);
disp(TBL) % Plot Distribution

% Black-Litterman PTF
portBL = Portfolio('NumAssets', numAssets, 'Name', 'MV with BL');
portBL = setDefaultConstraints(portBL);
portBL = setAssetMoments(portBL, muBL, CovMatrix_ann+covBL);

% Efficient frontier (computes 100 ptfs along efficient frontier,
%                     and calculates volatility and return for each one)

frontBL = estimateFrontier(portBL, 100);
[vol_BL, ret_BL] = estimatePortMoments(portBL,frontBL);

figure;
plot(vol_BL, ret_BL, 'LineWidth', 2);
xlabel('Volatility'); ylabel('Expected Return');
title('Efficient Frontier with Black–Litterman Views');
grid on;

%% d) MV and MS ptfs 
rf = 0; 

% Minimum Variance ptf 

% Portfolio E: Minimum Variance
w_minVarBL = estimateFrontierLimits(portBL, 'min');
[vol_minVarBL, ret_minVarBL] = estimatePortMoments(portBL, w_minVarBL);

PortfolioE_BL.weights = w_minVarBL;
PortfolioE_BL.vol     = vol_minVarBL;
PortfolioE_BL.ret     = ret_minVarBL;

% Portfolio F: Maximum Sharpe
w_maxSharpeBL = estimateMaxSharpeRatio(portBL);
[vol_maxSharpeBL, ret_maxSharpeBL] = estimatePortMoments(portBL, w_maxSharpeBL);

PortfolioF_BL.weights = w_maxSharpeBL;
PortfolioF_BL.vol     = vol_maxSharpeBL;
PortfolioF_BL.ret     = ret_maxSharpeBL;

fprintf('\nPortfolio E (BL Min Var):  vol = %.4f, ret = %.4f\n', vol_minVarBL, ret_minVarBL);
fprintf('Portfolio F (BL Max Sharpe): vol = %.4f, ret = %.4f\n', vol_maxSharpeBL, ret_maxSharpeBL);

% Check if Sharpe Ratio F> Sharpe Ratio E
Sharpe_E =(ret_minVarBL-rf)/vol_minVarBL;
Sharpe_F =(ret_maxSharpeBL-rf)/vol_maxSharpeBL;

%% Classical ptfs 
port = Portfolio('NumAssets', numAssets, 'Name', 'Mean-Variance');
port = setDefaultConstraints(port);
port = setAssetMoments(port, mean(Ret), CovMatrix);

w_minVar = estimateFrontierLimits(port, 'min');
[vol_minVar, ret_minVar] = estimatePortMoments(port, w_minVar);

w_maxSharpe = estimateMaxSharpeRatio(port);
[vol_maxSharpe, ret_maxSharpe] = estimatePortMoments(port, w_maxSharpe);

% Compare classical vs BL portfolio weights
Tweights = table(assetNames, w_minVarBL, w_minVar, w_maxSharpeBL, w_maxSharpe, ...
    'VariableNames', ["Asset","BL-MinVariance","Classical MinVariance","BL-MaxSharpe","Classical MaxSharpe"]);
disp(Tweights)

% Plot
figure;
subplot(2,2,4)
idxMS = w_maxSharpe > 0.001;
pie(w_maxSharpe(idxMS), assetNames(idxMS))
title('Classical Maximum Sharpe Portfolio')

subplot(2,2,3)
idxBLMS = w_maxSharpeBL > 0.001;
pie(w_maxSharpeBL(idxBLMS), assetNames(idxBLMS))
title('Black-Litterman Maximum Sharpe Portfolio')

subplot(2,2,2)
idxMV = w_minVar> 0.001;
pie(w_minVar(idxMV), assetNames(idxMV))
title('Classical Minimum Variance Portfolio')

subplot(2,2,1)
idxBLMV = w_minVarBL > 0.001;
pie(w_minVarBL(idxBLMV), assetNames(idxBLMV))
title('Black-Litterman Minimum Variance Portfolio')

%% Impact of views on portfolio (Delta weights) – BL MaxSharpe vs Classical MaxSharpe
delta_weights = w_maxSharpeBL - w_maxSharpe;
figure;
bar(delta_weights);
xlabel('Asset Index'); 
ylabel('Change in Weight');
title('Impact of Views on Max Sharpe Portfolio Allocation (BL - Classical)');

%% Analysis of contribution of each view 
contrib = zeros(numAssets, v);
for i = 1:v
    P_i     = P(i,:)';
    Omega_i = Omega(i,i);
    contrib(:,i) = CovMatrix_ann * P_i / (P_i' * CovMatrix_ann * P_i + Omega_i) ...
                   * (q(i) - P_i' * mu_mkt);
end

muBL_contrib = mu_mkt + sum(contrib,2);   % deve essere ~muBL

% Plot contributions (già annuali, NON serve *daysPerYear)
figure;
bar(contrib); 
xlabel('Asset Index'); 
ylabel('Annualized Contribution');
title('Contribution of Each View to BL Expected Returns');
legend("View1","View2","View3");


%% === Exercise 3 – Diversification-Based Optimization ===

%% Common contstraints (G, H, EW)
[prices_val, dates] = selectPriceRange(myPrice_dt, '01/01/2018', '31/12/2022');

% 0 ≤ wi ≤ 0.2
lb = zeros(numAssets,1);
ub = 0.2 * ones(numAssets,1);

% Sum of weights = 1
Aeq = ones(1,numAssets);
beq = 1;

% Group constraints: each macro-group ≥ 15%
% sum_{i in group} w_i ≥ 0.15   ↔  -sum_{i in group} w_i ≤ -0.15
A = [];
b = [];

if any(isCyc)
    rowC = zeros(1,numAssets);
    rowC(isCyc) = -1;
    A = [A; rowC];
    b = [b; -0.15];
end

if any(isNeu)
    rowN = zeros(1,numAssets);
    rowN(isNeu) = -1;
    A = [A; rowN];
    b = [b; -0.15];
end

if any(isDef)
    rowD = zeros(1,numAssets);
    rowD(isDef) = -1;
    A = [A; rowD];
    b = [b; -0.15];
end

% Initial guess
w0 = ones(numAssets,1) / numAssets;

opts = optimoptions('fmincon','Display','off', ...
                    'Algorithm','sqp','MaxIterations',1e8);

%% Portfolio G: Max Diversification Ratio
% Diversification ratio:
% DR(w) = [(w' * vol_i) / sqrt(w' * Sigma * w)]
fun_DR = @(w) - (w' * vol_i) / sqrt(w' * CovMatrix_ann * w);   
[w_G, fval_G] = fmincon(fun_DR, w0, A, b, Aeq, beq, lb, ub, [], opts);

DR = -fval_G;   % maximum DR value

%% Portfolio H: Equal Risk Contribution
% Objective: equal risk contribution
% Minimize the dispersion of risk contributions
fun_RPC = @(w) riskParityObjective(w, CovMatrix_ann);

opts_minmax = optimoptions('fminimax', ...
    'Display', 'off', ...
    'MaxIterations', 1e4, ...
    'ConstraintTolerance', 1e-8, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-10, ...
    'UseParallel', false);

[w_H, fval_H] = fminimax(fun_RPC, w0, A, b, Aeq, beq, lb, ub, [], opts_minmax);

%% Benchmark Equally-Weighted
w_EW = ones(numAssets,1) / numAssets;

%% Metrics: DR, VOL, Sharpe, N_eff
% Diversification Ratio
DR_G = getDiversificationRatio(w_G, LogRet);
DR_H = getDiversificationRatio(w_H, LogRet);
DR_EW = getDiversificationRatio(w_EW, LogRet);

% Annualized volatility
vol_G  = sqrt(w_G' * CovMatrix_ann * w_G);
vol_H  = sqrt(w_H' * CovMatrix_ann * w_H);
vol_EW = sqrt(w_EW' * CovMatrix_ann * w_EW);

% Annualized expected return
ret_G  = ExpRet_ann * w_G;
ret_H  = ExpRet_ann * w_H;
ret_EW = ExpRet_ann * w_EW;

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

%% Plot of Weights
figure;
bar([w_G, w_H, w_EW])
legend({'G: Max DR','H: Risk Parity','EW'},'Location','bestoutside')
xlabel('Asset')
ylabel('Weight')
title('Comparison of G, H and Benchmark Portfolio Weights')
grid on


%% === Exercise 4 – PCA and Conditional Value-at-Risk ===

%% 4(a) PCA on covariance matrix – find k explaining at least 85% variance
% Eigen-decomposition of covariance matrix (annual)
[Vecs,Vals] = eig(CovMatrix_ann);
lambda      = diag(Vals);                         % eigenvalues (unsorted)

% Sort eigenvalues/eigenvectors in descending order
[lambda_sorted, idxEV] = sort(lambda, 'descend');
V_sorted               = Vecs(:, idxEV);

explained_var = lambda_sorted / sum(lambda_sorted);   % fraction each component
cum_explained = cumsum(explained_var);

% Find smallest k with at least 85% cumulative variance
threshold = 0.85;
k = find(cum_explained > threshold, 1);

fprintf('\n=== PCA – Explained variance ===\n');
disp(table((1:numAssets)', lambda_sorted, explained_var, cum_explained, ...
    'VariableNames', {'PC','Eigenvalue','Explained','CumExplained'}));
fprintf('Number of PCs explaining at least 85%% variance: k = %d\n', k);

% Plot explained and cumulative variance (optional)
figure;
bar(explained_var(1:10)*100);
xlabel('Principal Component'); ylabel('Explained Variance (%)');
title('PCA – Explained variance per component');

figure;
plot(1:numAssets, cum_explained*100, '-o');
xlabel('Number of components'); ylabel('Cumulative Explained Variance (%)');
title('PCA – Cumulative explained variance');
grid on;

%% 4(b) Portfolio I – Max Sharpe with PCA covariance & v1-neutrality
% Reconstructed covariance using first k components
V_k = V_sorted(:,1:k);                          % numAssets x k
D_k = diag(lambda_sorted(1:k));                 % k x k
Cov_PCA = V_k * D_k * V_k';                     % numAssets x numAssets

% First eigenvector of full covariance (dominant market factor)
v1 = V_sorted(:,1);                             % numAssets x 1

% Constraints: full investment, 0 <= w_i <= 0.25, w'v1 <= 0.5
x0  = ones(numAssets,1)/numAssets;              % starting point
lb  = zeros(numAssets,1);
ub  = 0.25 * ones(numAssets,1);

% Inequality constraint: w' v1 <= 0.5  ->  (v1') * w <= 0.5
A  = v1';
b  = 0.5;

% Equality constraint: sum_i w_i = 1
Aeq = ones(1,numAssets);
beq = 1;

rf_I = 0;  % risk-free per Sharpe 

% Objective: maximize Sharpe = (mu'w - rf) / sqrt(w'Cov_PCA w)
% → minimize -Sharpe
fun_sharpe_I = @(w) - ((ExpRet_ann * w - rf_I) / sqrt(w' * Cov_PCA * w));

[w_I, fval_I, exitflag_I] = fmincon(fun_sharpe_I, x0, ...
                                    A, b, ...          % <-- nuovo vincolo
                                    Aeq, beq, ...
                                    lb, ub, [], opts);

% Check results
ret_I  = ExpRet_ann * w_I;
vol_I  = sqrt(w_I' * Cov_PCA * w_I);
Sharpe_I = (ret_I - rf_I) / vol_I;

fprintf('\n=== Portfolio I (PCA Max Sharpe, cap su v1, w_i<=0.25) ===\n');
Tab_I = table(nm', w_I, 'VariableNames', {'Asset','Weight'});
disp(Tab_I);
fprintf('Portfolio I: ret = %.4f, vol = %.4f, Sharpe = %.4f\n', ret_I, vol_I, Sharpe_I);

%% 4(b) Portfolio J – Min CVaR_5% with vol cap 15%% and 0<=w<=0.25
alpha = 0.95;            % CVaR at 5% tail
vol_cap = 0.15;          % 15% annualized volatility

% Objective: minimize CVaR_5% (historical, daily returns LogRet)
fun_CVaR = @(w) cvar_obj(w, LogRet, alpha);

% Linear constraints: full investment, 0 <= w_i <= 0.25
x0_J = ones(numAssets,1)/numAssets;
lb_J = zeros(numAssets,1);
ub_J = 0.25 * ones(numAssets,1);
Aeq_J = ones(1,numAssets);
beq_J = 1;

% Nonlinear constraint: vol(w) <= 15% annualized
nonlin_volcap = @(w) deal( w' * CovMatrix_ann * w - vol_cap^2, [] );
% c(w) = vol - 0.15 <= 0   ; ceq = []

optionsJ = optimoptions('fmincon','Display','iter','Algorithm','sqp','ConstraintTolerance',1e-3, ...
    'OptimalityTolerance',1e-6, ...
    'StepTolerance',1e-10);

[w_J, fval_J, exitflag_J] = fmincon(fun_CVaR, x0_J, [],[], Aeq_J, beq_J, lb_J, ub_J, nonlin_volcap, optionsJ);

% Performance of J (in-sample, daily)
pRet_J = LogRet * w_J;
VaR_J  = quantile(pRet_J, 1-alpha);
CVaR_J = -mean(pRet_J(pRet_J <= VaR_J));    % losses positive

ret_J  = ExpRet_ann * w_J;
vol_J  = sqrt(w_J' * CovMatrix_ann * w_J);
Sharpe_J = (ret_J - rf_I) / vol_J;

fprintf('\n=== Portfolio J (Min CVaR_5%%, vol<=15%%, w_i<=0.25) ===\n');
Tab_J = table(nm', w_J, 'VariableNames', {'Asset','Weight'});
disp(Tab_J);
fprintf('Portfolio J: ret = %.4f, vol = %.4f, Sharpe = %.4f, daily CVaR_5 = %.4f\n', ...
        ret_J, vol_J, Sharpe_J, CVaR_J);

g = sqrt(w_J' * CovMatrix_ann * w_J) - vol_cap;

%% Check the MVP under these constraints 
port_min = Portfolio('NumAssets', numAssets);
port_min = setDefaultConstraints(port_min);    % sum(w)=1, w>=0
port_min = setBounds(port_min, zeros(1,numAssets), 0.25*ones(1,numAssets));
port_min = setAssetMoments(port_min, ExpRet_ann, CovMatrix_ann);

w_minVarJ = estimateFrontierLimits(port_min, 'min');
vol_minVarJ = sqrt(w_minVarJ' * CovMatrix_ann * w_minVarJ);

% Given the in-sample covariance structure and the constraints (full investment, 0 ≤ wᵢ ≤ 0.25), the minimum 
% variance portfolio already exhibits an annual volatility above 15%. Therefore, a strict volatility cap at 15% 
% is infeasible in this universe. In practice, we keep the non-linear volatility constraint in the optimization 
% problem, but the solver converges to a solution with volatility around 17–18%, i.e. the minimum level compatible
% with the given constraints

%% 4(c) Tail risk (CVaR), Volatility, Max Drawdown – I vs J
% Daily returns of portfolios (in-sample)
pRet_I = LogRet * w_I;
pRet_J = LogRet * w_J;

% CVaR 5% (daily)
VaR_I  = quantile(pRet_I, 1-alpha);
VaR_J  = quantile(pRet_J, 1-alpha);

CVaR_I = -mean(pRet_I(pRet_I <= VaR_I));
CVaR_J = -mean(pRet_J(pRet_J <= VaR_J));

% Equity curves (usa simple returns per performance metrics)
ret_simple = prices_val(2:end,:) ./ prices_val(1:end-1,:) - 1;   % T-1 x N

equity_I = cumprod(1 + ret_simple * w_I);
equity_J = cumprod(1 + ret_simple * w_J);

equity_I = 100 * equity_I / equity_I(1);
equity_J = 100 * equity_J / equity_J(1);

% Performance metrics (usa la tua getPerformanceMetrics)
[annRet_I, annVol_I, Sharpe_I2, MaxDD_I, Calmar_I] = getPerformanceMetrics(equity_I);
[annRet_J, annVol_J, Sharpe_J2, MaxDD_J, Calmar_J] = getPerformanceMetrics(equity_J);

% Tabella confronto I vs J
perfTable_IJ = table( ...
    [CVaR_I; CVaR_J], ...
    [annVol_I; annVol_J], ...
    [MaxDD_I; MaxDD_J], ...
    [annRet_I; annRet_J], ...
    [Sharpe_I2; Sharpe_J2], ...
    'VariableNames', {'CVaR_5_daily','AnnVol','MaxDD','AnnRet','Sharpe'}, ...
    'RowNames', {'Portfolio I','Portfolio J'});

disp('=== Comparison Portfolio I vs Portfolio J ===');
disp(perfTable_IJ);

% Plot equity curves (optional)
figure;
plot(dates(2:end), equity_I, 'LineWidth',1.5); hold on;
plot(dates(2:end), equity_J, 'LineWidth',1.5);
legend('Portfolio I (PCA Max Sharpe)', 'Portfolio J (Min CVaR)');
xlabel('Date'); ylabel('Equity (base = 100)');
title('Equity curves – Portfolio I vs Portfolio J');
grid on;

%% Local function: CVaR objective
function cvar_val = cvar_obj(w, LogRet, alpha)
    pRet = LogRet * w;             % scenario returns
    VaR  = quantile(pRet, 1-alpha);
    tail = pRet(pRet <= VaR);
    cvar_val = -mean(tail);        % minimize losses in the 5% worst cases
end


%% === Exercise 5 – Personal Strategy ===

%%
F = V_sorted(:, 1:3);
sig_f = diag(lambda_sorted(1:3));
mu = ExpRet_ann';

vec = zeros(numAssets, 1);
for i = 1 : numAssets
    var_spiegata = 0;
    for j = 1 : k 
        var_spiegata = var_spiegata + lambda_sorted(j) * F(i, j)^2;
    end
    vec(i) = CovMatrix_ann(i, i) - var_spiegata;
end
D = diag(vec);

sig = @(w) w' * (F * sig_f * F' + D) * w;

gamma = 1;           
f = @(w) mu'*w - gamma * sig(w);

% vincolo di uguaglianza: sum(w) = 1
Aeq = ones(1, numAssets);
beq = 1;

% bounds: 0 <= w_i <= 0.2
lb = zeros(numAssets, 1);
ub = 0.25 * ones(numAssets, 1);

% punto iniziale: portafoglio equally-weighted
w0 = ones(numAssets, 1) / numAssets;

% fmincon MINIMIZZA: usiamo il negativo di f
obj = @(w) -f(w);

[w_opt_strategy, minus_f_opt, exitflag, output] = fmincon(obj, w0, ...
    [], [], Aeq, beq, lb, ub, [], opts);

% valore MASSIMO della funzione obiettivo
f_opt = -minus_f_opt;

% pesi ottimali

disp('=== Optimal portfolio weights (%) ===');
for i = 1:length(w_opt_strategy)
    fprintf('%-10s : %6.2f %%\n', nm{i}, w_opt_strategy(i)*100);
end

%% Selection of a subset of Dates
[prices_val_validation, dates_validation] = selectPriceRange(myPrice_dt, '01/01/2023', '30/11/2024');

%% Valore portafoglio e pesi nel tempo con ribilanciamento semestrale
V0 = 100;
rebalanceMonths = 6;

[ptf_value, weights_time] = simulateRebalancedPortfolio(w_opt_strategy, prices_val_validation, dates_validation, V0, rebalanceMonths);

%% Plot del valore del portafoglio nel tempo
figure;
plot(dates_validation, ptf_value);
xlabel('Data');
ylabel('Valore portafoglio');
title('Evoluzione del valore del portafoglio con ribilanciamento semestrale');
grid on;

% plot principali pesi nel tempo, per vedere l’effetto del ribilanciamento
figure;
plot(dates_validation, weights_time(:,1), dates_validation, weights_time(:,2), dates_validation, weights_time(:,5), dates_validation, weights_time(:,9), dates_validation, weights_time(:,12), dates_validation, weights_time(:,16));
xlabel('Data');
ylabel('Peso');
title('Evoluzione di alcuni pesi w_i(t)');
legend('Asset 1','Asset 2', 'Asset 5', 'Asset 9', 'Asset 12', 'Asset 16');
grid on;

%% Performance
[annRet_sim, annVol_sim, Sharpe_sim, MaxDD_sim, Calmar_sim] = getPerformanceMetrics(ptf_value);

fprintf('--- Performance del portafoglio ---\n');
fprintf('Rendimento annualizzato:   %.4f\n', annRet_sim);
fprintf('Volatilità annualizzata:   %.4f\n', annVol_sim);
fprintf('Sharpe ratio:              %.4f\n', Sharpe_sim);
fprintf('Max Drawdown:              %.4f\n', MaxDD_sim);
fprintf('Calmar ratio:              %.4f\n', Calmar_sim);

%% Confronto: portafoglio vs singoli asset (prezzi normalizzati a 100)
idxAssets = [1 2 5 9 12 16];

% prezzi normalizzati a 100 alla prima data
norm_prices = V0 * prices_val_validation(:, idxAssets) ./ prices_val_validation(1, idxAssets);

figure;
plot(dates_validation, ptf_value, 'LineWidth', 1.5);  % portafoglio (parte da 100)
hold on;
plot(dates_validation, norm_prices);                      % asset normalizzati
hold off;

xlabel('Data');
ylabel('Valore (base 100)');
title('Portafoglio vs prezzi normalizzati di alcuni asset');
legend('Portafoglio', 'Asset 1','Asset 2','Asset 5','Asset 9','Asset 12','Asset 16', ...
       'Location','best');
grid on;
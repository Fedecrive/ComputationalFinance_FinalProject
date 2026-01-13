%% Calibration and Simulation of Additive Models on WTI Options (June 2020)
tic
clear all;
close all;
clc;

addpath('Functions')
addpath('Dati Train')
addpath('Bootstrap')
profile on

%% Get the Data
[CallPrices, PutPrices, CallStrikes, PutStrikes, CallexpDates, PutexpDates, bidCall, askCall, bidPut, askPut] = buildOptionPrices('2017-12-08.csv');
formatData ='dd/MM/yyyy'; % pay attention to your computer settings
t0 = datetime('08-Dec-2017');
% Unisce tutte le expiry
AllExpDates = [CallexpDates; PutexpDates];

% Converte in datetime (se non lo sono già)
if ~isdatetime(AllExpDates)
    AllExpDates = datetime(AllExpDates);
end

% Ordina e rimuove duplicati
dates = unique(AllExpDates, 'sorted');

%% Point 1 - Get Discounts
[disc, fwd_prices, ExpDates] = bootstrap( ...
    CallPrices, PutPrices, CallStrikes, PutStrikes, CallexpDates, PutexpDates);

%% Step 2 - Additive  parameters calibration
% Initial guesses
k0     = 1;
eta0   = 0;
sigma0  = 0.2;    
alpha=0;
last=length(dates)-2;

tic
[eta, kappa, sigma, RMSE, MAPE] = calibration_with_metrics( ...
    CallPrices, PutPrices, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
    alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates);
dates = [t0; dates];
toc

%% pricing exotic
timegrid = datetime( ...
    {'08/12/2017', '09/04/2018', '08/08/2018', ...
     '10/12/2018', '08/04/2019', '08/08/2019', '09/12/2019'}, ...
    'InputFormat','dd/MM/yyyy');
disc_pricing=interp1(dates, disc, timegrid);
Nsim = 100000;
fwd = interp1(dates(2:end), fwd_prices, timegrid(end));   % forward alla data finale del certificato

[price] = pricing (Nsim, timegrid, fwd,alpha, eta, kappa, sigma, disc_pricing);

%% hedging
PTFvalue=price;
rng(42)

nC = length(CallPrices);
nP = length(PutPrices);
S0 = fwd * disc_pricing(end);

CallDisc = interp1(dates, disc, CallexpDates);
r_C = getZeroRates(CallDisc, [t0; CallexpDates]);
yf_C = yearfrac(t0, CallexpDates, 3);
sigma_call = blsimpv(S0 * ones(nC, 1), CallStrikes, r_C, yf_C, CallPrices, ...
                     'Yield', 0, ...       
                     'Class', 'call');

PutDisc = interp1(dates, disc, PutexpDates);
r_P = getZeroRates(PutDisc, [t0; PutexpDates]);
yf_P = yearfrac(t0, PutexpDates, 3);
sigma_put  = blsimpv(S0 * ones(nP, 1), PutStrikes, r_P, yf_P, PutPrices, ...
                     'Yield', 0, ...
                     'Class', 'put');

sigma_call(isnan(sigma_call)) = 10;
sigma_put(isnan(sigma_put))   = 10;

sigma_call_bp = sigma_call + 0.01;
sigma_call_bn = max(sigma_call - 0.01, 0);

sigma_put_bp = sigma_put + 0.01;
sigma_put_bn = max(sigma_put - 0.01, 0);

% --- CALL prices with bumped vols ---
[C_bp, ~] = blsprice(S0 * ones(nC, 1), CallStrikes, r_C, yf_C, sigma_call_bp, 0);
[C_bn, ~] = blsprice(S0 * ones(nC, 1), CallStrikes, r_C, yf_C, sigma_call_bn, 0);

% --- PUT prices with bumped vols ---
[~, P_bp] = blsprice(S0 * ones(nP, 1), PutStrikes, r_P, yf_P, sigma_put_bp, 0);
[~, P_bn] = blsprice(S0 * ones(nP, 1), PutStrikes, r_P, yf_P, sigma_put_bn, 0);

[EtaUpSensitivity,eta_omega, kappa_omega, sigma_omega] = computeEtaUp(fwd, disc_pricing, Nsim, ...
    timegrid, alpha, eta0, k0, sigma0, price, fwd_prices, C_bn, P_bp, CallStrikes, PutStrikes, t0, disc, CallexpDates, PutexpDates, dates);

[VegaUpSensitivity,eta_Vega, kappa_Vega, sigma_Vega] = computeVegaUp(fwd, disc_pricing, Nsim, ...
    timegrid, alpha, eta0, k0, sigma0, price, fwd_prices, C_bp, P_bp, CallStrikes, PutStrikes, t0, disc, CallexpDates, PutexpDates, dates);

%% eta hedging
nCertificate = -1;
VegaSensitivity = VegaUpSensitivity * nCertificate;
EtaSensitivity = EtaUpSensitivity*nCertificate;

[~, quantityCallEta, idxCallEta, quantityPut, idxPut,anshFlow, ~] = EtaUpHedging( ...
        PTFvalue, EtaSensitivity, ...
        CallexpDates, PutexpDates, timegrid, askCall, bidCall, askPut, bidPut, alpha, disc, dates, ...
        CallStrikes, PutStrikes, fwd_prices, eta, kappa, sigma, eta_omega, kappa_omega, sigma_omega, t0, 0, 0);

%% vega hedging
[PTFvalue, quantityCallVega, idxCallVega, quantityPutVega, idxPutVega, cashFlow, ~] = ...
    Vega_hedging(price, VegaSensitivity, ...
                 alpha, CallexpDates, PutexpDates, timegrid, ...
                 askCall, bidCall, askPut, bidPut, ...
                 disc, dates, CallStrikes, PutStrikes, fwd_prices, ...
                 eta, kappa, sigma, eta_omega, kappa_omega, sigma_omega, t0, ...
                 CallPrices, PutPrices,eta_Vega, kappa_Vega, sigma_Vega, idxCallEta, quantityCallEta, 0, 0, 0);

%% delta hedgind
quantityfwd = deltaHedging( ...
    fwd, disc_pricing, Nsim, timegrid, alpha, eta, kappa, sigma, price, ...
    CallStrikes, CallexpDates, ...
    PutStrikes, PutexpDates, ...
    dates, fwd_prices, disc, t0, ...
    quantityCallEta, idxCallEta, ...
    quantityCallVega, idxCallVega, ...
    quantityPutVega, idxPutVega);

%%
[fwd_bid, fwd_ask] = findCallPutFwd( ...
    CallexpDates, CallStrikes, bidCall, askCall, ...
    PutexpDates, PutStrikes, bidPut, askPut, disc, ExpDates, fwd);

%% Preallocate struct array for options
optionsBook = repmat(struct( ...
    'quantity', 0, ...      % how many contracts you take
    'expiry',   NaT, ...    % expiry as datetime
    'strike',   0 ...      % strike price
), 3, 1);

optionsBook(1).quantity = quantityCallEta;
optionsBook(2).quantity = quantityPutVega;
optionsBook(3).quantity = quantityCallVega;


optionsBook(1).expiry = CallexpDates(idxCallEta);
optionsBook(2).expiry = PutexpDates(idxPutVega);
optionsBook(3).expiry = CallexpDates(idxCallVega);

optionsBook(1).strike = CallStrikes(idxCallEta);
optionsBook(2).strike = PutStrikes(idxPutVega);
optionsBook(3).strike = CallStrikes(idxCallVega);

%%
liquidity = price;

if idxCallVega ~= idxCallEta
    if optionsBook(1).quantity > 0 
        liquidity = liquidity - askCall(idxCallEta) * optionsBook(1).quantity;
    else
        liquidity = liquidity - bidCall(idxCallEta) * optionsBook(1).quantity;
    end
    
    if optionsBook(2).quantity > 0 
        liquidity = liquidity - askPut(idxPutVega) * optionsBook(2).quantity;
    else
        liquidity = liquidity - bidPut(idxPutVega) * optionsBook(2).quantity;
    end
    
    if optionsBook(3).quantity > 0 
        liquidity = liquidity - askCall(idxCallVega) * optionsBook(3).quantity;
    else
        liquidity = liquidity - bidCall(idxCallVega) * optionsBook(3).quantity;
    end
else
    
    if optionsBook(1).quantity + optionsBook(3).quantity > 0 
        liquidity = liquidity - askCall(idxCallEta) * (optionsBook(1).quantity + optionsBook(3).quantity);
    else
        liquidity = liquidity - bidCall(idxCallEta) * (optionsBook(1).quantity + optionsBook(3).quantity);
    end
    
    if optionsBook(2).quantity > 0 
        liquidity = liquidity - askPut(idxPutVega) * optionsBook(2).quantity;
    else
        liquidity = liquidity - bidPut(idxPutVega) * optionsBook(2).quantity;
    end
end

if quantityfwd >= 0 
    liquidity = liquidity - quantityfwd * fwd_ask;
else
    liquidity = liquidity - quantityfwd * fwd_bid;
end

%% 
startDate = datetime('2017-12-08','InputFormat','yyyy-MM-dd'); 
n =125;                                                 
[dates_csv, files_csv] = get_dates_csv(startDate, n);
if length(dates) == 11
    dates = dates(2:end);
end

%% =========================
%  ROBUST VERSION (variable number of expiries per iteration)
%  Key change: store term structures as CELL ARRAYS:
%   - dates_all{i}        : vector of expiry dates (datetime) for iteration i
%   - disc_all{i}         : discount factors aligned with [dates_csv(i); dates_all{i}]  (same length)
%   - fwd_prices_all{i}   : forward prices aligned with dates_all{i} (same length)
%
%  Everything that depends on dates must use {...} not (:,i).
%% =========================

% -------------------------
% Allocation
% -------------------------
eta_all   = eta0   * ones(n, 1);
kappa_all = k0     * ones(n, 1);
sigma_all = sigma0 * ones(n, 1);

eta_all(1)   = eta;
kappa_all(1) = kappa;
sigma_all(1) = sigma;

% -------------------------
% Containers (CELL) because lengths vary with i
% -------------------------
dates_all       = cell(n, 1);     % expiry dates
disc_all        = cell(n, 1);     % discount curve values (aligned with [spotDate; expiries])
fwd_prices_all  = cell(n, 1);     % forward curve values (aligned with expiries)

% Scalars/vectors that stay fixed-size
fwd_all                 = NaN(n, 1);
price_certificate_all   = NaN(n, 1);

PL       = zeros(n, 1);
ptf_val  = zeros(n, 1);

certificate = zeros(n, 1); %#ok<NASGU>  % (left as in your code)

% -------------------------
% Initial data (iteration 1)
% -------------------------
dates_all{1} = dates;

% IMPORTANT:
% disc must be aligned with [dates_csv(1); dates_all{1}] OR with some consistent convention.
% In your original code you used disc_all(:,1)=disc and then interp1([dates_csv; dates_all], disc_all,...)
% That implies disc must have length = 1 + numel(dates_all{1}).
disc_all{1} = disc;

% If you also have initial forward curve vector aligned with dates_all{1}, store it:
% fwd_prices_all{1} = fwd_prices;  % (if available)
% If not available, you can keep it empty and avoid using it at i=1, or set NaN.
fwd_prices_all{1} = NaN(numel(dates_all{1}), 1);

% Portfolio value at t=1
ptf_val(1) = liquidity - price + ...
    optionsBook(1).quantity * CallPrices(idxCallEta) + ...
    optionsBook(2).quantity * PutPrices(idxPutVega) + ...
    optionsBook(2).quantity * CallPrices(idxCallVega) + ...
    quantityfwd * (fwd_ask + fwd_bid) / 2;

%% =========================
% MAIN LOOP
%% =========================
for i = 2:n

    fprintf('\n========== ITERATION %d ==========\n', i);

    % ----------------------------------------------------
    % 1) Roll liquidity to new date using discount interpolation
    % ----------------------------------------------------
    x_prev = [dates_csv(i-1); dates_all{i-1}];
    y_prev = disc_all{i-1};

    % Safety checks (fail fast, with clear message)
    if numel(x_prev) ~= numel(y_prev)
        error("Mismatch at iteration %d: length([dates_csv(i-1); dates_all{i-1}])=%d, length(disc_all{i-1})=%d", ...
              i, numel(x_prev), numel(y_prev));
    end

    d = interp1(x_prev, y_prev, dates_csv(i), 'linear', 'extrap');
    liquidity = liquidity / d;

    % ----------------------------------------------------
    % 2) Load option surface for this date
    % ----------------------------------------------------
    [CallPrices_new, PutPrices_new, CallStrikes_new, PutStrikes_new, CallexpDates_new, ...
        PutexpDates_new, bidCall_new, askCall_new, bidPut_new, askPut_new] = buildOptionPrices(files_csv(i));

    AllExpDates = [CallexpDates_new; PutexpDates_new];
    if ~isdatetime(AllExpDates)
        AllExpDates = datetime(AllExpDates);
    end

    % unique + sorted expiries
    dates_all{i} = unique(AllExpDates, 'sorted');

    % ----------------------------------------------------
    % 3) Bootstrap term structures (discount + forwards)
    % ----------------------------------------------------
    % You currently call:
    %   [disc_all(:, i), fwd_prices_all(:, i), ~] = bootstrap(...)
    % With variable expiries, make bootstrap return:
    %   disc_i  aligned with [dates_csv(i); dates_all{i}]
    %   fwd_i   aligned with dates_all{i}
    %
    % If your bootstrap already returns fixed-length vectors, adapt it or
    % reshape here. This code assumes VARIABLE-length outputs.
    [disc_i, fwd_i, ExpDates_new] = bootstrap( ...
        CallPrices_new, PutPrices_new, CallStrikes_new, PutStrikes_new, CallexpDates_new, PutexpDates_new);

    disc_all{i}       = disc_i;
    fwd_prices_all{i} = fwd_i;

    % Safety checks for bootstrap alignment
    x_i = [dates_csv(i); dates_all{i}];
    if numel(disc_all{i}) ~= numel(x_i)
        error("Bootstrap output mismatch at iteration %d: disc length=%d but expected %d (=1+num expiries).", ...
              i, numel(disc_all{i}), numel(x_i));
    end
    if numel(fwd_prices_all{i}) ~= numel(dates_all{i})
        error("Bootstrap output mismatch at iteration %d: forward length=%d but expected %d (=num expiries).", ...
              i, numel(fwd_prices_all{i}), numel(dates_all{i}));
    end

    % ----------------------------------------------------
    % 4) Calibration (pass expiries as dates_all{i})
    % ----------------------------------------------------
    [eta_all(i), kappa_all(i), sigma_all(i), ~, ~, ~] = calibration( ...
        CallPrices_new, PutPrices_new, CallStrikes_new, PutStrikes_new, fwd_prices_all{i}, ...
        dates_csv(i), disc_all{i}, ...
        alpha, eta0, k0, sigma0, ...
        CallexpDates_new, PutexpDates_new, dates_all{i});

    % ----------------------------------------------------
    % 5) Pricing grid and interpolations
    % ----------------------------------------------------
    timegrid_i = datetime( ...
        {'09/04/2018', '08/08/2018', '10/12/2018', '08/04/2019', '08/08/2019', '09/12/2019'}, ...
        'InputFormat','dd/MM/yyyy');
    timegrid_i = [dates_csv(i) timegrid_i];

    disc_pricing_i = interp1(x_i, disc_all{i}, timegrid_i, 'linear', 'extrap');
    fwd_all(i) = interp1(dates_all{i}, fwd_prices_all{i}, timegrid_i(end), 'linear', 'extrap');

    price_certificate_all(i) = pricing( ...
        Nsim, timegrid_i, fwd_all(i), alpha, eta_all(i), kappa_all(i), sigma_all(i), disc_pricing_i);

    % ----------------------------------------------------
    % 6) Find current hedge instruments
    % ----------------------------------------------------
    [idxopt1, idxopt2, idxopt3] = findOptions( ...
        optionsBook(1).expiry, optionsBook(2).expiry, optionsBook(3).expiry, ...
        optionsBook(1).strike, optionsBook(2).strike, optionsBook(3).strike, ...
        CallStrikes_new, PutStrikes_new, CallexpDates_new, PutexpDates_new);

    % Portfolio value
    ptf_val(i) = liquidity - price_certificate_all(i) + ...
        optionsBook(1).quantity * CallPrices_new(idxopt1) + ...
        optionsBook(2).quantity * PutPrices_new(idxopt2) + ...
        optionsBook(2).quantity * CallPrices_new(idxopt3) + ...
        fwd_all(i) * quantityfwd;

    PL(i) = ptf_val(i) - ptf_val(i-1);

    % ----------------------------------------------------
    % 7) Implied volatilities and bump
    % ----------------------------------------------------
    nC = numel(CallPrices_new);
    nP = numel(PutPrices_new);

    S0 = fwd_all(i) * disc_pricing_i(end);

    % Discount factors at option expiries
    CallDisc = interp1(x_i, disc_all{i}, CallexpDates_new, 'linear', 'extrap');
    r_C      = getZeroRates(CallDisc, [dates_csv(i); CallexpDates_new]);
    yf_C     = yearfrac(dates_csv(i), CallexpDates_new, 3);

    sigma_call = blsimpv( ...
        S0 * ones(nC, 1), CallStrikes_new, r_C, yf_C, CallPrices_new, ...
        'Yield', 0, 'Class', 'call');

    PutDisc = interp1(x_i, disc_all{i}, PutexpDates_new, 'linear', 'extrap');
    r_P     = getZeroRates(PutDisc, [dates_csv(i); PutexpDates_new]);
    yf_P    = yearfrac(dates_csv(i), PutexpDates_new, 3);

    sigma_put = blsimpv( ...
        S0 * ones(nP, 1), PutStrikes_new, r_P, yf_P, PutPrices_new, ...
        'Yield', 0, 'Class', 'put');

    sigma_call(isnan(sigma_call)) = 10;
    sigma_put(isnan(sigma_put))   = 10;

    sigma_call_bp = sigma_call + 0.01;
    sigma_call_bn = max(sigma_call - 0.01, 0);

    sigma_put_bp = sigma_put + 0.01;
    sigma_put_bn = max(sigma_put - 0.01, 0);

    [C_bp, ~] = blsprice(S0 * ones(nC, 1), CallStrikes_new, r_C, yf_C, sigma_call_bp, 0);
    [C_bn, ~] = blsprice(S0 * ones(nC, 1), CallStrikes_new, r_C, yf_C, sigma_call_bn, 0);

    [~, P_bp] = blsprice(S0 * ones(nP, 1), PutStrikes_new, r_P, yf_P, sigma_put_bp, 0);
    [~, P_bn] = blsprice(S0 * ones(nP, 1), PutStrikes_new, r_P, yf_P, sigma_put_bn, 0);

    % ----------------------------------------------------
    % 8) Sensitivities
    % ----------------------------------------------------
    [EtaUpSensitivity, eta_omega, kappa_omega, sigma_omega] = computeEtaUp( ...
        fwd_all(i), disc_pricing_i, Nsim, timegrid_i, alpha, eta0, k0, sigma0, ...
        price_certificate_all(i), fwd_prices_all{i}, C_bn, P_bp, ...
        CallStrikes_new, PutStrikes_new, dates_csv(i), disc_all{i}, ...
        CallexpDates_new, PutexpDates_new, x_i);

    [VegaUpSensitivity, eta_Vega, kappa_Vega, sigma_Vega] = computeVegaUp( ...
        fwd_all(i), disc_pricing_i, Nsim, timegrid_i, alpha, eta0, k0, sigma0, ...
        price_certificate_all(i), fwd_prices_all{i}, C_bp, P_bp, ...
        CallStrikes_new, PutStrikes_new, dates_csv(i), disc_all{i}, ...
        CallexpDates_new, PutexpDates_new, x_i);

    VegaSensitivity = VegaUpSensitivity * nCertificate;
    EtaSensitivity  = EtaUpSensitivity  * nCertificate;

    % ----------------------------------------------------
    % 9) Eta hedging
    % ----------------------------------------------------
    [~, quantityCallEta, idxCallEta, ~, ~, anshFlow, ~] = EtaUpHedging( ... %#ok<NASGU>
        price_certificate_all(i), EtaSensitivity, ...
        CallexpDates_new, PutexpDates_new, timegrid_i, askCall_new, bidCall_new, askPut_new, bidPut_new, ...
        alpha, disc_all{i}, x_i, ...
        CallStrikes_new, PutStrikes_new, fwd_prices_all{i}, ...
        eta_all(i), kappa_all(i), sigma_all(i), ...
        eta_omega, kappa_omega, sigma_omega, ...
        dates_csv(i), 1, idxopt1);

    % ----------------------------------------------------
    % 10) Vega hedging
    % ----------------------------------------------------
    [~, quantityCallVega, idxCallVega, quantityPutVega, idxPutVega, cashFlow, ~] = ... %#ok<NASGU>
        Vega_hedging( ...
            price_certificate_all(i), VegaSensitivity, ...
            alpha, CallexpDates_new, PutexpDates_new, timegrid_i, ...
            askCall_new, bidCall_new, askPut_new, bidPut_new, ...
            disc_all{i}, x_i, CallStrikes_new, PutStrikes_new, fwd_prices_all{i}, ...
            eta_all(i), kappa_all(i), sigma_all(i), ...
            eta_omega, kappa_omega, sigma_omega, ...
            dates_csv(i), CallPrices_new, PutPrices_new, ...
            eta_Vega, kappa_Vega, sigma_Vega, ...
            idxCallEta, quantityCallEta, 1, idxopt3, idxopt2);

    % ----------------------------------------------------
    % 11) Delta hedging (forward quantity)
    % ----------------------------------------------------
    quantityfwd_i = deltaHedging( ...
        fwd_all(i), disc_pricing_i, Nsim, timegrid_i, alpha, eta_all(i), kappa_all(i), sigma_all(i), ...
        price_certificate_all(i), ...
        CallStrikes_new, CallexpDates_new, PutStrikes_new, PutexpDates_new, ...
        x_i, fwd_prices_all{i}, disc_all{i}, dates_csv(i), ...
        quantityCallEta, idxCallEta, ...
        quantityCallVega, idxCallVega, ...
        quantityPutVega, idxPutVega);

    % ----------------------------------------------------
    % 12) Forward bid/ask
    % ----------------------------------------------------
    [fwd_bid, fwd_ask] = findCallPutFwd( ...
        CallexpDates_new, CallStrikes_new, bidCall_new, askCall_new, ...
        PutexpDates_new, PutStrikes_new, bidPut_new, askPut_new, ...
        disc_all{i}, ExpDates_new, fwd_all(i));

    % ----------------------------------------------------
    % 13) Trading to adjust hedge
    % ----------------------------------------------------
    q1 = quantityCallEta  - optionsBook(1).quantity;
    q2 = quantityPutVega  - optionsBook(2).quantity;
    q3 = quantityCallVega - optionsBook(3).quantity;
    q4 = quantityfwd_i    - quantityfwd;

    if idxopt1 ~= idxopt3
        % Call 1
        if q1 > 0
            liquidity = liquidity - askCall_new(idxopt1) * q1;
        else
            liquidity = liquidity - bidCall_new(idxopt1) * q1;
        end

        % Put
        if q2 > 0
            liquidity = liquidity - askPut_new(idxopt2) * q2;
        else
            liquidity = liquidity - bidPut_new(idxopt2) * q2;
        end

        % Call 3
        if q3 > 0
            liquidity = liquidity - askCall_new(idxopt3) * q3;
        else
            liquidity = liquidity - bidCall_new(idxopt3) * q3;
        end
    else
        % Same call instrument for both legs
        if q1 + q3 > 0
            liquidity = liquidity - askCall_new(idxopt1) * (q1 + q3);
        else
            liquidity = liquidity - bidCall_new(idxopt1) * (q1 + q3);
        end

        if q2 > 0
            liquidity = liquidity - askPut_new(idxopt2) * q2;
        else
            liquidity = liquidity - bidPut_new(idxopt2) * q2;
        end
    end

    % Forward trade (NOTE: this uses quantityfwd, like your original code.
    % If you intended to trade only the increment, you would use q4 instead.)
    if quantityfwd >= 0
        liquidity = liquidity - q4* fwd_ask;
    else
        liquidity = liquidity - q4 * fwd_bid;
    end

    % Update positions
    optionsBook(1).quantity = optionsBook(1).quantity + q1;
    optionsBook(2).quantity = optionsBook(2).quantity + q2;
    optionsBook(3).quantity = optionsBook(3).quantity + q3;
    quantityfwd             = quantityfwd_i;

    % Update “current surface”
    CallPrices   = CallPrices_new;   %#ok<NASGU>
    PutPrices    = PutPrices_new;    %#ok<NASGU>
    CallStrikes  = CallStrikes_new;  %#ok<NASGU>
    PutStrikes   = PutStrikes_new;   %#ok<NASGU>
    CallexpDates = CallexpDates_new; %#ok<NASGU>
    PutexpDates  = PutexpDates_new;  %#ok<NASGU>

    % Break condition
    if dates_csv(i) == timegrid(2)
        if fwd_all(i) >= fwd * 1.2
            break
        end
    end
end

profile viewer
profile off

eta_all
kappa_all
sigma_all

%%
figure('Color','w','Position',[200 200 1000 450])

plot(dates_csv, PL, '-', 'LineWidth', 1.6)
hold on
plot(dates_csv, PL, 'o', 'MarkerSize', 3)   % marker piccoli (facoltativi)
hold off

xlabel('Date')
ylabel('PL')
title('PL over time')

grid on
ax = gca;
ax.Box = 'off';
ax.FontSize = 12;
ax.LineWidth = 1;

% Se dates_csv è datetime, migliora l’asse x
if isa(dates_csv,'datetime')
    ax.XAxis.TickLabelFormat = 'dd-MMM-yy';
end

xlim([min(dates_csv) max(dates_csv)])
f = gcf;

% PDF vettoriale (consigliato)
exportgraphics(f, 'PL.pdf', 'ContentType','vector')

%%
figure('Color','w','Position',[200 200 1000 450])

plot(dates_csv, price_certificate_all, ...
     'LineWidth', 1.6)

xlabel('Date')
ylabel('Certificate Price')
title('Certificate Price Over Time')

grid on
ax = gca;
ax.Box = 'off';
ax.FontSize = 12;
ax.LineWidth = 1;

% Improve date axis formatting if datetime
if isa(dates_csv,'datetime')
    ax.XAxis.TickLabelFormat = 'dd-MMM-yy';
end

xlim([min(dates_csv) max(dates_csv)])

% Export in vector format (no quality loss)
exportgraphics(gcf, 'CertificatePrice.pdf', 'ContentType','vector');
% Optional alternative:
% exportgraphics(gcf, 'CertificatePrice.svg', 'ContentType','vector');

%% Combined plot (Price vs normalized PL)
pl_normalized = PL / 100;

figure('Color','w','Position',[200 200 1100 450])

yyaxis left
plot(dates_csv, price_certificate_all, '-', 'LineWidth', 1.6)
ylabel('Certificate Price')
grid on

yyaxis right
plot(dates_csv, pl_normalized, '-', 'LineWidth', 1.6)
hold on
plot(dates_csv, pl_normalized, 'o', 'MarkerSize', 3)  % optional markers
hold off
ylabel('PL (normalized = PL/100)')

xlabel('Date')
title('Certificate Price and Normalized PL Over Time')

ax = gca;
ax.Box = 'off';
ax.FontSize = 12;
ax.LineWidth = 1;

if isa(dates_csv,'datetime')
    ax.XAxis.TickLabelFormat = 'dd-MMM-yy';
end

xlim([min(dates_csv) max(dates_csv)])

% Vector export (zoom-proof)
exportgraphics(gcf, 'Price_and_PL_normalized.pdf', 'ContentType','vector');


%% Certificate price daily change + normalized PL (z-score)

% Ensure column vectors
price = price_certificate_all(:);
pl    = PL(:);

% Daily increment of certificate price: first day = 0
price_change = [0; diff(price)];

% Strong normalization for PL (z-score)
pl_norm = (pl - mean(pl,'omitnan')) ./ std(pl,'omitnan');

% Plot together (two y-axes)
figure('Color','w','Position',[200 200 1100 450])

yyaxis left
plot(dates_csv, price_change, '-', 'LineWidth', 1.6)
ylabel('Certificate Price Change (Δ)')
grid on

yyaxis right
plot(dates_csv, pl_norm, '-', 'LineWidth', 1.6)
hold on
plot(dates_csv, pl_norm, 'o', 'MarkerSize', 3)  % optional markers
hold off
ylabel('PL (normalized, z-score)')

xlabel('Date')
title('Certificate Price Change and Normalized PL Over Time')

ax = gca;
ax.Box = 'off';
ax.FontSize = 12;
ax.LineWidth = 1;

if isa(dates_csv,'datetime')
    ax.XAxis.TickLabelFormat = 'dd-MMM-yy';
end

xlim([min(dates_csv) max(dates_csv)])

% Vector export
exportgraphics(gcf, 'PriceChange_and_PL_normalized.pdf', 'ContentType','vector');

%% Certificate price daily change + PL (same Y scale, single axis)

price = price_certificate_all(:);
pl    = PL(:);

% Daily increment: first day = 0
price_change = [0; diff(price)];

figure('Color','w','Position',[200 200 1100 450])

plot(dates_csv, price_change, '-', 'LineWidth', 1.6)
hold on
plot(dates_csv, pl, '-', 'LineWidth', 1.6)
hold off

xlabel('Date')
ylabel('Value')
title('Certificate Price Change and PL Over Time')
legend({'Certificate Price Change (Δ)', 'PL'}, 'Location','best')

grid on
ax = gca;
ax.Box = 'off';
ax.FontSize = 12;
ax.LineWidth = 1;

if isa(dates_csv,'datetime')
    ax.XAxis.TickLabelFormat = 'dd-MMM-yy';
end

xlim([min(dates_csv) max(dates_csv)])

% Vector export (zoom-proof)
exportgraphics(gcf, 'PriceChange_and_PL_sameScale.pdf', 'ContentType','vector');

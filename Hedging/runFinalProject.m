%% Calibration and Simulation of Additive Models on WTI Options (June 2020)
tic
clear all;
close all;
clc;

addpath('Functions')
addpath('Dati Train')
addpath('Bootstrap')


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
start=1;
alpha=0;
last=length(dates)-2;

tic
[eta, kappa, sigma, MSE, pricesMkt_C, pricesMkt_P] = calibration( ...
    CallPrices, PutPrices, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
    alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates);
dates=[t0; dates];
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

[PTFvalue, quantityCallEta, idxCallEta, quantityPut, idxPut,anshFlow, spesaBidAsk] = EtaUpHedging( ...
        PTFvalue, EtaSensitivity, ...
        CallexpDates, PutexpDates, timegrid, askCall, bidCall, askPut, bidPut, alpha, disc, dates, ...
        CallStrikes, PutStrikes, fwd_prices, eta, kappa, sigma, eta_omega, kappa_omega, sigma_omega, t0, 0, 0);

%% vega hedging
[PTFvalue, quantityCallVega, idxCallVega, quantityPutVega, idxPutVega, cashFlow, spesaBidAsk] = ...
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
    PutexpDates, PutStrikes, bidPut, askPut, disc, ExpDates);

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

%% 
startDate = datetime('2017-12-08','InputFormat','yyyy-MM-dd'); 
n = 4;                                                 
[dates_csv, files_csv] = get_dates_csv(startDate, n);
if length(dates) == 11
    dates = dates(2:end);
end

% Allocation
eta_all = eta0 * ones(n, 1);
kappa_all = k0 * ones(n, 1);
sigma_all = sigma0 * ones(n, 1);

dates_all = NaT(length(dates), n);
disc_all = ones(length(dates) + 1, n);
fwd_prices_all = ones(length(dates), n);
fwd_all = ones(n, 1);
price_certificate_all = ones(n, 1);

dates_all(:, 1) = dates;
disc_all(:, 1) = disc;

PL = zeros(n, 1);
ptf_val = zeros(n, 1);

ptf_val(1) = liquidity - price + optionsBook(1).quantity * CallPrices(idxCallEta) + ...
    + optionsBook(2).quantity * PutPrices(idxPutVega) + optionsBook(2).quantity * CallPrices(idxCallVega);

%%
for i = 2:n
    
    fprintf('\n========== ITERATION %d ==========\n', i);

    d = interp1([dates_csv(i-1); dates_all(:, i-1)], disc_all(:, i-1), dates_csv(i));
    liquidity = liquidity / d;

    [CallPrices_new, PutPrices_new, CallStrikes_new, PutStrikes_new, CallexpDates_new, ...
        PutexpDates_new, bidCall_new, askCall_new, bidPut_new, askPut_new] = buildOptionPrices(files_csv(i));
    
    AllExpDates = [CallexpDates_new; PutexpDates_new];
    % Converte in datetime (se non lo sono già)
    if ~isdatetime(AllExpDates)
        AllExpDates = datetime(AllExpDates);
    end
    
    % Ordina e rimuove duplicati
    dates_all(:, i) = unique(AllExpDates, 'sorted');
    
    [disc_all(:, i), fwd_prices_all(:, i), ~] = bootstrap( ...
    CallPrices_new, PutPrices_new, CallStrikes_new, PutStrikes_new, CallexpDates_new, PutexpDates_new);

    [eta_all(i), kappa_all(i), sigma_all(i), ~, ~, ~] = calibration( ...
        CallPrices_new, PutPrices_new, CallStrikes_new, PutStrikes_new, fwd_prices_all(:, i), dates_csv(i), disc_all(:, i), ...
        alpha, eta0, k0, sigma0, CallexpDates_new, PutexpDates_new, dates_all(:, i));

    timegrid_i = datetime( ...
    {'09/04/2018', '08/08/2018', ...
     '10/12/2018', '08/04/2019', '08/08/2019', '09/12/2019'}, ...
    'InputFormat','dd/MM/yyyy');
    timegrid_i = [dates_csv(i) timegrid_i];

    disc_pricing_i = interp1([dates_csv(i); dates_all(:, i)], disc_all(:, i), timegrid_i);
    fwd_all(i) = interp1(dates_all(:, i), fwd_prices_all(:, i), timegrid_i(end));

    price_certificate_all(i) = pricing (Nsim, timegrid_i, fwd_all(i), alpha, eta_all(i), kappa_all(i), sigma_all(i), disc_pricing_i);


    [idxopt1, idxopt2, idxopt3] = findOptions(optionsBook(1).expiry, optionsBook(2).expiry, optionsBook(3).expiry, ...
        optionsBook(1).strike, optionsBook(2).strike, optionsBook(3).strike, ...
        CallStrikes_new, PutStrikes_new, CallexpDates_new, PutexpDates_new);

    ptf_val(i) = liquidity - price_certificate_all(i) + optionsBook(1).quantity * CallPrices_new(idxopt1) + ...
        + optionsBook(2).quantity * PutPrices_new(idxopt2) + optionsBook(2).quantity * CallPrices_new(idxopt3);

    PL(i) = ptf_val(i) - ptf_val(i-1);

    optionsBook(1).quantity
    optionsBook(2).quantity
    optionsBook(3).quantity

    nC = length(CallPrices_new);
    nP = length(PutPrices_new);
    S0 = fwd_all(i) * disc_pricing_i(end);

    CallDisc = interp1([dates_csv(i); dates_all(:, i)], disc_all(:, i), CallexpDates_new);
    r_C = getZeroRates(CallDisc, [dates_csv(i); CallexpDates_new]);
    yf_C = yearfrac(dates_csv(i), CallexpDates_new, 3);
    sigma_call = blsimpv(S0 * ones(nC, 1), CallStrikes_new, r_C, yf_C, CallPrices_new, ...
                         'Yield', 0, ...       
                         'Class', 'call');
    
    PutDisc = interp1([dates_csv(i); dates_all(:, i)], disc_all(:, i), PutexpDates_new);
    r_P = getZeroRates(PutDisc, [dates_csv(i); PutexpDates_new]);
    yf_P = yearfrac(dates_csv(i), PutexpDates_new, 3);
    sigma_put  = blsimpv(S0 * ones(nP, 1), PutStrikes_new, r_P, yf_P, PutPrices_new, ...
                         'Yield', 0, ...
                         'Class', 'put');
    
    sigma_call(isnan(sigma_call)) = 10;
    sigma_put(isnan(sigma_put))   = 10;
    
    sigma_call_bp = sigma_call + 0.01;
    sigma_call_bn = max(sigma_call - 0.01, 0);
    
    sigma_put_bp = sigma_put + 0.01;
    sigma_put_bn = max(sigma_put - 0.01, 0);
    % --- CALL prices with bumped vols ---
    [C_bp, ~] = blsprice(S0 * ones(nC, 1), CallStrikes_new, r_C, yf_C, sigma_call_bp, 0);
    [C_bn, ~] = blsprice(S0 * ones(nC, 1), CallStrikes_new, r_C, yf_C, sigma_call_bn, 0);
    
    % --- PUT prices with bumped vols ---
    [~, P_bp] = blsprice(S0 * ones(nP, 1), PutStrikes_new, r_P, yf_P, sigma_put_bp, 0);
    [~, P_bn] = blsprice(S0 * ones(nP, 1), PutStrikes_new, r_P, yf_P, sigma_put_bn, 0);



    [EtaUpSensitivity, eta_omega, kappa_omega, sigma_omega] = computeEtaUp(fwd_all(i), disc_pricing_i, Nsim, ...
        timegrid_i, alpha, eta0, k0, sigma0, price_certificate_all(i), fwd_prices_all(:, i), C_bn, P_bp, ...
        CallStrikes_new, PutStrikes_new, dates_csv(i), disc_all(:, i), CallexpDates_new, PutexpDates_new, ...
        [dates_csv(i); dates_all(:, i)]);

    [VegaUpSensitivity, eta_Vega, kappa_Vega, sigma_Vega] = computeVegaUp(fwd_all(i), disc_pricing_i, Nsim, ...
        timegrid_i, alpha, eta0, k0, sigma0, price_certificate_all(i), fwd_prices_all(:, i), C_bp, P_bp, ...
        CallStrikes_new, PutStrikes_new, dates_csv(i), disc_all(:, i), CallexpDates_new, PutexpDates_new, ...
        [dates_csv(i); dates_all(:, i)]);

    VegaSensitivity = VegaUpSensitivity * nCertificate;
    EtaSensitivity = EtaUpSensitivity * nCertificate;

    % eta hedging
    [~, quantityCallEta, idxCallEta, quantityPut, idxPut,anshFlow, ~] = EtaUpHedging( ...
        price_certificate_all(i), EtaSensitivity, ...
        CallexpDates_new, PutexpDates_new, timegrid_i, askCall_new, bidCall_new, askPut_new, bidPut_new, ...
        alpha, disc_all(:, i), [dates_csv(i); dates_all(:, i)], ...
        CallStrikes_new, PutStrikes_new, fwd_prices_all(:, i), eta_all(i), kappa_all(i), sigma_all(i), ...
        eta_omega, kappa_omega, sigma_omega, ...
        dates_csv(i), 1, idxopt1);

    % vega hedging
    [~, quantityCallVega, idxCallVega, quantityPutVega, idxPutVega, cashFlow, ~] = ...
        Vega_hedging(price_certificate_all(i), VegaSensitivity, ...
                     alpha, CallexpDates_new, PutexpDates_new, timegrid_i, ...
                     askCall_new, bidCall_new, askPut_new, bidPut_new, ...
                     disc_all(:, i), [dates_csv(i); dates_all(:, i)], CallStrikes_new, PutStrikes_new, fwd_prices_all(:, i), ...
                     eta_all(i), kappa_all(i), sigma_all(i), eta_omega, kappa_omega, sigma_omega, dates_csv(i), ...
                     CallPrices_new, PutPrices_new, eta_Vega, kappa_Vega, sigma_Vega, idxCallEta, ...
                     quantityCallEta, 1, idxopt3, idxopt2);

    q1 = quantityCallEta - optionsBook(1).quantity;
    q2 = quantityPutVega - optionsBook(2).quantity;
    q3 = quantityCallVega - optionsBook(3).quantity;

    if idxopt1 ~= idxopt3
        if q1 > 0 
            liquidity = liquidity - askCall_new(idxopt1) * q1;
        else
            liquidity = liquidity - bidCall_new(idxopt1) * q1;
        end
    
        if q2 > 0 
            liquidity = liquidity - askPut_new(idxopt2) * q2;
        else
            liquidity = liquidity - bidPut_new(idxopt2) * q2;
        end
    
        if q3 > 0 
            liquidity = liquidity - askCall_new(idxopt3) * q3;
        else
            liquidity = liquidity - bidCall_new(idxopt3) * q3;
        end

    else
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

    optionsBook(1).quantity = optionsBook(1).quantity + q1;
    optionsBook(2).quantity = optionsBook(2).quantity + q2;
    optionsBook(3).quantity = optionsBook(3).quantity + q3;


    CallPrices = CallPrices_new;
    PutPrices = PutPrices_new;
    CallStrikes = CallStrikes_new; 
    PutStrikes = PutStrikes_new;
    CallexpDates = CallexpDates_new;
    PutexpDates = PutexpDates_new;
end

eta_all
kappa_all
sigma_all
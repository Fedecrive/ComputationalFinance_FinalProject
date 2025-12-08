function [Delta, Vega, Omega] = ComputeSensitivities(CallPrices, PutPrices, CallStrikes, PutStrikes, fwd_prices, fwd, t0, disc, dates, ...
    disc_pricing, alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, timegrid, price, eta, kappa, sigma)

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

%% Vega
% [eta_bp, kappa_bp, sigma_bp, ~] = calibration( ...
%     C_bp, P_bp, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
%     alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates(2:end))
% 
% [eta_bn, kappa_bn, sigma_bn, ~] = calibration( ...
%     C_bn, P_bn, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
%     alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates(2:end))

eta_bp = 4.5107;
kappa_bp = 2.4154;
sigma_bp = 0.1370;

eta_bn = 5.4240;
kappa_bn = 2.4457;
sigma_bn = 0.1170;

Nsim = 1e4;
alpha = 0.5;
[price_bp] = pricing (Nsim, timegrid, fwd, alpha, eta_bp, kappa_bp, sigma_bp, disc_pricing);
[price_bn] = pricing (Nsim, timegrid, fwd, alpha, eta_bn, kappa_bn, sigma_bn, disc_pricing);

Vega = (price_bp - price_bn) / 0.02;

%% Omega
% [eta_bp, kappa_bp, sigma_bp, ~] = calibration( ...
%      C_bn, P_bp, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
%      alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates(2:end))

eta_omega = 23.1693;
kappa_omega = 2.8877;
sigma_omega = 0.0724;

[price_omega] = pricing (Nsim, timegrid, fwd, alpha, eta_omega, kappa_omega, sigma_omega, disc_pricing);

Omega = price_omega - price;

%% Delta
fwd_p = (fwd / disc_pricing(end) + 0.01) * disc_pricing(end);
fwd_n = (fwd / disc_pricing(end) - 0.01) * disc_pricing(end);
[price_deltap] = pricing (Nsim, timegrid, fwd_p, alpha, eta, kappa, sigma, disc_pricing);
[price_deltan] = pricing (Nsim, timegrid, fwd_n, alpha, eta, kappa, sigma, disc_pricing);

Delta = (price_deltap - price_deltan) / 0.02;

end
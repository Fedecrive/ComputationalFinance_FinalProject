function ValueChange=computeSigmaUp(fwd, disc_pricing, Nsim, timegrid, alpha, eta0, k0, sigma0, Price, fwd_prices)
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


sigma_put_bp = sigma_put + 0.01;


% --- CALL prices with bumped vols ---
[C_bp, ~] = blsprice(S0 * ones(nC, 1), CallStrikes, r_C, yf_C, sigma_call_bp, 0);

% --- PUT prices with bumped vols ---
[~, P_bp] = blsprice(S0 * ones(nP, 1), PutStrikes, r_P, yf_P, sigma_put_bp, 0);


% Vega
[eta_bp, kappa_bp, sigma_bp, ~] = calibration( ...
    C_bp, P_bp, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
    alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates(2:end));


[price_bp] = pricing (Nsim, timegrid, fwd, alpha, eta_bp, kappa_bp, sigma_bp, disc_pricing);


ValueChange=price_bp-Price;

function [VegaUpSensitivity,eta_Vega, kappa_Vega, sigma_Vega]=computeVegaUp(fwd, disc_pricing, Nsim, timegrid, alpha, eta0, k0, sigma0, price, fwd_prices, C_bp, P_bp, CallStrikes, PutStrikes, t0, disc, CallexpDates, PutexpDates, dates)
[eta_Vega, kappa_Vega, sigma_Vega, ~] = calibration( ...
     C_bp, P_bp, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
     alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates(2:end));
% eta_omega = 23.1693;
% kappa_omega = 2.8877;
% sigma_omega = 0.0724;

[price_Vega] = pricing (Nsim, timegrid, fwd, alpha, eta_Vega, kappa_Vega, sigma_Vega, disc_pricing);

VegaUpSensitivity = price_Vega - price;
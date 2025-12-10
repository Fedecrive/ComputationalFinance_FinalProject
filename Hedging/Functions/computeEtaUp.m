function [ValueChange,eta_omega, kappa_omega, sigma_omega] = computeEtaUp(fwd, disc_pricing, Nsim, timegrid, alpha, eta0, k0, sigma0, Price, fwd_prices, C_bn, P_bp, CallStrikes, PutStrikes, t0, disc, CallexpDates, PutexpDates, dates)
[eta_omega, kappa_omega, sigma_omega, ~] = calibration( ...
     C_bn, P_bp, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
     alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates(2:end));
% eta_omega = 23.1693;
% kappa_omega = 2.8877;
% sigma_omega = 0.0724;

[price_omega] = pricing (Nsim, timegrid, fwd, alpha, eta_omega, kappa_omega, sigma_omega, disc_pricing);

ValueChange = price_omega - Price;
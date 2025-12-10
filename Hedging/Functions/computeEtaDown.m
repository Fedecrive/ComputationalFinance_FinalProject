function ValueChange=computeEtaDown(fwd, disc_pricing, Nsim, timegrid, alpha, eta0, k0, sigma0, Price, fwd_prices, C_bp, P_bn)
[eta_omega, kappa_omega, sigma_omega, ~] = calibration( ...
     C_bp, P_bn, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
     alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates(2:end));

[price_omega] = pricing (Nsim, timegrid, fwd, alpha, eta_omega, kappa_omega, sigma_omega, disc_pricing);

ValueChange = price_omega - Price;
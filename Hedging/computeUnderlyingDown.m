function ValueChange=computeUnderlyingDown(fwd, disc_pricing, Nsim, timegrid, alpha, eta, kappa, sigma, Price)
fwd_d = (fwd / disc_pricing(end) - 0.01) * disc_pricing(end);

[price_deltad] = pricing (Nsim, timegrid, fwd_d, alpha, eta, kappa, sigma, disc_pricing);


ValueChange = (price_deltad - Price);
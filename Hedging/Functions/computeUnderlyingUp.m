function ValueChange = computeUnderlyingUp(fwd, disc_pricing, Nsim, timegrid, alpha, eta, kappa, sigma, Price)

fwd_p = (fwd / disc_pricing(end) + 0.01) * disc_pricing(end);

[price_deltap] = pricing (Nsim, timegrid, fwd_p, alpha, eta, kappa, sigma, disc_pricing);

ValueChange = (price_deltap - Price);

end
function price_margrabe = priceMargrabe(F1_0,F2_0,sigma1,sigma2,ttm,discPricing)
sigM = norm(sigma1 - sigma2);
tau  = ttm;

d1 = (log(F1_0/F2_0) + 0.5*sigM^2*tau) / (sigM*sqrt(tau));
d2 = d1 - sigM*sqrt(tau);

price_margrabe = discPricing * (F1_0*normcdf(d1) + F2_0*normcdf(-d2));
end
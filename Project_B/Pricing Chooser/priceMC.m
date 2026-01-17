function [price_MC,F1_end,F2_end] = priceMC(F1_0,F2_0,sigma1,sigma2,ttm,discPricing,nSim,nComp)
Z = randn(nSim, nComp);
F1_end = F1_0*exp(-0.5*norm(sigma1)^2*ttm+sqrt(ttm)*sigma1*Z');
F2_end = F2_0*exp(-0.5*norm(sigma2)^2*ttm+sqrt(ttm)*sigma2*Z');
payoff = max(F2_end,F1_end);
price_MC = discPricing * mean(payoff);
end
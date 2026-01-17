function price_closedform = priceClosedForm(F1_0,F2_0,sigma1,sigma2,ttm,discPricing)
prob1 = normcdf((log(F1_0/F2_0)+0.5*(norm(sigma2)^2-norm(sigma1)^2)*ttm)/(norm(sigma1-sigma2)*sqrt(ttm)));
prob2 = normcdf((log(F2_0/F1_0)+0.5*(norm(sigma1)^2-norm(sigma2)^2)*ttm)/(norm(sigma2-sigma1)*sqrt(ttm)));

m1  = log(F1_0) - 0.5*norm(sigma1)^2 * ttm;
m2  = log(F2_0) - 0.5*norm(sigma2)^2 * ttm;

s1  = norm(sigma1) * sqrt(ttm);
s2  = norm(sigma2) * sqrt(ttm);

rho = (sigma1*sigma2')/(norm(sigma1)*norm(sigma2));  % correlazione dei log

E_F1_cond = exp(m1+0.5*s1^2)*normcdf((m1-m2+s1^2-rho*s1*s2)/sqrt(s1^2+s2^2-2*rho*s1*s2))/normcdf((m1-m2)/sqrt(s1^2+s2^2-2*rho*s1*s2));
E_F2_cond = exp(m2+0.5*s2^2)*normcdf((m2-m1+s2^2-rho*s1*s2)/sqrt(s1^2+s2^2-2*rho*s1*s2))/normcdf((m2-m1)/sqrt(s1^2+s2^2-2*rho*s1*s2));

price_closedform = discPricing*(E_F1_cond*prob1+E_F2_cond*prob2);
end
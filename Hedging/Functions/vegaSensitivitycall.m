function ValueChangeCall= vegaSensitivitycall(alpha, CallexpDates, CallPrices, disc, dates, CallStrikes, fwd_prices, eta, kappa, sigma, eta_vega, kappa_vega, sigma_vega, t0)


if alpha == 0
    L_fun = @(u, t, eta, kappa, sigma) ...
        exp(-t./kappa .* log(1 + kappa .* u .* sigma.^2));
else
    L_fun = @(u, t, eta, kappa, sigma) ...
        exp(t./kappa .* (1 - sqrt(1 + 2 .* kappa .* u .* sigma.^2)));
end
a=0.5;
phi_fun = @(u, t, eta, kappa, sigma) ...
    exp(-1i .* u .* log(L_fun(eta, t, eta, kappa, sigma))) .* ...
    L_fun(0.5 .* (u.^2 + 1i .* (1 + 2 .* eta) .* u), t, eta, kappa, sigma);

% Lewis formula vettoriale in x:
% x può essere scalare o vettore
M = 15;    % Grid size parameter
dz = 0.01; % Grid spacing
pricesModelEtaUp = [];
pricesModel   = [];
for i = 1:length(CallexpDates)
    idx = find(dates(2:end) == CallexpDates(i));


    mon = log(fwd_prices(idx)./CallStrikes(i));
    %if ~isnan(CallPrices(i)) && ~(mon < 0.005 && mon > -0.005) && year(CallexpDates(i))~=2017 && year(CallexpDates(i))~=2018
        disc_i = disc(idx+1);                  % Discount factor for this maturity
        ttm_i  = yearfrac(t0, dates(idx), 3);  % Maturity in anni
        
        pricesModelEtaUp =[ pricesModelEtaUp, PriceCallOption(CallStrikes(i),fwd_prices(idx),disc_i,@(u)phi_fun(u,ttm_i,eta_vega, kappa_vega, sigma_vega),M,dz,a)];
        pricesModel =[ pricesModel, PriceCallOption(CallStrikes(i),fwd_prices(idx),disc_i,@(u)phi_fun(u,ttm_i,eta,kappa,sigma),M,dz,a)];
    %end
end

ValueChangeCall=pricesModelEtaUp-pricesModel;
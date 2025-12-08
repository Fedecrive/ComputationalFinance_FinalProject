function [eta, kappa, sigma, MSE, pricesMkt_C, pricesMkt_P] = calibration( ...
    CallPrices, PutPrices, CallStrikes, PutStrikes, fwdPrices, t0, disc, ...
    alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates)

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





pricesModel = {}; % Cell array to store model price functions
pricesMkt   = []; % Vector to store normalized market prices


% Loop through maturities (skipping the first one)
for i = 1:length(CallexpDates)
    
    idx = find(dates == CallexpDates(i));


    mon = log(fwdPrices(idx)./CallStrikes(i));

    % Select only valid call and put options that are out-of-the-money
    if ~isnan(CallPrices(i)) && mon < 0 && mon > -0.5 && year(CallexpDates(i))~=2017 && year(CallexpDates(i))~=2018
    
    
    
    

        disc_i = disc(idx+1);                  % Discount factor for this maturity
        ttm_i  = yearfrac(t0, dates(idx), 3);  % Maturity in anni
    
        % Call: vettore lunghezza sum(maskCall)
        pricesModel{end+1} = @(eta, kappa, sigma) ...
            PriceCallOption(CallStrikes(i),fwdPrices(idx),disc_i,@(u)phi_fun(u,ttm_i,eta,kappa,sigma),M,dz,a);
    
    
        % Market prices (stessa logica: solo opzioni selezionate)
        pricesMkt = [pricesMkt, CallPrices(i)];
        
    end
end

pricesMkt_C = pricesMkt;

for i = 1:length(PutexpDates)
    

   idx = find(dates==PutexpDates(i));


    mon = log(fwdPrices(idx)./PutStrikes(i));

    % Select only valid call and put options that are out-of-the-money
    if ~isnan(PutPrices(i)) && mon > 0 && mon < 0.5 && year(CallexpDates(i))~=2017 && year(CallexpDates(i))~=2018
    
    
    
    

        disc_i = disc(idx+1);                  % Discount factor for this maturity
        ttm_i  = yearfrac(t0, dates(idx), 3);  % Maturity in anni
    
        % Call: vettore lunghezza sum(maskCall)
        pricesModel{end+1} = @(eta, kappa, sigma) ...
            PriceCallOption(PutStrikes(i),fwdPrices(idx),disc_i,@(u)phi_fun(u,ttm_i,eta,kappa,sigma),M,dz,a)-disc_i*(fwdPrices(idx)-PutStrikes(i));
    
    
        % Market prices (stessa logica: solo opzioni selezionate)
        pricesMkt = [pricesMkt, PutPrices(i)];
        
    end
end

pricesMkt_P = pricesMkt((length(CallexpDates) + 1):end);

%% ===== Dopo il ciclo sulle maturities =====

% pricesModel contiene handle del tipo @(eta,kappa,sigma) ...
% Vettorizziamo concatenando tutti i prezzi modello
pricesModelVec = @(p) cell2mat( ...
    cellfun(@(f) f(p(1), p(2), p(3)), pricesModel, 'UniformOutput', false) );

% Funzione obiettivo: somma dei quadrati degli errori prezzo
objective = @(p) sum( (real(pricesModelVec(p)) - pricesMkt).^2 );

% Guess iniziale sui tre parametri (devi avere sigma0 definito)
x0 = [eta0; k0; sigma0];

% Vincoli: kappa > 0, sigma > 0
lb = [-Inf; 1e-6; 1e-6];   % eta libero, kappa>0, sigma>0
ub = [ Inf; 500; 500];       % metti tu un bound sensato per sigma

% Opzioni ottimizzazione
opts = optimoptions('fmincon', ...
    'Display', 'iter', ...              % mostra progresso
     ...     % funzione che stampa parametri
    'MaxIterations', 200, ...
    'OptimalityTolerance', 1e-8, ...
    'StepTolerance', 1e-12);


% Calibrazione
[par, ~] = fmincon(objective, x0, [], [], [], [], lb, ub, [], opts);

% Parametri calibrati
eta   = par(1);
kappa = par(2);
sigma = par(3);

%==== Compute global MSE on the same options used for calibration ====

% Prezzi modello con i parametri calibrati, usando gli stessi handle
modelPricesOpt = pricesModelVec(par);   % vettore 1 x N
modelPricesOpt = real(modelPricesOpt);  % togli eventuale parte immaginaria numerica

% Vettore prezzi di mercato usato in calibrazione
marketPrices   = pricesMkt;             % 1 x N

% Controllo di sicurezza
if numel(modelPricesOpt) ~= numel(marketPrices)
    error('Dimension mismatch: modelPricesOpt and marketPrices have different lengths.');
end

residuals = modelPricesOpt(:) - marketPrices(:);
MSE       = mean(abs(residuals));

end

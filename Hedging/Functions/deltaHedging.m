function underlying_quantity = deltaHedging( ...
    fwd, disc_pricing, Nsim, timegrid, alpha, eta, kappa, sigma, Price, ...
    CallStrikes, CallexpDates, ...
    PutStrikes, PutexpDates, ...
    dates, fwd_prices, disc, t0, ...
    quantityCallEta, idxCallEta, ...
    quantityCallVega, idxCallVega, ...
    quantityPutVega, idxPutVega)
% DELTAHEDGING
% Calcola la variazione di valore del portafoglio (certificato + opzioni di hedge)
% per un bump del sottostante di +0.01 (coerente con le funzioni computeUnderlyingUp*).
%
% INPUT:
%   fwd, disc_pricing, Nsim, timegrid, alpha, eta, kappa, sigma, Price
%       -> per computeUnderlyingUp (certificato)
%
%   CallStrikes, CallexpDates, dates, fwd_prices, disc, alpha, eta, kappa, sigma, t0
%       -> per computeUnderlyingUpCall (tutte le call)
%
%   PutStrikes, PutexpDates, dates, fwd_prices, disc, alpha, eta, kappa, sigma, t0
%       -> per computeUnderlyingUpPut (tutte le put)
%
%   quantityCallEta,  idxCallEta  : quantità e indice della call usata per hedge in eta
%   quantityCallVega, idxCallVega : quantità e indice della call usata per hedge in vega
%   quantityPutVega,  idxPutVega  : quantità e indice della put usata per hedge in vega
%
% OUTPUT:
%   totalValueChange : variazione complessiva di valore del portafoglio
%                      per un bump +0.01 del sottostante.

    % 1) Bump del certificato
    certValueChange = computeUnderlyingUp( ...
        fwd, disc_pricing, Nsim, timegrid, alpha, eta, kappa, sigma, Price);

    % 2) Bump di TUTTE le call
    valueChangeCallAll = computeUnderlyingUpCall( ...
        CallStrikes, CallexpDates, dates, ...
        fwd_prices, disc, ...
        alpha, eta, kappa, sigma, ...
        t0);
    valueChangeCallAll = valueChangeCallAll(:);  % forzo vettore colonna

    % 3) Bump di TUTTE le put
    valueChangePutAll = computeUnderlyingUpPut( ...
        PutStrikes, PutexpDates, dates, ...
        fwd_prices, disc, ...
        alpha, eta, kappa, sigma, ...
        t0);
    valueChangePutAll = valueChangePutAll(:);    % forzo vettore colonna

    % 4) Estraggo i contributi delle singole opzioni usate per l'hedge
    deltaCallEta  = valueChangeCallAll(idxCallEta);
    deltaCallVega = valueChangeCallAll(idxCallVega);
    deltaPutVega  = valueChangePutAll(idxPutVega);

    % 5) Somma complessiva: certificato + opzioni di hedge
    totalValueChange = - certValueChange ...
        + quantityCallEta  * deltaCallEta ...
        + quantityCallVega * deltaCallVega ...
        + quantityPutVega  * deltaPutVega;

    underlying_quantity = - totalValueChange / 0.01;
end


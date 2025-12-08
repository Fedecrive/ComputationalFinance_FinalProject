
function [PTFvalue, quantityCall, idxCall, quantityPut, idxPut, cashFlow, spesaBidAsk] = EtaUpHedging( ...
        PTFvalue, EtaUpSensitivity, ...
        CallexpDates, PutexpDates, timegrid, askCall, bidCall, askPut, bidPut, alpha, disc, dates, CallStrikes, PutStrikes, fwd_prices, eta, kappa, sigma, eta_omega, kappa_omega, sigma_omega, t0)

    % inizializzo tutto
    quantityCall = 0;
    quantityPut  = 0;
    idxCall      = 0;
    idxPut       = 0;
    cashFlow     = 0;
    
    
    if EtaUpSensitivity > 0
        % Hedging con CALL: usa la funzione ottimale
        ValueChangeCall= etaSensitivitycall(alpha, CallexpDates, bidCall, disc, dates, CallStrikes, fwd_prices, eta, kappa, sigma, eta_omega, kappa_omega, sigma_omega, t0);
        [cashFlow, idxCall, quantityCall, spesaBidAsk] = selectOptimalCall( ...
            EtaUpSensitivity, CallexpDates, timegrid, askCall, bidCall, ValueChangeCall);
        
    elseif EtaUpSensitivity < 0
        ValueChangePut= etaSensitivityput(alpha, PutexpDates, bidPut, disc, dates, PutStrikes, fwd_prices, eta, kappa, sigma, eta_omega, kappa_omega, sigma_omega, t0);
        [cashFlow, idxPut, quantityPut] = selectOptimalPut( ...
        EtaUpSensitivity, PutexpDates, timegrid, askPut, bidPut, ValueChangePut);
    end

    % aggiorno il valore del portafoglio dopo il cash-flow dell'hedge
    PTFvalue = PTFvalue - cashFlow;
end

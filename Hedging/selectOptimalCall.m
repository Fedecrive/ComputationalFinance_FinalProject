function [cashFlow, idxCall, quantityCall, spesaBidAsk]=selectOptimalCall(EtaUpSensitivity, CallexpDates, timegrid, askCall, bidCall, ValueChangeCall)

    % Forzo tutto a vettori colonna per sicurezza
    CallexpDates    = CallexpDates(:);
    askCall         = askCall(:);
    bidCall         = bidCall(:);
    ValueChangeCall = ValueChangeCall(:);

    % 1) vincolo di scadenza: CallexpDates(idx) >= timegrid(end)
    maskMaturity = CallexpDates >= timegrid(end);

    % 2) scarto le call con ValueChangeCall = 0 (non hedgiano nulla)
    maskNonZero = (ValueChangeCall ~= 0);

    % 3) sensitività opposte (per hedgiare):
    %    EtaUpSensitivity * ValueChangeCall < 0
    maskSign = (EtaUpSensitivity .* ValueChangeCall) < 0;

    % 4) spread positivo
    spread = askCall - bidCall;
    maskSpread = spread > 0;

    % indici candidati
    maskValid = maskMaturity & maskNonZero & maskSign & maskSpread;

    fprintf('selectOptimalCall debug:\n');
    fprintf('  nTot        = %d\n', numel(CallexpDates));
    fprintf('  maskMaturity true = %d\n', sum(maskMaturity));
    fprintf('  maskNonZero  true = %d\n', sum(maskNonZero));
    fprintf('  maskSign     true = %d\n', sum(maskSign));
    fprintf('  maskSpread   true = %d\n', sum(maskSpread));
    fprintf('  maskValid    true = %d\n', sum(maskValid));

    if ~any(maskValid)
        error('selectOptimalCall: nessuna call soddisfa i vincoli.');
    end

    % 4) quantità richiesta per NEUTRALIZZARE la sensibilità:
    %    EtaUpSensitivity + q * ValueChangeCall = 0
    qtyAll = - EtaUpSensitivity ./ ValueChangeCall;   % <-- MENO QUI

    % 5) costo di spread: |qty| * (ask - bid)
    costAll = abs(qtyAll) .* spread;

    % rendo inf i non validi così non vengono scelti
    costAll(~maskValid) = inf;

    % 6) scelgo l'indice con costo minimo
    [~, idxCall] = min(costAll);

    % quantità della call scelta (ora positiva, dato il vincolo di segno)
    quantityCall = qtyAll(idxCall);

    % 7) cash flow iniziale (solo casi long: quantityCall > 0)
    cashFlow =  quantityCall * askCall(idxCall);
    spesaBidAsk=costAll(idxCall);
end

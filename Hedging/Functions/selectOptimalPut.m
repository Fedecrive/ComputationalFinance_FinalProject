function [cashFlow, idxPut, quantityPut] = selectOptimalPut( ...
        EtaUpSensitivity, PutexpDates, timegrid, askPut, bidPut, ValueChangePut)

    % Metto tutto a vettori colonna
    PutexpDates    = PutexpDates(:);
    askPut         = askPut(:);
    bidPut         = bidPut(:);
    ValueChangePut = ValueChangePut(:);

    % 1) Vincolo di scadenza: PutexpDates(idx) >= timegrid(end)
    maskMaturity = PutexpDates >= timegrid(end);

    % 2) Scarto le put con ValueChangePut = 0 (non hedgiano nulla)
    maskNonZero = (ValueChangePut ~= 0);

    % 3) Stessa direzione di hedge:
    %    voglio strumenti che abbiano lo stesso segno di EtaUpSensitivity
    %    (così la quantità risultante è > 0 se i segni coincidono)
    maskSign = sign(EtaUpSensitivity) .* sign(ValueChangePut) < 0;

    % 4) Spread positivo
    spread = askPut - bidPut;
    maskSpread = spread > 0;

    % Indici validi
    maskValid = maskMaturity & maskNonZero & maskSign & maskSpread;
    if ~any(maskValid)
        error('selectOptimalPut: nessuna put soddisfa i vincoli.');
    end

    % 5) Quantità richiesta per ciascuna put:
    %    quantityPut_i * ValueChangePut_i = EtaUpSensitivity
    qtyAll = EtaUpSensitivity ./ ValueChangePut;

    % 6) Costo di spread: |qty| * (ask - bid)
    costAll = abs(qtyAll) .* spread;

    % Escludo le non valide mettendo costo infinito
    costAll(~maskValid) = inf;

    % 7) Scelgo l'indice con costo minimo (indice nel vettore originale)
    [~, idxPut] = min(costAll);

    % Quantità della put scelta
    quantityPut = qtyAll(idxPut);

    % 8) Cash flow iniziale:
    %    se quantityPut > 0 → compri al ask → esborso (cashFlow < 0)
    cashFlow = - quantityPut * askPut(idxPut);
end

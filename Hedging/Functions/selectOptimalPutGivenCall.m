function [cashFlow,idxCall, idxPut, quantityPut, quantityCall, spesaBidAsk] = ...
    selectOptimalPutGivenCall( ...
        VegaUpSensitivity, CallexpDates, PutexpDates, timegrid, ...
        askCall, bidCall, askPut, bidPut, ...
        ValueChangeCallVega, ValueChangePutVega, ...
        ValueChangeCallEta,  ValueChangePutEta, eps, flag, idxC, idxP) %#ok<INUSD>
% Nuova versione:
% - Non imponiamo più a priori quale call usare.
% - Filtriamo sia call che put per scadenza >= timegrid(end).
% - Ciclciamo su tutte le coppie (call, put) ammissibili.
% - Per ogni coppia (iC, iP) risolviamo:
%       [ C_vega  P_vega ] [qC] = [ -VegaUpSensitivity ]
%       [ C_eta   P_eta  ] [qP]   [ 0                 ]
%   e accettiamo solo le soluzioni che soddisfano
%       abs(qC * C_eta + qP * P_eta) < eps
%   minimizzando la spesa in bid-ask:
%       |qP| * (askPut - bidPut) + |qC| * (askCall - bidCall).

    % Forzo a vettori colonna
    CallexpDates        = CallexpDates(:);
    PutexpDates         = PutexpDates(:);
    askCall             = askCall(:);
    bidCall             = bidCall(:);
    askPut              = askPut(:);
    bidPut              = bidPut(:);
    ValueChangeCallVega = ValueChangeCallVega(:);
    ValueChangePutVega  = ValueChangePutVega(:);
    ValueChangeCallEta  = ValueChangeCallEta(:);
    ValueChangePutEta   = ValueChangePutEta(:);

    % Inizializzo output
    idxPut        = 0;
    quantityPut   = 0;
    quantityCall  = 0;
    cashFlow      = 0;
    spesaBidAsk   = NaN;   % se non trovo nulla rimane NaN

    % 1) Filtro sulle date: scadenza >= timegrid(end)
    maturitaMin = timegrid(end);

    maskCallMaturity = (CallexpDates >= maturitaMin);
    maskPutMaturity  = (PutexpDates  >= maturitaMin);

    if flag == 1
        maskCallMaturity = zeros(length(CallexpDates), 1);
        maskCallMaturity(idxC) = 1;
    end

    if flag == 1
        maskPutMaturity = zeros(length(PutexpDates), 1);
        maskPutMaturity(idxP) = 1;
    end

    idxCallCandidates = find(maskCallMaturity);
    idxPutCandidates  = find(maskPutMaturity);

    if isempty(idxCallCandidates)
        warning('selectOptimalPutGivenCall:nessunaCallValida', ...
            'Nessuna call soddisfa il vincolo di scadenza.');
        return;
    end
    if isempty(idxPutCandidates)
        warning('selectOptimalPutGivenCall:nessunPutValido', ...
            'Nessuna put soddisfa il vincolo di scadenza.');
        return;
    end

    % Spread bid-ask
    spreadCall = askCall - bidCall;
    spreadPut  = askPut  - bidPut;

    % Best value della funzione obiettivo (da minimizzare)
    bestSpesa    = inf;
    bestIdxCall  = 0;
    bestIdxPut   = 0;
    bestqC       = 0;
    bestqP       = 0;

    % Tolleranza per singolarità
    tolSing = 1e-12;

    % 2) Loop su tutte le coppie (call, put) che superano il filtro di scadenza
    for ic = idxCallCandidates(:).'   % loop sulle call
        C_vega = ValueChangeCallVega(ic);
        C_eta  = ValueChangeCallEta(ic);

        for jp = idxPutCandidates(:).'  % loop sulle put
            P_vega = ValueChangePutVega(jp);
            P_eta  = ValueChangePutEta(jp);

            % Matrice del sistema per (quantityCall, quantityPut):
            % [ C_vega  P_vega ] [qC] = [ -VegaUpSensitivity ]
            % [ C_eta   P_eta  ] [qP]   [ 0                 ]
            A = [C_vega, P_vega;
                 C_eta,  P_eta];

            b = [-VegaUpSensitivity; 0];

            % Controllo che il sistema non sia quasi singolare
            if rcond(A) < tolSing
                % Sistema mal condizionato → salto questa coppia
                continue;
            end

            % Soluzione (qC, qP)
            q = A \ b;
            qC = q(1);
            qP = q(2);

            % Controllo dell'hedge in eta (soft, con eps)
            etaHedge = qC * C_eta + qP * P_eta;
            if abs(etaHedge) >= eps
                continue;
            end

            % Funzione obiettivo: spesa in bid-ask (uso i moduli delle quantità)
            spesaCand = abs(qP) * spreadPut(jp) + abs(qC) * spreadCall(ic);
            if spesaCand < bestSpesa
    %fprintf('Nuova best: call=%d, put=%d, spesa=%.6g, spreadPut=%.6g, |qP|=%.6g, |qC|=%.6g\n', ...
        %ic, jp, spesaCand, spreadPut(jp), abs(qP), abs(qC));
            end


            % Se migliore, aggiorno soluzione
            if spesaCand < bestSpesa
                bestSpesa   = spesaCand;
                bestIdxCall = ic;
                bestIdxPut  = jp;
                bestqC      = qC;
                bestqP      = qP;
            end
        end
    end

    % Se non ho trovato nessuna coppia call-put valida, esco
    if bestIdxPut == 0
        warning('selectOptimalPutGivenCall:nessunaSoluzione', ...
            'Nessuna combinazione call-put soddisfa i vincoli di hedge.');
        spesaBidAsk  = NaN;
        cashFlow     = NaN;
        quantityCall = 0;
        quantityPut  = 0;
        idxPut       = 0;
        return;
    end

    % Assegno la soluzione migliore trovata
    idxCall       = bestIdxCall;
    idxPut        = bestIdxPut;
    quantityCall  = bestqC;
    quantityPut   = bestqP;
    spesaBidAsk   = bestSpesa;

    % Cash flow (segno in base alla direzione del trade)
    % Se quantity > 0 → compro a ask
    % Se quantity < 0 → vendo a bid
    priceCall = (quantityCall >= 0) * askCall(bestIdxCall) + ...
                (quantityCall <  0) * bidCall(bestIdxCall);
    pricePut  = (quantityPut  >= 0) * askPut(bestIdxPut)  + ...
                (quantityPut  <  0) * bidPut(bestIdxPut);

    cashFlow = quantityCall * priceCall + quantityPut * pricePut;
end

function mu = build_mu_fixed_columns(d, b)
% build_mu_fixed_columns
% Costruisce una curva mensile (36x1) usando:
% - precedenza ai contratti mensili
% - se mancano mesi dentro un trimestre e c'è il trimestrale: riempi i mancanti
%   imponendo la media semplice del trimestre
% - se mancano mesi dentro un anno e c'è l'annuale: riempi i mancanti
%   imponendo la media semplice dell'anno
%
% INPUT:
%   d : datetime (data osservazione)
%   b : 1x10 double [Mc1 Mc2 Mc3 Mc4 Qc1 Qc2 Qc3 Qc4 Yc1 Yc2], con NaN possibili
%
% OUTPUT:
%   mu: 36x1 double, mesi da (mese successivo a d) per 36 mesi

    idx = contract_month_indices_fixed(d);
    mu  = NaN(36,1);

    % 1) Mensili (precedenza assoluta)
    for k = 1:4
        if ~isnan(b(k))
            mu(idx.M(k)) = b(k);
        end
    end

    % 2) Trimestrali: media semplice sui 3 mesi
    for k = 1:4
        Q = b(4+k);                 % Qc(k)
        if isnan(Q), continue; end

        mIdx  = idx.Q(k,:);         % 1x3 (posizioni 1..36)
        known = ~isnan(mu(mIdx));
        miss  = isnan(mu(mIdx));

        if all(known), continue; end

        rhs = 3*Q - sum(mu(mIdx(known)));      % somma dei mancanti
        mu(mIdx(miss)) = rhs / sum(miss);      % spalmo uguale
    end

    % 3) Annuali: media semplice sui 12 mesi
    for k = 1:2
        Y = b(8+k);                 % Yc(k)
        if isnan(Y), continue; end

        mIdx  = idx.Y(k,:);         % 1x12
        known = ~isnan(mu(mIdx));
        miss  = isnan(mu(mIdx));

        if all(known), continue; end

        rhs = 12*Y - sum(mu(mIdx(known)));
        mu(mIdx(miss)) = rhs / sum(miss);
    end

    % 4) Fallback locale (se ancora NaN)
    if any(isnan(mu))
        if any(~isnan(mu))
            mu(isnan(mu)) = mean(mu(~isnan(mu)));
        else
            mu(:) = 0;
        end
    end
end


function idx = contract_month_indices_fixed(d)
% Calcola la mappa (posizioni 1..36) per:
% - Mc1..Mc4 (sempre i primi 4 mesi della griglia)
% - Qc1..Qc4 (primo trimestre "intero" non iniziato + i successivi)
% - Yc1..Yc2 (primo anno solare intero non iniziato + successivo)

    % asse 36 mesi: mese successivo a d
    startMonth = dateshift(d,'start','month') + calmonths(1);
    grid36 = startMonth + calmonths(0:35);

    % ---- Mensili ----
    idx.M = 1:4;

    % ---- Trimestri ----
    qStarts = [1 4 7 10];           % mesi inizio trimestre
    m = month(d);
    y = year(d);

    % primo trimestre "intero" non iniziato: inizio trimestre > mese corrente
    nextQ = qStarts(find(qStarts > m, 1, 'first'));
    if isempty(nextQ)
        nextQ = 1;
        yQ = y + 1;
    else
        yQ = y;
    end
    qStartDate = datetime(yQ, nextQ, 1);

    idx.Q = zeros(4,3);
    for k = 1:4
        qkStart = qStartDate + calmonths(3*(k-1));
        monthsQ = qkStart + calmonths(0:2);
        idx.Q(k,:) = arrayfun(@(dt) find(year(grid36)==year(dt) & month(grid36)==month(dt),1), monthsQ);
    end

    % ---- Annuali ----
    % primo anno solare intero non iniziato
    y1 = y + 1;

    idx.Y = zeros(2,12);
    for k = 1:2
        yk = y1 + (k-1);
        monthsY = datetime(yk,1,1) + calmonths(0:11);
        idx.Y(k,:) = arrayfun(@(dt) find(year(grid36)==year(dt) & month(grid36)==month(dt),1), monthsY);
    end
end

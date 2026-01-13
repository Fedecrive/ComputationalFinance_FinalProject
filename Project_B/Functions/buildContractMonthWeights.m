function W = buildContractMonthWeights(obsMonth)
% buildContractMonthWeights  Costruisce la matrice 10x36 dei pesi dei contratti
%
% INPUT
%   obsMonth : intero 1..12 (mese corrente di osservazione)
%
% OUTPUT
%   W : matrice 10x36
%       Righe 1-4  : mensili  (Mc1..Mc4)
%       Righe 5-8  : trimestrali (Qc1..Qc4)
%       Righe 9-10 : annuali (Yc1..Yc2)
%
% Convenzione colonne:
%   colonna 1 = mese successivo a obsMonth, colonna 36 = 36 mesi avanti.

    % --- controlli ---
    if ~(isscalar(obsMonth) && isnumeric(obsMonth) && isfinite(obsMonth) ...
            && obsMonth == floor(obsMonth) && obsMonth >= 1 && obsMonth <= 12)
        error('obsMonth deve essere un intero tra 1 e 12.');
    end

    H = 36;
    W = zeros(10, H);

    % -------------------------
    % 1) Contratti mensili Mc1..Mc4
    % -------------------------
    % Mc1 = mese +1 (col 1), Mc2 = col 2, ...
    for i = 1:4
        W(i, i) = 1;
    end

    % -------------------------
    % 2) Contratti trimestrali Qc1..Qc4
    % -------------------------
    % Trova il primo trimestre con inizio strettamente dopo obsMonth.
    % Trimestri: (1) Jan-Mar, (2) Apr-Jun, (3) Jul-Sep, (4) Oct-Dec
    currentQuarter = ceil(obsMonth/3);
    firstQuarter = currentQuarter + 1;          % sempre il prossimo trimestre
    if firstQuarter == 5
        firstQuarter = 1;
    end

    quarterStartMonth = [1 4 7 10];             % mesi di inizio trimestre
    qStart = quarterStartMonth(firstQuarter);

    % offset (in colonne) al mese di inizio del primo trimestre
    % formula: offset = mesi avanti fino a qStart, contando col1 = mese successivo
    offsetQ1 = mod(qStart - obsMonth - 1, 12) + 1;   % in {1,...,12}

    % Qc1..Qc4: ciascuno copre 3 mesi, poi si sposta di +3 mesi
    for q = 1:4
        startCol = offsetQ1 + (q-1)*3;
        cols = startCol:(startCol+2);
        cols = cols(cols <= H); % per sicurezza sul bordo dell'orizzonte
        W(4+q, cols) = 1/3;
    end

    % -------------------------
    % 3) Contratti annuali Yc1..Yc2
    % -------------------------
    % Primo anno: il prossimo gennaio strettamente dopo obsMonth.
    % offset al prossimo gennaio (mese 1)
    offsetJan = mod(1 - obsMonth - 1, 12) + 1;  % in {1,...,12}

    % Yc1: 12 mesi da offsetJan
    colsY1 = offsetJan:(offsetJan+11);
    colsY1 = colsY1(colsY1 <= H);
    W(9, colsY1) = 1/12;

    % Yc2: 12 mesi successivi
    colsY2 = (offsetJan+12):(offsetJan+23);
    colsY2 = colsY2(colsY2 <= H);
    W(10, colsY2) = 1/12;

    % -------------------------
    % Check: ogni riga deve sommare a 1
    % (se una riga taglia l'orizzonte di 36 mesi, può sommare < 1: qui NON succede
    % con la tua struttura, ma lasciamo un check robusto)
    % -------------------------
    rs = sum(W, 2);
    if any(abs(rs - 1) > 1e-12)
        error('Alcune righe non sommano a 1. Controlla l''orizzonte o le regole.');
    end
end
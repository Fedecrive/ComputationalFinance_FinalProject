function Ret = buildReturnsPCA(X, toRemove,T)
Ret = diff(log(X));
% un return è valido se entrambe le righe prezzo sono valide
keepRet = ~toRemove(1:end-1) & ~toRemove(2:end);
Ret = Ret(keepRet, :);
dates = T.date;                     % datetime, stessa lunghezza di X (nRows)
sameMonth = (year(dates(2:end))  == year(dates(1:end-1))) & ...
            (month(dates(2:end)) == month(dates(1:end-1)));

Ret = Ret(sameMonth(1:length(Ret(:,1))), :);       % tieni solo return intra-mese
end

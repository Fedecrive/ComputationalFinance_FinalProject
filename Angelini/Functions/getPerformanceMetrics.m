function [annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(equity)
% equity: vettore (T x 1) della equity line (base 100 o qualsiasi livello)

    % 1) Ritorni semplici giornalieri dalla equity
    ret = equity(2:end) ./ equity(1:end-1) - 1;   % T-1 x 1

    daysPerYear = 252;

    % 2) Rendimento annualizzato
    % puoi usare (1+mean(ret))^252 - 1 oppure mean(ret)*252 se vuoi l'approssimazione
    annRet = (1 + mean(ret))^daysPerYear - 1;

    % 3) Volatilità annualizzata
    annVol = std(ret) * sqrt(daysPerYear);

    % 4) Sharpe ratio (assumo rf = 0, altrimenti passalo come input)
    rf = 0;
    Sharpe = (annRet - rf) / annVol;

    % 5) Max Drawdown
    runningMax = cummax(equity);
    drawdowns  = equity ./ runningMax - 1;   % valori <= 0
    MaxDD      = min(drawdowns);             % numero negativo (es: -0.30 = -30%)

    % 6) Calmar ratio (annRet / |MaxDD|)
    Calmar = annRet / abs(MaxDD);

end
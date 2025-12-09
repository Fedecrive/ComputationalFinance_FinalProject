function [annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(equity)
% equity: (T x 1) vector of equity line (base 100)

    % 1) Daily log returns
    LogRet = log(equity(2:end,:) ./ equity(1:end-1,:));
    daysPerYear = 252;

    % 2) Annualized returns 
    annRet = mean(LogRet)*daysPerYear;

    % 3) Annualized volatility 
    annVol = std(LogRet) * sqrt(daysPerYear);

    % 4) Sharpe ratio (assuming rf = 0)
    rf = 0;
    Sharpe = (annRet - rf) / annVol;

    % 5) Max Drawdown
    runningMax = cummax(equity);
    drawdowns  = equity ./ runningMax - 1; 
    MaxDD      = min(drawdowns);             

    % 6) Calmar ratio (annRet / |MaxDD|)
    Calmar = annRet / abs(MaxDD);

end

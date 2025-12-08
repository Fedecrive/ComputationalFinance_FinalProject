function cvar_val = cvar_obj(w, LogRet, alpha)
    pRet = LogRet * w;             % scenario returns
    VaR  = quantile(pRet, 1-alpha);
    tail = pRet(pRet <= VaR);
    cvar_val = -mean(tail);        % minimize losses in the 5% worst cases
end

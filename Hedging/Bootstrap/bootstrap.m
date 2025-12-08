function [disc, fwd_prices, ExpDates] = bootstrap( ...
    CallPrices, PutPrices, CallStrikes, PutStrikes, CallexpDates, PutexpDates)
% BOOTSTRAP
%   Calcola discount factors e forward prices a partire da:
%   - CallPrices, PutPrices
%   - CallStrikes, PutStrikes
%   - CallexpDates, PutexpDates
%
% Output:
%   disc      : [nMat+1 x 1]  discount factors (disc(1)=1 per t=0)
%   fwd_prices: [nMat   x 1]  forward prices per scadenza
%   ExpDates  : [nMat   x 1]  scadenze uniche, ordinate

    % ---------------------------------------------------------------------
    % 1) Scadenze uniche ordinate
    % ---------------------------------------------------------------------
    AllExpDates = [CallexpDates; PutexpDates];

    if ~isdatetime(AllExpDates)
        ExpDates = datetime(AllExpDates);
    else
        ExpDates = AllExpDates;
    end

    ExpDates = unique(ExpDates, 'sorted');  % nMat x 1
    nMat = numel(ExpDates);

    % disc ha una entry in più (t=0)
    disc       = ones(nMat+1, 1);
    fwd_prices = zeros(nMat,   1);

    % ---------------------------------------------------------------------
    % 2) Loop sulle scadenze: bootstrap via regressione C-P contro K
    % ---------------------------------------------------------------------
    for m = 1:nMat
        thisExp = ExpDates(m);

        % indici call/put per questa scadenza
        idxC = (CallexpDates == thisExp);
        idxP = (PutexpDates  == thisExp);

        Kc = CallStrikes(idxC);
        C  = CallPrices(idxC);

        Kp = PutStrikes(idxP);
        P  = PutPrices(idxP);

        % trova solo gli strike in comune tra call e put
        [K_common, ic, ip] = intersect(Kc, Kp);

        if numel(K_common) < 2
            error('Per la scadenza %s non ci sono abbastanza coppie Call/Put con stesso strike.', ...
                string(thisExp));
        end

        C_common = C(ic);
        P_common = P(ip);

        % G = C - P
        G = C_common - P_common;

        % -----------------------------------------------------------------
        % Stima discount factor come -slope della regressione G = a + b*K
        % -----------------------------------------------------------------
        G_mean = mean(G);
        K_mean = mean(K_common);

        num = (G - G_mean)' * (K_common - K_mean);
        den = sum((K_common - K_mean).^2);

        disc(m+1) = -num / den;   % attenzione: m+1 (disc(1)=1 per t=0)

        % Intercetta tramite regressione lineare semplice
        par = [K_common, ones(size(K_common))] \ G;
        % par(1) = slope (dovrebbe essere ~ -disc(m+1))
        % par(2) = intercept = disc * F

        fwd_prices(m) = par(2) / disc(m+1);
    end
end

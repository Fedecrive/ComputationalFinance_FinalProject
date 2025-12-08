function [CallPrices, PutPrices, CallStrikes, PutStrikes, ...
          CallexpDates, PutexpDates, ...
          CallBid, CallAsk, PutBid, PutAsk] = buildOptionPrices(dataFile)
% buildOptionPrices
%   Legge un file CSV di opzioni SPX e restituisce:
%   - mid prices call / put
%   - strike call / put
%   - expiry call / put
%   - bid/ask call / put
%
% Assunto: colonne del CSV sono:
%   'ASK PRICE', 'BID PRICE', 'OPT STRIKE PRICE', 'exp_date', 'type'

    % Leggo senza modificare i nomi
    T = readtable(dataFile, 'VariableNamingRule', 'preserve');

    % Bid e Ask
    Bid = T.("BID PRICE");
    Ask = T.("ASK PRICE");

    % Mid price
    mid = (Bid + Ask) ./ 2;

    % Tipo opzione
    sType = upper(strtrim(string(T.type)));

    isCall = (sType == "C") | (sType == "CALL");
    isPut  = (sType == "P") | (sType == "PUT");

    % Strike ed expiry
    Strike   = T.("OPT STRIKE PRICE");
    ExpDates = T.exp_date;

    % Prezzi mid
    CallPrices = mid(isCall);
    PutPrices  = mid(isPut);

    % Strike
    CallStrikes = Strike(isCall);
    PutStrikes  = Strike(isPut);

    % Expiry
    CallexpDates = ExpDates(isCall);
    PutexpDates  = ExpDates(isPut);

    % Bid/Ask delle call
    CallBid = Bid(isCall);
    CallAsk = Ask(isCall);

    % Bid/Ask delle put
    PutBid = Bid(isPut);
    PutAsk = Ask(isPut);

    % Se vuoi che tutto sia colonna, scommenta:
    %{
    CallPrices   = CallPrices(:);
    PutPrices    = PutPrices(:);
    CallStrikes  = CallStrikes(:);
    PutStrikes   = PutStrikes(:);
    CallexpDates = CallexpDates(:);
    PutexpDates  = PutexpDates(:);
    CallBid      = CallBid(:);
    CallAsk      = CallAsk(:);
    PutBid       = PutBid(:);
    PutAsk       = PutAsk(:);
    %}
end

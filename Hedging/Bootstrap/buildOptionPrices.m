function [CallPrices, PutPrices, CallStrikes, PutStrikes, CallexpDates, PutexpDates] = buildOptionPrices(dataFile)
% buildOptionPrices
%   Legge un file CSV di opzioni SPX e restituisce:
%   - mid prices call / put
%   - strike call / put
%   - expiry call / put
%
% Assunto: colonne del CSV sono:
%   'ASK PRICE', 'BID PRICE', 'OPT STRIKE PRICE', 'exp_date', 'type'

    % Leggo senza far cambiare i nomi a MATLAB
    T = readtable(dataFile, 'VariableNamingRule', 'preserve');

    % Mid price
    Bid = T.("BID PRICE");
    Ask = T.("ASK PRICE");
    mid = (Bid + Ask) ./ 2;

    % Tipo opzione
    sType = upper(strtrim(string(T.type)));

    isCall = (sType == "C") | (sType == "CALL");
    isPut  = (sType == "P") | (sType == "PUT");

    % Strike ed expiry
    Strike   = T.("OPT STRIKE PRICE");
    ExpDates = T.exp_date;

    % Estrazione vettori (mantieni tutti i doppioni)
    CallPrices   = mid(isCall);
    PutPrices    = mid(isPut);

    CallStrikes  = Strike(isCall);
    PutStrikes   = Strike(isPut);

    CallexpDates = ExpDates(isCall);
    PutexpDates  = ExpDates(isPut);

    % % Forza a colonne
    % CallPrices   = CallPrices(:);
    % PutPrices    = PutPrices(:);
    % CallStrikes  = CallStrikes(:);
    % PutStrikes   = PutStrikes(:);
    % CallexpDates = CallexpDates(:);
    % PutexpDates  = PutexpDates(:);
end


function [idxopt1, idxopt2, idxopt3] = findOptions( ...
    expiry1, expiry2, expiry3, ...
    strike1, strike2, strike3, ...
    CallStrikes_new, PutStrikes_new, CallexpDates_new, PutexpDates_new)
% findOptions
%   Given three (expiry, strike) pairs, find the corresponding indices in:
%     - calls for option 1 and 3
%     - puts  for option 2
%
%   If no match is found, the index is returned as NaN.

    % Force column vectors
    CallStrikes_new  = CallStrikes_new(:);
    CallexpDates_new = CallexpDates_new(:);
    PutStrikes_new   = PutStrikes_new(:);
    PutexpDates_new  = PutexpDates_new(:);

    % --------- Option 1: search in calls ---------
    mask1 = (CallStrikes_new == strike1) & (CallexpDates_new == expiry1);
    pos1  = find(mask1, 1);      % take first match if multiple

    if isempty(pos1)
        idxopt1 = NaN;
    else
        idxopt1 = pos1;
    end

    % --------- Option 2: search in puts ----------
    mask2 = (PutStrikes_new == strike2) & (PutexpDates_new == expiry2);
    pos2  = find(mask2, 1);

    if isempty(pos2)
        idxopt2 = NaN;
    else
        idxopt2 = pos2;
    end

    % --------- Option 3: search in calls ---------
    mask3 = (CallStrikes_new == strike3) & (CallexpDates_new == expiry3);
    pos3  = find(mask3, 1);

    if isempty(pos3)
        idxopt3 = NaN;
    else
        idxopt3 = pos3;
    end
end

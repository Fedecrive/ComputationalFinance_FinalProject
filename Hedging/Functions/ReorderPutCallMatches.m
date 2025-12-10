function [CallexpDatesMatched, CallStrikesMatched, CallBidMatched, CallAskMatched, ...
          PutexpDatesMatched,  PutStrikesMatched,  PutBidMatched,  PutAskMatched, ...
          MatchMap] = ...
          ReorderPutCallMatches(CallexpDates, CallStrikes, CallBid, CallAsk, ...
                               PutexpDates,  PutStrikes,  PutBid,  PutAsk)
% ReorderPutCallMatches
%   Keep only options where there is BOTH a call and a put with the same
%   expiry date and strike. All returned vectors are reordered and trimmed
%   so that element i of calls matches element i of puts.
%
%   INPUTS:
%     CallexpDates : vector of call expiry dates (datetime)
%     CallStrikes  : vector of call strikes
%     CallBid      : vector of call bid prices
%     CallAsk      : vector of call ask prices
%     PutexpDates  : vector of put expiry dates (datetime)
%     PutStrikes   : vector of put strikes
%     PutBid       : vector of put bid prices
%     PutAsk       : vector of put ask prices
%
%   OUTPUTS (matched and sorted):
%     CallexpDatesMatched
%     CallStrikesMatched
%     CallBidMatched
%     CallAskMatched
%     PutexpDatesMatched
%     PutStrikesMatched
%     PutBidMatched
%     PutAskMatched
%     MatchMap : N x 2 matrix, where
%               MatchMap(i,1) = index of the call in the ORIGINAL call arrays
%               MatchMap(i,2) = index of the put  in the ORIGINAL put  arrays

    % Force column vectors for calls
    CallexpDates = CallexpDates(:);
    CallStrikes  = CallStrikes(:);
    CallBid      = CallBid(:);
    CallAsk      = CallAsk(:);

    % Force column vectors for puts
    PutexpDates  = PutexpDates(:);
    PutStrikes   = PutStrikes(:);
    PutBid       = PutBid(:);
    PutAsk       = PutAsk(:);

    % Build keys: (expiry, strike) for calls and puts
    callKey = [datenum(CallexpDates), CallStrikes];
    putKey  = [datenum(PutexpDates),  PutStrikes];

    % Find common (expiry, strike) pairs
    [~, idxCall, idxPut] = intersect(callKey, putKey, 'rows');

    % Use these indices to build matched / reordered copies for calls
    CallexpDatesMatched = CallexpDates(idxCall);
    CallStrikesMatched  = CallStrikes(idxCall);
    CallBidMatched      = CallBid(idxCall);
    CallAskMatched      = CallAsk(idxCall);

    % Use these indices to build matched / reordered copies for puts
    PutexpDatesMatched  = PutexpDates(idxPut);
    PutStrikesMatched   = PutStrikes(idxPut);
    PutBidMatched       = PutBid(idxPut);
    PutAskMatched       = PutAsk(idxPut);

    % Map of original indices: column 1 = call index, column 2 = put index
    MatchMap = [idxCall, idxPut];
end

function toRemove = compute_row_filter_nan_blocks(X)
%COMPUTE_ROW_FILTER_NAN_BLOCKS Replicates your NaN row removal logic.
%
% X is the numeric matrix containing all original series (before dropping columns).
% This matches your original block indexing (1:21 in your snippet).

    % Safety: if columns are fewer, adapt (but keep your logic when possible)
    n = size(X,2);

    % Helper to safely take column ranges
    getRange = @(a,b) X(:, max(1,a):min(n,b));

    cond1 = sum(isnan(getRange(1,5)),  2) > 2;
    cond2 = sum(isnan(getRange(6,9)),  2) > 2;
    cond3 = sum(isnan(getRange(10,11)),2) > 1;
    cond4 = sum(isnan(getRange(12,15)),2) > 2;
    cond5 = sum(isnan(getRange(16,19)),2) > 2;
    cond6 = sum(isnan(getRange(20,21)),2) > 1;

    toRemove = cond1 | cond2 | cond3 | cond4 | cond5 | cond6;
end

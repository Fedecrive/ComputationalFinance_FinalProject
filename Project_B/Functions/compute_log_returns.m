function [R, datesR] = compute_log_returns(T)
%COMPUTE_LOG_RETURNS Computes NaN-safe log returns from table T (date + series).
%
% Output:
%   R      (T-1 x N)
%   datesR (T-1 x 1)

    X = T{:,2:end};
    X(~isfinite(X)) = NaN;
    X(X <= 0) = NaN;              % avoids log(0) and log(negative)

    R = diff(log(X));
    R(~isfinite(R)) = NaN;

    datesR = T.date(2:end);
end

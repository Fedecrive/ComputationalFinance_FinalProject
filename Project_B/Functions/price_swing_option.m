function price = price_swing_option(t0, dates, discounts, startDate, endDate, F0, sigma, N, K)
% PRICE_SWING_OPTION Prices an upswings-only swing option on a recombining binomial tree.
%
% INPUTS:
%   t0         : pricing date 
%   dates      : discount curve pillar dates 
%   discounts  : discount factors at 'dates' 
%   startDate  : first delivery/exercise date 
%   endDate    : last delivery/exercise date 
%   F0         : initial forward/spot used as tree root 
%   sigma      : volatility input 
%   N          : maximum number of exercises 
%   K          : strike 
%
% OUTPUT:
%   price      : swing option price at t0 

    % "1) Build business-day date grid"
    datesVec = (startDate:endDate).';
    datesVec = businessdayoffset(datesVec);
    n_days = length(datesVec);

    % "2) Build recombining binomial tree for the underlying"
    S = zeros(n_days, n_days);
    S(1,1) = F0;
    dt = 1 / 252;
    u = exp(norm(sigma) * sqrt(dt));
    d = exp(-norm(sigma) * sqrt(dt));
    for j = 2:n_days
        for i = 1:j
            S(i,j) = F0 * u^(j-i) * d^(i-1);
        end
    end

    % "3) Interpolate discount factors on the exercise grid and compute one-step DFs"
    dfVec = discount_interp(dates, discounts, datesVec, t0);
    dfDaily = dfVec(2:end) ./ dfVec(1:end-1);

    % "4) Initialize terminal payoffs for all remaining-rights layers"
    V = zeros(n_days, n_days, N);
    for z = 1:N
        for i = 1:n_days
            V(i, n_days, z) = max(0, S(i, n_days) - K);
        end
    end

    % "5) Backward induction for n=1"
    for j = n_days-1:-1:1
        for i = 1:j
            Continuation_value = (V(i, j + 1, 1) + V(i + 1, j + 1, 1)) / 2 * dfDaily(j);
            Exercise_value = max(S(i, j) - K, 0);
            V(i, j, 1) = max(Exercise_value, Continuation_value);
        end
    end

    % "6) Backward induction for n=2..N"
    for z = 2:N
        for j = n_days-1:-1:1
            for i = 1:j
                Continuation_value = (V(i, j + 1, z) + V(i + 1, j + 1, z)) / 2 * dfDaily(j);
                Exercise_value = max(S(i, j) - K, 0) + dfDaily(j) * (V(i, j + 1, z - 1) + V(i + 1, j + 1, z - 1)) / 2;
                V(i, j, z) = max(Exercise_value, Continuation_value);
            end
        end
    end

    % "7) Return price"
    price = V(1, 1, N) * dfVec(1);

end

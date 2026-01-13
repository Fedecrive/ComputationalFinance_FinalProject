function price = price_sum_calls(t0, dates, discounts, startDate, endDate, F0, sigma, K)
% PRICE_SUM_CALLS Prices the sum (strip) of daily European call options
% using a recombining binomial tree.
%
% INPUTS:
%   t0         : pricing date 
%   dates      : discount curve pillar dates 
%   discounts  : discount factors at 'dates' 
%   startDate  : first option maturity date 
%   endDate    : last option maturity date 
%   F0         : initial forward/spot used as tree root 
%   sigma      : volatility input 
%   K          : strike 
%
% OUTPUT:
%   price      : price at t0 of the sum of daily European calls

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

    % "3) Interpolate discount factors on the maturity grid and compute one-step DFs"
    dfVec = discount_interp(dates, discounts, datesVec, t0);
    dfDaily = dfVec(2:end) ./ dfVec(1:end-1);

    % "4) Price European calls for each maturity via backward induction"
    C = zeros(n_days, 1);
    for k = 1:n_days
        P = zeros(k, k);

        % Terminal payoff at maturity k
        for i = 1:k
            P(i, k) = max(0, S(i, k) - K);
        end

        % Backward induction
        for j = k-1:-1:1
            for i = 1:j
                P(i, j) = (P(i, j + 1) + P(i + 1, j + 1)) / 2 * dfDaily(j);
            end
        end

        % Time-0 price of the call with maturity k
        C(k) = P(1, 1);
    end

    % "5) Sum all call prices and discount to t0"
    price = sum(C) * dfVec(1);

end

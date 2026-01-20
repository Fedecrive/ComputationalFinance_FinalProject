function [X_DE, X_FR, T_DE, T_FR] = p2_build_monthly_curves(T, nMonths)
%P2_BUILD_MONTHLY_CURVES Builds monthly forward curves for DE and FR.
%
% Assumptions:
%   T columns: date + 20 contracts (first 10 DE, last 10 FR)
%
% Outputs:
%   X_DE (nRows x nMonths)
%   X_FR (nRows x nMonths)
%   T_DE, T_FR split tables (useful later)

    if nargin < 2, nMonths = 24; end

    % Split markets
    T_DE = T(:,1:11);
    T_FR = [T(:,1) T(:,12:end)];

    nRows = height(T_DE);

    X_DE = NaN(nRows, nMonths);
    X_FR = NaN(nRows, nMonths);

    for i = 1:nRows
        % Germany
        d = T_DE.date(i);
        b = T_DE{i,2:end};
        mu = build_mu_fixed_columns(d, b);
        m = min(nMonths, numel(mu));
        X_DE(i,1:m) = mu(1:m).';

        % France
        d = T_FR.date(i);
        b = T_FR{i,2:end};
        mu = build_mu_fixed_columns(d, b);
        m = min(nMonths, numel(mu));
        X_FR(i,1:m) = mu(1:m).';
    end
end

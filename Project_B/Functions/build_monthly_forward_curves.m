function CURVES = build_monthly_forward_curves(T, nMonths)
%BUILD_MONTHLY_FORWARD_CURVES Builds monthly forward curves for DE and FR.
%
% Assumes T columns: date + 20 contracts (first 10 DE, last 10 FR)

    if nargin < 2, nMonths = 24; end

    T_DE = T(:,1:11);
    T_FR = [T(:,1), T(:,12:end)];

    nRows = height(T_DE);

    X_DE = NaN(nRows, nMonths);
    X_FR = NaN(nRows, nMonths);

    for i = 1:nRows
        d = T_DE.date(i);
        b = T_DE{i,2:end};
        mu = build_mu_fixed_columns(d, b);
        m = min(nMonths, numel(mu));
        X_DE(i,1:m) = mu(1:m).';

        d = T_FR.date(i);
        b = T_FR{i,2:end};
        mu = build_mu_fixed_columns(d, b);
        m = min(nMonths, numel(mu));
        X_FR(i,1:m) = mu(1:m).';
    end

    CURVES = struct();
    CURVES.T_DE = T_DE;
    CURVES.T_FR = T_FR;
    CURVES.X_DE = X_DE;
    CURVES.X_FR = X_FR;
end

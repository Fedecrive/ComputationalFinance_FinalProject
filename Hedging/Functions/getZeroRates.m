function zrate = getZeroRates(df, dates)
% getZeroRates  Compute zero rates from discount factors and dates
%
%   df    : discount factors corresponding to dates(2:end)
%   dates : datetime vector, dates(1) is the valuation date t0


    % Valuation date (datetime)
    t0 = dates(1);

    % Year fractions between t0 and each future date (ACT/365 => 3)
    tau = yearfrac(t0, dates(2:end), 3);   % tau is a double vector

    % Make sure df and tau are column vectors with same size
    df_vec  = df(:);
    tau_vec = tau(:);

    % Compute continuous-compounded zero rates: df = exp(-z * tau)
    zrate_vec = -log(df_vec) ./ tau_vec;

    % Reshape back to original shape of df
    zrate = reshape(zrate_vec, size(df));
end

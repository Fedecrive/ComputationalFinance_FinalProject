function diag = p2_compute_diagnostics(R, seriesNames, datesR)
%P2_COMPUTE_DIAGNOSTICS Computes moments, skew/kurt, ADF/JB tests and correlations.
%
% Inputs:
%   R           (T x N) returns matrix
%   seriesNames (1 x N) cellstr/string
%   datesR      datetime vector (optional). If empty, horizon correlations are skipped.
%
% Output struct diag:
%   .stats table (Mean, Std, Skew, Kurtosis)
%   .adf   table (h, pValue)
%   .jb    table (h, pValue)
%   .C     correlation matrix (pairwise)
%   .corr_w, .corr_m (only if datesR provided)

    if iscell(seriesNames)
        names = string(seriesNames);
    else
        names = string(seriesNames);
    end

    % Moments
    meanR = mean(R, 1, 'omitnan');
    stdR  = std(R, 0, 1, 'omitnan');

    % Skewness / Kurtosis (NaN-safe)
    nComp = size(R,2);
    skewR = NaN(1,nComp);
    kurtR = NaN(1,nComp);

    for j = 1:nComp
        x = R(:,j);
        x = x(isfinite(x));
        if numel(x) > 2 && std(x) > 0
            skewR(j) = skewness(x);
            kurtR(j) = kurtosis(x);
        end
    end

    stats = table( ...
        names(:), meanR(:), stdR(:), skewR(:), kurtR(:), ...
        'VariableNames', {'Contract','Mean','Std','Skew','Kurtosis'} );

    % ADF / JB tests
    h_adf = NaN(nComp,1); p_adf = NaN(nComp,1);
    h_jb  = NaN(nComp,1); p_jb  = NaN(nComp,1);

    for j = 1:nComp
        x = R(:,j);
        x = x(isfinite(x));
        if numel(x) > 30 && std(x) > 0
            [h_adf(j), p_adf(j)] = adftest(x);
            [h_jb(j),  p_jb(j)]  = jbtest(x);
        end
    end

    adf = table(names(:), h_adf, p_adf, 'VariableNames', {'Contract','h','pValue'});
    jb  = table(names(:), h_jb,  p_jb,  'VariableNames', {'Contract','h','pValue'});

    % Correlation
    C = corr(R, 'Rows','pairwise');

    diag = struct();
    diag.stats = stats;
    diag.adf   = adf;
    diag.jb    = jb;
    diag.C     = C;

    % Weekly/monthly correlation (only if datesR is provided)
    diag.corr_w = [];
    diag.corr_m = [];
    if nargin >= 3 && ~isempty(datesR)
        TT = array2timetable(R, 'RowTimes', datesR, 'VariableNames', cellstr(names));
        R_w = retime(TT, 'weekly',  'mean');
        R_m = retime(TT, 'monthly', 'mean');
        diag.corr_w = corr(R_w.Variables, 'Rows','pairwise');
        diag.corr_m = corr(R_m.Variables, 'Rows','pairwise');
    end
end

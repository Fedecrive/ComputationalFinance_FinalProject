function diag = compute_return_diagnostics(R, seriesNames, datesR)
%COMPUTE_RETURN_DIAGNOSTICS Moments + skew/kurt + ADF/JB + correlations.
%
% If datesR is empty, weekly/monthly correlations are skipped.

    names = string(seriesNames);

    meanR = mean(R, 1, 'omitnan');
    stdR  = std(R, 0, 1, 'omitnan');

    nComp = size(R,2);
    skewR = NaN(1,nComp);
    kurtR = NaN(1,nComp);

    for j = 1:nComp
        x = R(:,j);
        x = x(isfinite(x));
        if numel(x) > 30 && std(x) > 0
            skewR(j) = skewness(x);
            kurtR(j) = kurtosis(x);
        end
    end

    stats = table(names(:), meanR(:), stdR(:), skewR(:), kurtR(:), ...
        'VariableNames', {'Contract','Mean','Std','Skew','Kurtosis'});

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

    C = corr(R, 'Rows','pairwise');

    diag = struct();
    diag.stats = stats;
    diag.adf   = adf;
    diag.jb    = jb;
    diag.C     = C;
    diag.corr_w = [];
    diag.corr_m = [];

    if nargin >= 3 && ~isempty(datesR)
        TT = array2timetable(R, 'RowTimes', datesR, 'VariableNames', cellstr(names));
        R_w = retime(TT, 'weekly',  'mean');
        R_m = retime(TT, 'monthly', 'mean');
        diag.corr_w = corr(R_w.Variables,'Rows','pairwise');
        diag.corr_m = corr(R_m.Variables,'Rows','pairwise');
    end
end

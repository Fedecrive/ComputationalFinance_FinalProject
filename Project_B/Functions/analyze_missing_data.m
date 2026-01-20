function miss = analyze_missing_data(F, seriesNames, doPlot)
%ANALYZE_MISSING_DATA Prints NaN count and optionally plots NaN map.

    if nargin < 3, doPlot = false; end

    nCols = size(F,2);
    nanCount = zeros(1,nCols);

    fprintf('=== Missing Data Analysis ===\n');
    for j = 1:nCols
        nanCount(j) = sum(isnan(F(:,j)));
        fprintf('%-15s | NaN: %4d |\n', string(seriesNames{j}), nanCount(j));
    end

    miss = struct();
    miss.nanCount = nanCount;
    miss.nanRatio = nanCount ./ size(F,1);

    if doPlot
        figure;
        spy(isnan(F));
        xlabel('Series');
        ylabel('Time');
        title('NaN map');
        pbaspect([1 1 1]);
    end
end

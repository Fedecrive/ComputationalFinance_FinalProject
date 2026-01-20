function miss = p2_missing_data_analysis(F, seriesNames, doPlot)
%P2_MISSING_DATA_ANALYSIS Prints NaN count and optionally plots NaN map.
%
% Inputs:
%   F           (T x N) data matrix (e.g., prices)
%   seriesNames (1 x N) names for printing
%   doPlot      logical
%
% Output:
%   miss struct with fields:
%     .nanCount, .nanRatio

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

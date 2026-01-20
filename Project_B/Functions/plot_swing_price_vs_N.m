function plot_swing_price_vs_N(N_min, N_max, t0, datesDisc, discounts, startDate, endDate, F0, sigma_DE, K)

    N_vec = (N_min:N_max).';
    price_swing_vec = zeros(size(N_vec));
    
    N15 = 15;
    N22 = 22;

    for i = 1:numel(N_vec)
        N_i = N_vec(i);
        price_swing_vec(i) = price_swing_option( ...
            t0, datesDisc, discounts, startDate, endDate, F0, sigma_DE, N_i, K);
    end
    
    price_strip = price_sum_calls(t0, datesDisc, discounts, startDate, endDate, F0, sigma_DE, K);

    % Indices for highlighted points
    idx15 = find(N_vec == N15, 1);
    idx22 = find(N_vec == N22, 1);   
    
    figure;
    
    % Main curve
    plot(N_vec, price_swing_vec, 'LineWidth', 1.5);
    grid on; hold on;
    
    % Upper bound line
    yline(price_strip, '--', 'Strip upper bound', 'LineWidth', 1.2);
    
    % Vertical reference lines (optional, keep them if you like)
    xline(N15, ':', sprintf('N = %d', N15), 'LineWidth', 1.0);
    if ~isempty(idx22)
        xline(22, ':', 'N = 22', 'LineWidth', 1.0);
    end
    
    % Highlight points in red
    if ~isempty(idx15)
        plot(N15, price_swing_vec(idx15), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    end
    if ~isempty(idx22)
        plot(22, price_swing_vec(idx22), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    end
    
    % Make the plot more "centered": add headroom above max line/values
    y_max = max([price_swing_vec; price_strip]);
    y_min = min(price_swing_vec);
    
    top_margin = 0.08;   % 8% headroom
    bot_margin = 0.05;   % 5% below
    ylim([y_min - bot_margin*abs(y_max - y_min), y_max + top_margin*abs(y_max - y_min)]);
    
    % Labels
    xlabel('N (max number of exercises)');
    ylabel('Swing option price');
    title('Swing option price vs N');
    
    legend('Swing price', 'Strip upper bound', 'Highlighted N', 'Location', 'best');
    % exportgraphics(gcf, fullfile("Images","swing_vs_N_K40.pdf"), "ContentType","vector");
    hold off;

end
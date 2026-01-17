function plotHistWithNormalFit(Ret)
isto = Ret(:,1);
isto = isto(~isnan(isto));   
mu = mean(isto);
sigma = std(isto);
figure;
histogram(isto,'Normalization','pdf','NumBins',40);               
hold on;
xx = linspace(min(isto), max(isto), 500);
yy = normpdf(xx, mu, sigma);
plot(xx, yy, 'LineWidth', 2);
grid on;
xlabel('Return');
ylabel('Density');
title('Histogram of returns with fitted normal');
legend('Empirical histogram','Normal(\mu,\sigma^2)');
end

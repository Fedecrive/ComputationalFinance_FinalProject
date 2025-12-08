%% Are returns normal? 
clear all
close all
clc

%% Read Prices
path_map        = 'C:\Users\ginevra.angelini\OneDrive - Anima SGR S.p.A\Desktop\lezioni_poli\lezioni\Lezione3\';
filename        = 'geo_index_prices.xlsx';

table_prices = readtable(strcat(path_map, filename));
%% Calculate Returns (all period)
dt = table_prices(:,1).Variables;
values = table_prices(:,2).Variables;

Ret = log(values(2:end)./values(1:end-1));
% Plot
figure;
histogram(Ret, 'EdgeAlpha', 0.8)
title('Histogram of Empirical Returns')
%% Generation of N  random values from a normal distribution with same mean and variance
N = length(Ret);
NormDistrVar = normrnd(mean(Ret), std(Ret), [N,1]);
%% Ensemble plot
figure;
histogram(Ret, 'Normalization','pdf','EdgeAlpha',0.8)
hold on 
histogram(NormDistrVar,'Normalization','pdf','EdgeAlpha',0.7)
legend('Original Returns Distr', 'Generated Normal Values')
title('Comparison of Empirical vs Normal Distribution')

%% Calculate Moments of the distributions
meanRet = mean(Ret);
varRet = var(Ret);
skewRet = skewness(Ret);
KurtRet = kurtosis(Ret);

meanNorm = mean(NormDistrVar);
varNorm = var(NormDistrVar);
skewNorm = skewness(NormDistrVar);
KurtNorm = kurtosis(NormDistrVar);

%% QQ-plot: graphical test of normality
% If the data are normally distributed, the points should fall on a straight 45° line.
% If the points deviate, especially at the top or bottom, it means the empirical tails are heavier

figure;
qqplot(Ret)
title('QQ-Plot of Returns vs Normal')

%% Kolgornov-Smirnov H0 = values belongs to normal distribution
x = (Ret-meanRet)/sqrt(varRet);
[h,pval] = kstest(x); %if h = 1 rejects the null hypothesis

% Let's see how the test is done!
% Empirical CDF
[ecdf_y, ecdf_x] = ecdf(Ret);
[ecdf_x_unique, ia] = unique(ecdf_x);
ecdf_y_unique = ecdf_y(ia);
% Theoretical Normal CDF
x_vals = linspace(min(Ret), max(Ret), 200);
cdf_normal = normcdf(x_vals, meanRet, std(Ret));

% Interpolate ECDF at same grid as normal CDF
ecdf_interp = interp1(ecdf_x_unique, ecdf_y_unique, x_vals, 'previous','extrap');

% KS statistic = max vertical distance
[KSstat, idx] = max(abs(ecdf_interp - cdf_normal));

% Plot
figure;
plot(ecdf_x, ecdf_y, 'b-', 'LineWidth', 2); hold on
plot(x_vals, cdf_normal, 'r--', 'LineWidth', 2);
% Highlight the max distance
plot([x_vals(idx) x_vals(idx)], [ecdf_interp(idx) cdf_normal(idx)], 'k-', 'LineWidth', 2);
legend('Empirical CDF','Normal CDF','KS distance','Location','best')
xlabel('Returns')
ylabel('CDF')
title(['Kolmogorov-Smirnov Test: KS statistic = ', num2str(KSstat)])

% If the returns were truly normal, there would be only a 0.00000000003% chance of observing a difference this large (KStest) between the empirical CDF and the normal CDF
%% Fat tails check: count returns beyond 3 standard deviations
% Under normality, we expect ~0.27% beyond ±3σ
prop_outliers = sum(abs(Ret-meanRet) > 3*std(Ret)) / N;

% 1.3% This means that extreme events are about 5 times more frequent than what the Normal model would predict

% Identify outliers beyond 3 sigma
mu = meanRet;
sigma = std(Ret);
outliers = Ret(abs(Ret - mu) > 3*sigma);
normal_obs = Ret(abs(Ret - mu) <= 3*sigma);

figure;
hold on
% Plot "normal" returns in blue
histogram(normal_obs,'BinWidth',0.002,'FaceColor',[0.3 0.3 1],'EdgeAlpha',0.5)
% Plot outliers in red
histogram(outliers,'BinWidth',0.002,'FaceColor',[1 0.2 0.2],'EdgeAlpha',0.8)

legend('Within ±3\sigma','Beyond ±3\sigma','Location','best')
title('Fat tails: returns beyond ±3\sigma')
xlabel('Returns')
ylabel('Frequency')

%% Practical connection
% Display a table summarizing results (moments + tests)
Results = table([meanRet;meanNorm],[varRet;varNorm],[skewRet;skewNorm],[KurtRet;KurtNorm],...
    'VariableNames',{'Mean','Variance','Skewness','Kurtosis'},...
    'RowNames',{'Empirical','NormalSimulated'});

disp('--- Moments of Empirical vs Normal Distribution ---')
disp(Results)
disp('--- Normality Tests ---')
disp(['Kolmogorov-Smirnov test rejects normality? h = ', num2str(h), ...
      ' (p-value = ', num2str(pval), ')'])
disp(['Proportion of returns > 3σ = ', num2str(prop_outliers)]);

% Moments 
% The empirical distribution of returns is slightly negatively skewed (left-peak, extreme losses are more frequent than extreme gains)and strongly leptokurtic, with much heavier tails than the comparable normal distribution


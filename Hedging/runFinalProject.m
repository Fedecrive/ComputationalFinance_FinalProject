%% Calibration and Simulation of Additive Models on WTI Options (June 2020)
tic
clear all;
close all;
clc;


addpath('Bootstrap')

%% Get the Data

[CallPrices, PutPrices, CallStrikes, PutStrikes, CallexpDates, PutexpDates] = buildOptionPrices('2017-12-08.csv');
formatData ='dd/MM/yyyy'; % pay attention to your computer settings
t0 = datetime('08-Dec-2017');
% Unisce tutte le expiry
AllExpDates = [CallexpDates; PutexpDates];

% Converte in datetime (se non lo sono già)
if ~isdatetime(AllExpDates)
    AllExpDates = datetime(AllExpDates);
end

% Ordina e rimuove duplicati
dates = unique(AllExpDates, 'sorted');


%% Point 1 - Get Discounts
[disc, fwd_prices, ExpDates] = bootstrap( ...
    CallPrices, PutPrices, CallStrikes, PutStrikes, CallexpDates, PutexpDates);
% zero_rates = zeroRates(dates(2:end), disc(2:end),dates(1)); % get zero rates from discounts
% plot_disc(dates,disc)
% plot_zerorates(dates,zero_rates)



%% Step 2 - Additive  parameters calibration
% Initial guesses
k0     = 1;
eta0   = 0;
sigma0  = 0.2;    
start=1;
alpha=0;
last=length(dates)-2;

tic
[eta, kappa, sigma, MSE, pricesMkt_C, pricesMkt_P] = calibration( ...
    CallPrices, PutPrices, CallStrikes, PutStrikes, fwd_prices, t0, disc, ...
    alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, dates);
dates=[t0; dates];
toc


%% pricing exotic
timegrid = datetime( ...
    {'08/12/2017', '09/04/2018', '08/08/2018', ...
     '10/12/2018', '08/04/2019', '08/08/2019', '09/12/2019'}, ...
    'InputFormat','dd/MM/yyyy');
disc_pricing=interp1(dates, disc, timegrid);
Nsim = 100000;
fwd = interp1(dates(2:end), fwd_prices, timegrid(end));   % forward alla data finale del certificato

[price] = pricing (Nsim, timegrid, fwd,alpha, eta, kappa, sigma, disc_pricing);

%% Certificate sensitivities

[Delta, Vega, Omega] = ComputeSensitivities(CallPrices, PutPrices, CallStrikes, PutStrikes, fwd_prices, fwd, t0, disc, dates, ...
    disc_pricing, alpha, eta0, k0, sigma0, CallexpDates, PutexpDates, timegrid, price, eta, kappa, sigma)

%% 
function [price]=pricing (Nsim, timegrid, fwd,alpha, eta, kappa, sigma, disc)
underlyingMatrix=zeros(Nsim, length(timegrid));
underlyingMatrix(:,1)=fwd*ones(Nsim,1); %fwd relativo alla scadenza finale
if alpha == 0
    L_fun = @(u, t) ...
        exp(-t./kappa .* log(1 + kappa .* u .* sigma.^2));
else
    L_fun = @(u, t) ...
        exp(t./kappa .* (1 - sqrt(1 + 2 .* kappa .* u .* sigma.^2)));
end
a=0.5;
phi_fun = @(u, t) ...
    exp(-1i .* u .* log(L_fun(eta, t))) .* ...
    L_fun(0.5 .* (u.^2 + 1i .* (1 + 2 .* eta) .* u), t);

% Lewis formula vettoriale in x:
% x può essere scalare o vettore
M = 15;    % Grid size parameter
dz = 0.01; % Grid spacing
for i=2:length(timegrid)
    underlyingMatrix(:,i)=underlyingMatrix(:,i-1).*exp(SimulateFromCF(@(u)phi_fun(u,yearfrac(timegrid(i-1),timegrid(i),3)),M,dz,a,Nsim));
    %SimulateFromCF(@(u)phi_fun(u,yearfrac(timegrid(i-1),timegrid(i),3)),M,dz,a,Nsim)

end

% trasformazione matrice dei fwd in matrice degli spot
for i=1:length(timegrid)
    underlyingMatrix(:,i)=underlyingMatrix(:,i)*disc(end)/disc(i);
end

% ==============================
% Parametri del certificato
% ==============================
S = underlyingMatrix;                 % Nsim x 6  (spot sulle 6 dates)
[Nsim, nDates] = size(S);             % nDates = 6 atteso

Notional     = 15e6;
Strike       = S(1,1);                % 100% dello spot iniziale
TriggerLev   = 1.20 * Strike;         % 120%
BarrierLev   = 0.90 * Strike;         % 90%
Factor       = 0.90;
AddFinal     = 0.23;                  % 23%
liq_aut      = [1.02 1.03 1.05 1.10 1.15];  % coeff. autocall (prime 5 date)
liq_final    = 1.20;                  % 120% alla finale

payoffMat = zeros(Nsim, nDates);      % matrice payoff che vuoi

% ==============================
% 1) Autocall sulle prime 5 date
% ==============================
isAutocall = S(:,1:5) >= TriggerLev;          % Nsim x 5 (logico)

% per ogni path, primo indice in cui scatta l’autocall (se esiste)
idxMat = isAutocall .* repmat(1:5, Nsim, 1);  % se falso -> 0, se vero -> indice
idxMat(idxMat == 0) = inf;
[firstAutIdx, ~] = min(idxMat, [], 2);        % Nsim x 1
hasAutocall = firstAutIdx <= 5;

% path che vengono autocallati
idx_aut   = find(hasAutocall);                % indici di riga
autDates  = firstAutIdx(idx_aut);             % data di autocall (1..5)

% assegno il payoff alla data giusta (una sola colonna per riga)
payoffMat(sub2ind(size(payoffMat), idx_aut, autDates)) = ...
    Notional .* liq_aut(autDates).';

% ==========================================
% 2) Path che NON sono mai stati autocallati
% ==========================================
idx_noaut = find(~hasAutocall);

if ~isempty(idx_noaut)

    S_no = S(idx_noaut,:);               % path senza autocall
    ST   = S_no(:, nDates);              % livello finale
    % barriera toccata in QUALSIASI data (barrier discreta sulle 6 date)
    barrierHit = any(S_no <= BarrierLev, 2);

    % maschere per i 4 casi finali
    mA = (ST >= TriggerLev) & ~barrierHit;   % sopra trigger, NO barriera
    mB = (ST <  TriggerLev) & ~barrierHit;   % sotto trigger, NO barriera
    mC = (ST <  TriggerLev) &  barrierHit;   % sotto trigger, barriera
    mD = (ST >= TriggerLev) &  barrierHit;   % sopra trigger, barriera

    payoffFinal = zeros(length(idx_noaut),1);

    % Caso A: (Liquidation price + 23%) * Notional
    payoffFinal(mA) = Notional * (liq_final + AddFinal);

    % Caso B: Liquidation price = ST/Strike * Notional
    payoffFinal(mB) = Notional .* (ST(mB) / Strike);

    % Caso C: max(Protection, Factor * ST/Strike) * Notional
    % Protection = 0, quindi solo parte lineare tagliata a zero
    payoffFinal(mC) = Notional .* max(0, Factor * (ST(mC) / Strike));

    % Caso D: 110% * Notional
    payoffFinal(mD) = Notional * 1.10;

    % metto il payoff alla data finale (colonna 6)
    payoffMat(idx_noaut, nDates) = payoffFinal;
end
DF = disc(:);   % discount factor in ogni data
pv_path = payoffMat * DF;   % Nsim x 1, valore attuale a t=0 di ogni path

price = mean(pv_path);



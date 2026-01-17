function Ret_pca = removeOutliersMahalanobis(Ret)
%% rimozione outliers 
Ret_pca = Ret(all(isfinite(Ret),2), :);

% Media e covarianza classiche
mu = mean(Ret_pca, 1);
S  = cov(Ret_pca);

% Mahalanobis^2 (classica) con pseudoinversa per evitare problemi numerici
Xc = Ret_pca - mu;
Sinv = pinv(S);
d2 = sum((Xc * Sinv) .* Xc, 2);

% Soglia chi-quadro (usa il rango effettivo della covarianza)
r = rank(S);
alpha = 0.995;
thr = chi2inv(alpha, r);

% Outlier e filtraggio
outliers = d2 > thr;
fprintf('Outliers trovati: %d su %d\n', sum(outliers), size(Ret_pca,1));

Ret_pca = Ret_pca(~outliers, :);
fprintf('Osservazioni rimanenti: %d\n', size(Ret_pca,1));
end

function [cumExplained,nComp] = plotPCAExplained(explained,eigenvalues, threshold)
% explained variance
figure;
bar(explained);
xlabel('Principal Component');
ylabel('Explained variance (%)');
title('Explained variance per principal component');
grid on;
cumExplained = cumsum(explained);
figure;
plot(cumExplained,'-o','LineWidth',2);
hold on;

yline(90,'--','90%','LineWidth',1.5);
yline(95,'--','95%','LineWidth',1.5);

xlabel('Number of principal components');
ylabel('Cumulative explained variance (%)');
title('Cumulative explained variance');
grid on;

figure;
plot(eigenvalues, '-o', 'LineWidth', 2);
xlabel('Principal Component');
ylabel('Eigenvalue');
title('Elbow plot (PCA eigenvalues)');
grid on;

% Numero minimo di componenti per spiegare il 90%
nComp = find(cumExplained >= threshold, 1);
fprintf('Number of components explaining %.0f%% variance: %d\n', threshold, nComp);
figure;
plot(cumExplained,'-o','LineWidth',2);
hold on;
yline(threshold,'--','90%','LineWidth',1.5);
xline(nComp,'--',sprintf('PC = %d',nComp),'LineWidth',1.5);

xlabel('Number of principal components');
ylabel('Cumulative explained variance (%)');
title('Cumulative explained variance (90% threshold)');
grid on;
end

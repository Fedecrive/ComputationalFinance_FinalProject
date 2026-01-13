function mu = build_mu_hierarchical(b_full, A_full)
% mu: 36x1 mesi target
nMonths = size(A_full,2);
mu = NaN(nMonths,1);

support = sum(A_full > 0, 2);
rowsM = find(support == 1);
rowsQ = find(support >= 2 & support <= 4);
rowsY = find(support >= 10);

% 1) Mensili: metto subito (precedenza)
for rr = rowsM(:)'
    if isnan(b_full(rr)), continue; end
    m = find(A_full(rr,:) > 0, 1);
    mu(m) = b_full(rr);
end

% 2) Trimestrali: spalmo sui mesi mancanti del trimestre (tenendo fissi i mensili già noti)
for rr = rowsQ(:)'
    if isnan(b_full(rr)), continue; end
    idx = find(A_full(rr,:) > 0);     % i 3 mesi del trimestre
    w   = A_full(rr, idx)';           % pesi (somma 1) -> se vuoi uguali: w = ones(numel(idx),1)/numel(idx);

    known = ~isnan(mu(idx));
    miss  = isnan(mu(idx));

    if all(known), continue; end

    Q = b_full(rr);
    rhs = Q - sum(w(known) .* mu(idx(known)));
    mu(idx(miss)) = rhs / sum(w(miss));
end

% 3) Annuali: come fallback sui mesi ancora NaN
for rr = rowsY(:)'
    if isnan(b_full(rr)), continue; end
    idx = find(A_full(rr,:) > 0);     % 12 mesi dell'anno “attivo” (o comunque quelli con peso>0)
    w   = A_full(rr, idx)';

    known = ~isnan(mu(idx));
    miss  = isnan(mu(idx));
    if all(known), continue; end

    Y = b_full(rr);
    rhs = Y - sum(w(known) .* mu(idx(known)));
    mu(idx(miss)) = rhs / sum(w(miss));
end

% 4) Se resta NaN: fallback locale (non globale sui prezzi)
if any(isnan(mu))
    if any(~isnan(mu))
        mu(isnan(mu)) = mean(mu(~isnan(mu)));
    else
        mu(:) = 0;
    end
end
end

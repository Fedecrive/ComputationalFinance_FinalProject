function P2 = p2_run_point2(T, F, dates, legendLabels, discountFile, opts)
%P2_RUN_POINT2 Packages all Point 2 computations into one call.
%
% Inputs:
%   T            timetable/table with column 1 = date, next columns = contracts
%   F            matrix of prices for plotting (same order as legendLabels)
%   dates        datetime vector for plotting F
%   legendLabels cellstr/string labels for the 20 series (10 DE + 10 FR)
%   discountFile string path to discount_factors.xlsx
%   opts         struct with fields (optional):
%       .doPlots (default false)
%       .nMonths (default 24)
%       .maxLag  (default 20)
%
% Outputs (P2 struct, used later in Exercises 3-5):
%   P2.X_DE, P2.X_FR, P2.X
%   P2.datesDisc, P2.discounts
%   P2.As
%   P2.R_contracts, P2.contractNames
%   P2.diag_contracts, P2.diag_X_DE, P2.diag_X_FR
%   P2.corr_w, P2.corr_m (for contracts)
%   P2.fig (only if doPlots = true)

    if nargin < 6, opts = struct(); end
    if ~isfield(opts, 'doPlots'), opts.doPlots = false; end
    if ~isfield(opts, 'nMonths'), opts.nMonths = 24; end
    if ~isfield(opts, 'maxLag'),  opts.maxLag  = 20; end

    % -------------------------
    % 1) Missing data diagnostics (contracts F)
    % -------------------------
    miss = p2_missing_data_analysis(F, T.Properties.VariableNames(2:end), opts.doPlots);

    % -------------------------
    % 2) Load discount factors
    % -------------------------
    [datesDisc, discounts] = p2_load_discount_factors(discountFile);

    % -------------------------
    % 3) Contract month weights (needed later / useful)
    % -------------------------
    As = zeros(10, 36, 12);
    for m = 1:12
        As(:,:,m) = buildContractMonthWeights(m);
    end

    % -------------------------
    % 4) Build monthly forward curves X_DE and X_FR
    % -------------------------
    [X_DE, X_FR, T_DE, T_FR] = p2_build_monthly_curves(T, opts.nMonths);

    % Combine for PCA later (Exercise 3)
    X = [X_DE X_FR];

    % -------------------------
    % 5) Returns on original contracts
    % -------------------------
    contractNames = T.Properties.VariableNames(2:end);
    R_contracts = diff(log(T{:,2:end}));
    R_contracts(~isfinite(R_contracts)) = NaN;

    datesR = T.date(2:end);

    % -------------------------
    % 6) Diagnostics on returns (contracts)
    % -------------------------
    diag_contracts = p2_compute_diagnostics(R_contracts, contractNames, datesR);

    % -------------------------
    % 7) Diagnostics on monthly forward returns (DE/FR)
    % -------------------------
    R_X_DE = diff(log(X_DE));  R_X_DE(~isfinite(R_X_DE)) = NaN;
    R_X_FR = diff(log(X_FR));  R_X_FR(~isfinite(R_X_FR)) = NaN;

    names_DE = "DE_M" + string(1:size(R_X_DE,2));
    names_FR = "FR_M" + string(1:size(R_X_FR,2));

    diag_X_DE = p2_compute_diagnostics(R_X_DE, names_DE, []);
    diag_X_FR = p2_compute_diagnostics(R_X_FR, names_FR, []);

    % -------------------------
    % 8) Package outputs (variables needed later)
    % -------------------------
    P2 = struct();
    P2.miss = miss;

    P2.datesDisc = datesDisc;
    P2.discounts = discounts;

    P2.As = As;

    P2.T_DE = T_DE;
    P2.T_FR = T_FR;

    P2.X_DE = X_DE;
    P2.X_FR = X_FR;
    P2.X    = X;

    P2.R_contracts = R_contracts;
    P2.contractNames = contractNames;

    P2.diag_contracts = diag_contracts;
    P2.diag_X_DE      = diag_X_DE;
    P2.diag_X_FR      = diag_X_FR;

    P2.corr_w = diag_contracts.corr_w;
    P2.corr_m = diag_contracts.corr_m;

    % -------------------------
    % 9) Plots (optional)
    % -------------------------
    if opts.doPlots
        P2.fig = p2_make_plots(dates, F, legendLabels, T, P2, opts);
    end
end

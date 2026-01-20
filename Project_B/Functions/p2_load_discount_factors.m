function [datesDisc, discounts] = p2_load_discount_factors(discountFile)
%P2_LOAD_DISCOUNT_FACTORS Loads discount factors from an Excel file.
%
% Expected format (as in your code):
%   row 1: Excel dates
%   row 2: discounts
%
% Outputs:
%   datesDisc (datetime vector)
%   discounts (numeric vector)

    discTable = readtable(discountFile);

    excelDates = table2array(discTable(1,:));
    datesDisc = datetime(excelDates, 'ConvertFrom', 'excel');

    discounts = table2array(discTable(2,:));
    discounts = discounts(:);
    datesDisc = datesDisc(:);
end

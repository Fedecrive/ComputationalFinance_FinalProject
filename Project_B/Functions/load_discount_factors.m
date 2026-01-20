function [datesDisc, discounts] = load_discount_factors(discountFile)
%LOAD_DISCOUNT_FACTORS Loads discount factors from an Excel file.
% Expected format:
%   row 1: Excel serial dates
%   row 2: discount factors

    discTable = readtable(discountFile);

    excelDates = table2array(discTable(1,:));
    datesDisc = datetime(excelDates, 'ConvertFrom', 'excel');

    discounts = table2array(discTable(2,:));

    datesDisc = datesDisc(:);
    discounts = discounts(:);
end

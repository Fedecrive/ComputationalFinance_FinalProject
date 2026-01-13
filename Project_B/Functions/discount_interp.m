function [discounts_interp] = discount_interp(dates, discounts, dates_interp, today)
    yf = yearfrac(today, dates, 3);
    yf_interp = yearfrac(today, dates_interp, 3);
    dt = 1/252;
    
    zrates = -log(discounts)./yf;
    
    zrates_interp = interp1(dates, zrates, dates_interp, "linear");
    discounts_interp = exp(- zrates_interp .* yf_interp);
end
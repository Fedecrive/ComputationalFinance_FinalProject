function interp_disc = getDiscountFromZR(disc, dates, target)

t0 = dates(1);
r = getZeroRates(disc, dates);
interp_r = interp1(dates, r, target, "spline");
interp_disc = exp(-interp_r * yearfrac(t0, target, 3));

end
function [lower_bound,upper_bound] = boundsLowerUpper(F1_end,F2_end,discPricing)
lower_bound = discPricing * max(mean(F1_end),mean(F2_end));
upper_bound = discPricing * mean(F1_end+F2_end);
end
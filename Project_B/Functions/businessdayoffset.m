function dates_bus = businessdayoffset(dates)
% BUSINESSDAYOFFSET Adjust weekend dates using Following convention and remove duplicates
%
% INPUT:
%   dates : vector of datetime (or datenums, but datetime recommended)
% OUTPUT:
%   dates_bus : adjusted business dates (Following), with duplicates removed

dates_bus = dates;

for i = 1:length(dates)
    wd = weekday(dates(i)); % 1=Sun, ..., 7=Sat
    if wd == 1
        dates_bus(i) = dates(i) + 1;   % Sunday -> Monday
    elseif wd == 7
        dates_bus(i) = dates(i) + 2;   % Saturday -> Monday
    end
end

% Remove duplicates (keep order of first occurrence)
dates_bus = unique(dates_bus, 'stable');

end

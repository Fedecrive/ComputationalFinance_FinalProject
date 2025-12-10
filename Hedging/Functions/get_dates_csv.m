function [dates_csv, files_csv] = get_dates_csv(startDate, n)

% List all csv files in current folder (adapt pattern if needed)
files = dir('Dati Train\*.csv');

% Preallocate
numFiles   = numel(files);
fileDates  = NaT(numFiles,1);   % datetime array
fileNames  = strings(numFiles,1);

% Extract dates from file names of the form 'YYYY-MM-DD.csv'
for i = 1:numFiles
    fileNames(i) = string(files(i).name);
    % first 10 characters are the date part: 'YYYY-MM-DD'
    dateStr      = extractBetween(fileNames(i),1,10); 
    fileDates(i) = datetime(dateStr,'InputFormat','yyyy-MM-dd');
end

% Keep only files with date >= startDate
mask        = fileDates >= startDate;
fileDates   = fileDates(mask);
fileNames   = fileNames(mask);

% Sort by date (in case dir() is not already sorted)
[fileDatesSorted, sortIdx] = sort(fileDates);
fileNamesSorted            = fileNames(sortIdx);

% Take the first n available dates (handling case with fewer than n)
nTake = min(n, numel(fileDatesSorted));
dates_csv  = fileDatesSorted(1:nTake);
files_csv  = fileNamesSorted(1:nTake);

end
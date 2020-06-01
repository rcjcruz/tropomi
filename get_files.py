import time
import datetime
from datetime import timedelta
import pandas as pd
from glob import glob
import os
import paths

######################


def get_odd_week_number(date):
    """
    Get the nearest odd week number for a given date. If week number is odd,
    return week number. If week number if even, return the floor odd week number.

    date: str(YYYYMMDD)
    """
    d = date.week
    if d % 2 == 0:
        d -= 1
    return d

######################


def get_start_and_end_date_from_calendar_week(date):
    """
    Return start and end date given date.

    date: str(YYYYMMDD)
    """

    # Convert date to datetime
    d = pd.to_datetime(date, format='%Y%m%d', errors='ignore')

    # Get year and week number
    year, calendar_week = d.year, get_odd_week_number(d)

    # Load Monday of week number
    monday = datetime.datetime.strptime(
        f'{year}-{calendar_week}-1', '%Y-%W-%w').date()

    # If Monday of week 1 falls between Jan 5 to Jan 7, subtract 1 from week_number
    # because the Epoch calendar is weird
    week1_start = pd.to_datetime(str(str(year) + '0105'), format='%Y%m%d')
    week1_end = pd.to_datetime(str(str(year) + '0107'), format='%Y%m%d')
    week1_monday = datetime.datetime.strptime(f'{year}-1-1', '%Y-%W-%w').date()

    if week1_start <= week1_monday <= week1_end:
        calendar_week -= 1
        monday = datetime.datetime.strptime(
            f'{year}-{calendar_week}-1', '%Y-%W-%w').date()

    return monday, monday + datetime.timedelta(days=13.9)

##################


def get_files(date):
    """
    Return .txt file with TROPOMI no2 files within a two week range given date.
    """

    # Load start and end date of two week range
    start, end = get_start_and_end_date_from_calendar_week(date)
    outputfile = "inventory.txt"  # file to save the results

    # If text file is not empty, erase it
    if os.path.getsize(outputfile) > 0:
        open(outputfile, 'w').close()

    # Open the text file
    file_object = open(outputfile, "w+")

    # Iterate over the files in the no2 directory and append them to the list
    for file in glob('/export/data/scratch/tropomi/no2/*.nc'):
        date_str = file[53:61]
        # Find date of .nc file and convert to datetime object
        date = pd.to_datetime(date_str, format='%Y%m%d', errors='ignore')
        # Write .nc file to txt file if found between start and end
        if start <= date <= end:
            file_object.write('*__%s*_*.nc' % (date_str))
            file_object.write('\n')

    file_object.close()

######################


if __name__ == '__main__':
    get_files('20200505')

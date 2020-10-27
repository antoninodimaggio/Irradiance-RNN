import argparse
import calendar
import json
import os

import pandas as pd


def define_paths(lat, lon, year):
    make_dir(f'./data/csv/{lat}_{lon}/')
    return f'./data/csv/{lat}_{lon}/{year}.csv'


def is_downloaded(lat, lon, year):
    path = f'./data/csv/{lat}_{lon}/{year}.csv'
    if os.path.isfile(f'./data/csv/{lat}_{lon}/{year}.csv'):
        print(f'Already Downloaded: {path}')
        return True
    print(f'Downloaded: {path}')
    return False


def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def leap_year(year):
    return str(calendar.isleap(int(year))).lower()


def split_years(years):
    return years.split(',')


def format_url(lat, lon, year, interval, attributes='ghi'):
    with open('./config/config.json') as json_file:
        data = json.load(json_file)
    is_leap_year = leap_year(year)
    url_frmt_str = (
        f'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv'
        f'?wkt=POINT({lon}%20{lat})&names={year}&leap_day={is_leap_year}&interval'
        f'={interval}&utc=false&full_name={data["YOUR_NAME"]}'
        f'&email={data["YOUR_EMAIL"]}&affiliation={data["YOUR_AFFILIATION"]}'
        f'&mailing_list=false&reason={data["REASON_FOR_USE"]}'
        f'&api_key={data["API_KEY"]}&attributes={attributes}'
    )
    return url_frmt_str


# TODO; could be to many requests error
def download(lat, lon, year, interval):
    """https://developer.nrel.gov/docs/solar/nsrdb/psm3-download/"""
    url = format_url(lat, lon, year, interval)
    df = pd.read_csv(url, skiprows=2).dropna(axis=1)
    df.insert(
        0, 'Date',
        pd.to_datetime(
            df.loc[:, ['Year', 'Month', 'Day', 'Hour', 'Minute']],
            format='%Y-%m-%d %H:%M'
        )
    )
    df.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1, inplace=True)
    df = df[df.loc[:, 'GHI'] > 0]
    csv_path = define_paths(lat, lon, year)
    df.to_csv(csv_path, index=False)


def download_all(lat, lon, train_years, test_years, interval):
    years = split_years(train_years) + split_years(test_years)
    for year in years:
        if not is_downloaded(lat, lon, year):
            download(lat, lon, year, interval)


def main():
    parser = argparse.ArgumentParser(
        description='Download train and test data from NSRDB'
    )
    parser.add_argument(
        '--lat',
        type=float,
        help=
        'Latitude (to avoid errors make sure this value is within the continental United States) [Required]',
        required=True
    )
    parser.add_argument(
        '--lon',
        type=float,
        help=
        'Longitude (to avoid errors make sure this value is within the continental United States) [Required]',
        required=True
    )
    parser.add_argument(
        '--train-years',
        type=str,
        help=(
            'Comma seperated value string with years to download training data \
        from (1998-2017 according to the official NSRDB docs) [Required]'
        ),
        required=True
    )
    parser.add_argument(
        '--test-years',
        type=str,
        help=(
            'Comma seperated value string with years to download testing data \
        from (1998-2017 according to the official NSRDB docs) [Required]'
        ),
        required=True
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='30',
        help='30 or 60 minute interval data [default: 30]'
    )
    args = vars(parser.parse_args())

    download_all(**args)


if __name__ == '__main__':
    main()

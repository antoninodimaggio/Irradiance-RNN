import argparse
import json
import os
import numpy as np
import pandas as pd


def define_paths(lat, lon, year):
    dir_path = f'./data/csv/{lat}_{lon}/'
    make_dir(dir_path)
    return f'{dir_path}/{year}.csv'


def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def format_url(lat, lon, year, leap_year, interval, attributes='ghi', utc='false',
    mailing_list='false'):
    with open('./config/config.json') as json_file:
        data = json.load(json_file)
    url_frmt_str =  (
        f'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv'
        f'?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap_year}&interval'
        f'={interval}&utc={utc}&full_name={data["YOUR_NAME"]}'
        f'&email={data["YOUR_EMAIL"]}&affiliation={data["YOUR_AFFILIATION"]}'
        f'&mailing_list={mailing_list}&reason={data["REASON_FOR_USE"]}'
        f'&api_key={data["API_KEY"]}&attributes={attributes}'
    )
    return url_frmt_str


def split_years(years):
    """years should be a comma seperated string"""
    return years.split(',')


def download(lat, lon, year, leap_year, interval):
    """https://developer.nrel.gov/docs/solar/nsrdb/psm3-download/"""
    print('Start downloading ...')
    url = format_url(lat, lon, year, leap_year, interval)
    df = pd.read_csv(url, skiprows=2).dropna(axis=1)
    df.insert(0, 'Date', pd.to_datetime(df.loc[:, ['Year', 'Month', 'Day', 'Hour', 'Minute']],
              format= '%Y-%m-%d %H:%M'))
    df.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1, inplace=True)
    df = df[df.loc[:, 'GHI'] > 0]
    csv_path = define_paths(lat, lon, year)
    print(f'Saving csv to the path: {csv_path}')
    df.to_csv(csv_path, index=False)
    print('Done downloading!')
    return csv_path


def download_all(lat, lon, years, leap_year, interval):
    df = pd.DataFrame(columns=['Date','GHI'])
    years = split_years(years)
    for year in years:
        csv_path = download(lat, lon, year, leap_year, interval)
        df = df.append(pd.read_csv(csv_path, parse_dates=True))
    df.to_csv(f'./data/csv/{lat}_{lon}/{"_".join(years)}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Download data from NSRDB')
    parser.add_argument('--lat', type=float,
        help='Latitude (to avoid errors make sure this value is within the continental United States)',
        required=True)
    parser.add_argument('--lon', type=float,
        help='Longitude (to avoid errors make sure this value is within the continental United States)',
        required=True)
    parser.add_argument('--year', type=str, help='Year (1998-2017 according to the official NSRDB docs)', required=True)
    parser.add_argument('--leap-year', type=str, default='false',
        help='Is it a leap year, make sure the string is either true or false (default: false)')
    parser.add_argument('--interval', type=str, default='30', help='30 or 60 minute interval data (default: 30)')

    args = parser.parse_args()

    if (args.leap_year != 'false') and (args.leap_year != 'true'):
        # --leap-year not specified correctly
        parser.print_help()
        raise ValueError('--leap-year should either be true or false make sure everything is lowercase')

    download_all(args.lat, args.lon, args.year, args.leap_year, args.interval)


if __name__ == '__main__':
    """
    Notes:
        lat comes before lon
    """
    """
    python download.py --lat 33.2164 \
      --lon -97.1292 \
      --year 2010,2011 \
      --leap-year false \
      --interval 30
    """
    main()

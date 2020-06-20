import argparse
import json
import numpy as np
import pandas as pd


def download_data(lat, lon, year, leap_year, interval, attributes='ghi',
                  utc='false', mailing_list='false'):
    with open('./config/config.json') as json_file:
        data = json.load(json_file)
    api_key = data['API_KEY']
    your_name = data['YOUR_NAME']
    your_affiliation = data['YOUR_AFFILIATION']
    reason_for_use = data['REASON_FOR_USE']
    your_email = data['YOUR_EMAIL']
    url_frmt_str = ('http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv'
                    '?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval'
                    '={interval}&utc={utc}&full_name={name}&email={email}&affiliation'
                    '={affiliation}&mailing_list={mailing_list}&reason={reason}'
                    '&api_key={api}&attributes={attr}')
    url = url_frmt_str.format(year=year, lat=lat, lon=lon, leap=leap_year,
                              interval=interval, utc=utc, name=your_name, email=your_email,
                              mailing_list=mailing_list, affiliation=your_affiliation,
                              reason=reason_for_use, api=api_key, attr=attributes)
    print('Start downloading ...')
    df = pd.read_csv(url, skiprows=2).dropna(axis=1)
    df.insert(0, 'Date', pd.to_datetime(df.loc[:, ['Year', 'Month', 'Day', 'Hour', 'Minute']],
              format= '%Y-%m-%d %H:%M'))
    df.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1, inplace=True)
    df = df[df.loc[:, 'GHI'] > 0]
    csv_path = f'./data/csv/{lat}_{lon}_{year}.csv'
    print(f'Saving csv to the path: {csv_path}')
    df.to_csv(csv_path, index=False)
    print('Done downloading!')


def main():
    parser = argparse.ArgumentParser(description='Download data from NSRDB')
    parser.add_argument('--lat', type=float,
        help='Latitude (to avoid errors make sure this value is within the continental United States)',
        required=True)
    parser.add_argument('--lon', type=float,
        help='Longitude (to avoid errors make sure this value is within the continental United States)',
        required=True)
    parser.add_argument('--year', type=int, help='Year (1998-2017 according to the official NSRDB docs)', required=True)
    parser.add_argument('--leap-year', type=str, default='false',
        help='Is it a leap year, make sure the string is either true or false (default: false)')
    parser.add_argument('--interval', type=str, default='30', help='30 or 60 minute interval data (default: 30)')

    args = parser.parse_args()

    if (args.leap_year != 'false') and (args.leap_year != 'true'):
        # --leap-year not specified correctly
        parser.print_help()
        raise ValueError('--leap-year should either be true or false make sure everything is lowercase')

    download_data(args.lat, args.lon, args.year, args.leap_year, args.interval)


if __name__ == '__main__':
    main()

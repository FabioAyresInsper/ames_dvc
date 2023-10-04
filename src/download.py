import pathlib
import pickle
import requests

import pandas as pd


def download_data(data_dir):
    raw_data_dir = data_dir / 'raw'
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    raw_data_file_path = data_dir / 'raw' / 'ames.csv'
    if not raw_data_file_path.exists():
        source_url = 'https://www.openintro.org/book/statdata/ames.csv'
        headers = {
            'User-Agent': \
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) ' \
                'AppleWebKit/537.36 (KHTML, like Gecko) ' \
                'Chrome/39.0.2171.95 Safari/537.36',
        }
        response = requests.get(source_url, headers=headers)
        csv_content = response.content.decode()
        with open(raw_data_file_path, 'w', encoding='utf8') as file:
            file.write(csv_content)


def main():
    DATA_DIR = pathlib.Path(__file__).parents[1] / 'data'
    download_data(DATA_DIR)


if __name__ == '__main__':
    main()

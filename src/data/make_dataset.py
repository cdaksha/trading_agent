# -*- coding: utf-8 -*-


# Standard library
import logging
from pathlib import Path
import os

# 3rd party packages
# import click

# Local source
from src.data.stock_io import read_time_series, write_time_series


# @click.command()
# @click.argument('input_file', type=click.Path(exists=True))
# @click.argument('output_file', type=click.Path())
def main(input_file, output_file):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    raw_data = read_time_series(input_file)
    write_time_series(raw_data, output_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    raw_data_path = os.path.join(project_dir, "data", "raw", "AAPL.csv")
    processed_data_path = os.path.join(project_dir, "data", "processed", "AAPL.csv")
    main(raw_data_path, processed_data_path)

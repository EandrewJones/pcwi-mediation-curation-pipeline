import pandas as pd
import json
import os
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from openpyxl import load_workbook

# Configure logger
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][ %(levelname)s ]: %(message)s", "%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

#
# Helper functions
#


def json_to_dataframe(path):
    logger.info("Converting  json list to pandas dataframe")
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return pd.json_normalize(data)


def sort_by_date(df):
    logger.info("Sorting dataframe by ascending date")
    return df.sort_values(by="publication_date", axis=0)


def convert_epochms_to_timestamp(df):
    logger.info("Converting Epoch MS to YYYY-mm-dd HH:MM:SS.MS timestamp")
    df.publication_date = df.publication_date.apply(
        lambda x: datetime.fromtimestamp(x / 1000).strftime("%Y-%m-%d %H:%M:%S.%f")
    )
    return df


def filter_out_negative_predictions(df):
    logger.info("Filtering predicted 0s from data")
    return df[df["label"] == 1]


def transform_pipeline(df):
    df = filter_out_negative_predictions(df=df)
    df = sort_by_date(df=df)
    df = convert_epochms_to_timestamp(df=df)
    return df


def df_to_excel(path, sheet_name, df):
    logger.info(f"Writing dataframe to Excel {path}...")
    try:
        file_exists = Path(path).exists()
        if file_exists:
            book = load_workbook(path)
        with pd.ExcelWriter(path, engine="openpyxl") as excel_writer:
            if file_exists:
                excel_writer.book = book
            df.to_excel(excel_writer=excel_writer, sheet_name=sheet_name, index=False)
            excel_writer.save()
    except Exception as e:
        logger.error(f"Encountered error {e} when writing Excel file")
        raise e
    logger.info("Excel file saved.")


def main():
    # Load data
    completed = [f.split(".")[0] for f in os.listdir("data/dispute-final-out")]
    files = [
        f
        for f in os.listdir("data/dispute-predictions")
        if f.split("?") not in completed
    ]
    for f in files:
        start_time = time.time()
        df = json_to_dataframe(path=os.path.join("data/dispute-predictions", f))
        df = transform_pipeline(df=df)

        # Save output
        outfile = f.split("?")[0] + ".xlsx"
        sheet_name = f.split("?")[1].split(".")[0].replace("_", " ")
        outpath = os.path.join("data/dispute-final-out", outfile)
        logger.info(f"Saving file: {outpath}")
        df_to_excel(path=outpath, sheet_name=sheet_name, df=df)

        logger.info(f"Total runtime: {time.time() - start_time}\n")


if __name__ == "__main__":
    main()

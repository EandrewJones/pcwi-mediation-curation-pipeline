import pandas as pd
import json
import os
import re
from lxml import etree
import time
import pickle
import logging

# Configure logger
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


#
# Data transformation utility classes
#


class ContentParser:
    def __init__(self, xml_str):
        self.tree = etree.fromstring(xml_str)
        self.nsmap = self.get_nsmap()
        # Parse
        self.title = self.get_title()
        self.author = self.get_author()
        self.publication_date = self.get_published()
        self.body_text = self.get_body_text()
        self.lang = self.get_language()

    def get_nsmap(self):
        nsmap = {}
        for ns in self.tree.xpath("//namespace::*"):
            if not ns[0] and ns[1] != "":
                nsmap["atom"] = ns[1]
            elif ns[0]:
                nsmap[ns[0]] = ns[1]
        return nsmap

    def get_title(self):
        title = self.tree.find("atom:title", namespaces=self.nsmap).text
        if not title:
            title_list = self.tree.xpath(".//nitf:hl1", namespaces=self.nsmap)
            title = title_list[0].text if len(title_list) > 0 else None
        return title

    def get_author(self):
        if self.tree.find(".//author/person/nameText") is not None:
            return self.tree.find(".//author/person/nameText").text
        else:
            return None

    def get_published(self):
        return self.tree.find("atom:published", namespaces=self.nsmap).text

    def get_body_text(self):
        text = None
        try:
            text = "\n\n".join(
                self.tree.find("atom:content/articleDoc", namespaces=self.nsmap)
                .find(".//bodyText")
                .itertext()
            )
        except Exception as e:
            logger.error(f"Parser failed to parse document content: {e}")
        return text

    def get_content_map(self):
        return {
            "title": self.title,
            "author": self.author,
            "publication_date": self.publication_date,
            "text": self.body_text,
            "lang": self.lang,
        }

    def get_language(self):
        lang_list = self.tree.find(
            "atom:content/articleDoc", namespaces=self.nsmap
        ).xpath("./@xml:lang")
        return lang_list[0] if len(lang_list) > 0 else "en"


def parse_content(xml):
    content = ContentParser(xml_str=xml)
    return content.get_content_map()


#
# Helper functions
#


def json_to_dataframe(json_list, cols=None):
    """
    Converts json list of fetched results into dataframe and subsets to columns of interest

    params
    ------
    json_list:   list of dicts (JSON), data fetched from NU api
    cols:   optional, list of column names to subset returned dataframe on

    returns
    -------
    pandas dataframe
    """
    logger.info("Converting json list to Pandas DataFrame...")
    df = pd.json_normalize(json_list)
    if cols:
        df = df.filter(items=cols)
    return df


def parse_results_df(results_df):
    """
    Extracts the body text from news articles in API search results
    """
    # Parse XML and append to df
    logger.info("Parsing results...")
    doc_cont_parsed = "Document.Content.Parsed"
    results_df[doc_cont_parsed] = results_df["Document.Content"].apply(parse_content)
    results_df = results_df.join(pd.json_normalize(results_df[doc_cont_parsed]))
    results_df.drop(columns=doc_cont_parsed, inplace=True)
    return results_df


def clean_lang_field_and_subset(results_df, lang="en"):
    logger.info(f"Cleaning language field and subsetting df to language: {lang} ...")
    results_df["lang"] = results_df["lang"].str.lower()
    en_idx = results_df["lang"] == lang  # Subset to desired language
    return results_df.loc[en_idx]


def clean_doc_ids(results_df):
    logger.info("Cleaning document ids...")
    results_df["Document.DocumentId"] = (
        results_df["Document.DocumentId"].str.split("contentItem:").str[1]
    )
    return results_df


def drop_null_rows(df):
    n_rows = df[df["Document.Content"].isnull()].shape[0]
    logger.info(f"Dropping {n_rows} rows with missing content...")
    return df[~df["Document.Content"].isnull()]


def drop_missing_text(df):
    n_rows = df[df["text"].isnull()].shape[0]
    logger.info(f"Dropping {n_rows} rows with missing text after parsing...")
    return df[~df["text"].isnull()]


def drop_unnecessary_cols(df, cols=["Document.Content", "author", "lang"]):
    logger.info("Removing unneeded columns...")
    return df.drop(columns=cols)


def sort_by_date(df, asc=True):
    logger.info(f"Sorting documents by date, {'ascending' if asc else 'descending'}...")
    df["publication_date"] = pd.to_datetime(df["publication_date"])
    return df.sort_values(by="publication_date", ascending=asc)


def transform_pipeline(json_list):
    """
    Pipeline of transformations to perform on the fetched data
    """
    df = json_to_dataframe(
        json_list=json_list,
        cols=["ResultID", "Document.DocumentId", "Document.Content", "Source.Name"],
    )
    df = clean_doc_ids(results_df=df)
    df = drop_null_rows(df=df)
    df = parse_results_df(results_df=df)
    df = clean_lang_field_and_subset(results_df=df)
    df = drop_unnecessary_cols(df=df)
    df = sort_by_date(df=df)
    df = drop_missing_text(df=df)
    return df


def load_json_list(path):
    logger.info(f"Loading file: {path}")
    with open(path, "rb") as infile:
        json_list = json.load(infile)
    return json_list


def df_to_json(path, df):
    logger.info(f"Writing dataframe to JSON {path}...")
    try:
        df.to_json(path, orient="records", lines=True)
    except Exception as e:
        logger.error(f"Encountered error {e} when writing JSON")
        raise e
    logger.info("JSON saved.")


def main():
    logger.info("Retrieving raw data for transforms")
    transformed = [f for f in os.listdir("data/dispute-dataframes/")]
    files = [f for f in os.listdir("data/nu-api-data-raw") if f not in transformed]

    for f in files:
        start_time = time.time()

        logger.info(f"Loading dataset: {f}")
        json_list = load_json_list(path=os.path.join("data/nu-api-data-raw", f))

        if not json_list or len(json_list) < 1:
            continue

        df = transform_pipeline(json_list=json_list)

        outpath = os.path.join("data/dispute-dataframes", f)
        logger.info(f"Saving file: {outpath}")
        df_to_json(path=outpath, df=df)

        logger.info(f"Total runtime: {time.time() - start_time}")


if __name__ == "__main__":
    main()

import pickle
import os
import time
import pandas as pd
import sys

sys.path.append(os.getcwd())
from src.content import parse_content
from src.collections import Collections
from src.search import Search
from src.client import API
from src.config import Config


current_dir = os.path.dirname(__file__)
outdir = os.path.join(current_dir, "../data/")


def search_and_concatenate(api_client, collection, search_str, top=50, filter=None):
    """
    Utility class for performing an API search on the specified collection with the
    given parameters. Iterates through all pages of the search and concatenates into
    a pandas dataframe.

    Params
    ------
    api_client: class method, API client method from the API class
    collection: class variable, A NexisUni collection endpoint from the Collections class
    search_str: str, the api search query
    top: int, number of results to return per page. (Default = 50, max = 50)
    filter: str, optional, additional query filters. Must be in NexisUni approved format.

    Return
    ------
    results_df: pandas data frame, the concatenate raw results from the search.
    """
    # Search
    payload = {"$search": search_str, "$top": top, "$expand": "Document"}
    if filter:
        payload["$filter"] = filter
    s = Search(
        client=api_client,
        collection=collection,
        search_str=search_str,
        params=payload,
        verbose=True,
    )

    # Iterate through all results and combine in a single list
    print("\nConcatenating results...", end="\n")
    all_results = []
    for i, page in enumerate(s):
        print(f"Page: {i+1} / {s.n_pages}", end="\r")
        all_results.extend(page)
        time.sleep(18)  # Sleep request so as not to hit rate limit

    # Check all results captured
    assert (
        len(all_results) == s.n_results
    ), "Did not successfully concatenate all results."

    # convert to dataframe
    results_df = pd.json_normalize(all_results)
    cols = ["ResultID", "Document.DocumentId", "Document.Content", "Source.Name"]
    results_df = results_df.filter(items=cols)

    return results_df


def parse_results_df(results_df):
    """
    Extracts the body text from news articles in API search results
    """
    # Parse XML and append to df
    print("\nParsing results...")
    doc_cont_parsed = "Document.Content.Parsed"
    results_df[doc_cont_parsed] = results_df["Document.Content"].apply(parse_content)
    results_df = results_df.join(pd.json_normalize(results_df[doc_cont_parsed]))
    results_df.drop(columns=doc_cont_parsed, inplace=True)

    return results_df


def clean_doc_ids(results_df):
    results_df["Document.DocumentId"] = (
        results_df["Document.DocumentId"].str.split("contentItem:").str[1]
    )
    return results_df


def main():
    # Batch size
    # TODO: turn into cli arg
    batch_size = 10

    # Load search log
    search_log_path = os.path.join(outdir, "mediation_search_log.pkl")
    with open(search_log_path, "rb") as infile:
        search_log = pickle.load(infile)

    # Instantiate API
    api = API(config=Config)

    # Load in results dataframe if it exists
    mediation_search_results = None
    mediation_search_result_path = os.path.join(outdir, "mediation_search_results.pkl")
    if os.path.exists(mediation_search_result_path):
        with open(mediation_search_result_path, "rb") as infile:
            mediation_search_results = pickle.load(infile)

    # Iterate through dataframe and perform search
    for index, row in search_log.iterrows():
        # If token is expired, refresh
        if api.token_status == "Expired":
            api._fetch_token()

        # Check to see if batch finished
        if batch_size < 1:
            print(f"Batch of size {batch_size} completed.")
            break

        # Skip row if already completed
        # This is necessary so that the index
        # matches the correct row in the dataframe
        # when updating completed searches
        if row["search_completed"]:
            continue

        # Print status
        print(f"Starting search and concatentate for search: {row['search_str']}")

        # Search and concatenate results
        results_df = search_and_concatenate(
            api_client=api.client,
            collection=Collections.BatchNews,
            search_str=row["search_str"],
            filter=row["dates"],
        )

        # clean up doc ID strings
        results_df = clean_doc_ids(results_df=results_df)

        # Check if the Document.DocumentID is in our confirmed list
        # and add binary flag
        results_df["ConfirmedMediation"] = results_df["Document.DocumentId"].isin(
            row["docIds"]
        )

        # Parse news articles
        results_df = parse_results_df(results_df=results_df)

        # Add search string to results
        results_df["search_str"] = row["search_str"]

        # If mediation search results dataframe exists, append new rows and save
        # else save current results as the initial dataframe
        if mediation_search_results is not None:
            print("Updating mediation search results dataframe.")
            mediation_search_results = pd.concat([mediation_search_results, results_df])
            mediation_search_results.reset_index(drop=True, inplace=True)
        else:
            print("Creating new mediation search results dataframe.")
            mediation_search_results = results_df

        tries = 3
        for i in range(tries):
            try:
                print(f"Attempt {i}: Saving mediation search results and log.")
                with open(mediation_search_result_path, "wb") as outfile:
                    pickle.dump(
                        mediation_search_results,
                        outfile,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                # Update original completion status in search log
                # and save
                search_log.loc[index, "search_completed"] = True
                with open(search_log_path, "wb") as outfile:
                    pickle.dump(search_log, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                break
            except Exception as e:
                if i < tries:
                    # Add exponential back-off
                    time.sleep(1 * 2**i)
                    continue
                else:
                    print("Write to disk failed")
                    raise e

        batch_size -= 1


if __name__ == "__main__":
    main()

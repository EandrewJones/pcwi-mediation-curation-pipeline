#!/home/evan/Documents/projects/srdp/SRDP/API/venv/bin/python3
import math
import requests
import time
import backoff
import re
import os
import json
import configparser
import logging
from authlib.integrations.requests_client import OAuth2Session, OAuth2Auth

# Configure logger
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d %(levelname)s - %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Read in configuration variables
parser = configparser.ConfigParser()
parser.read("pipeline.conf")

#
# Constants
#
COMPLETE_SEARCHES_PATH = "data/dispute_searches-complete.json"
INCOMPLETE_SEARCHES_PATH = "data/dispute_searches-incomplete.json"
BOOKMARKS_PATH = "data/bookmarks.json"
MAX_REQUEST_PER_DISPUTE = None
MAX_REQUEST_PER_DAY = 1000

#
# Nexis Uni API Utility Classes
#


class Config:
    """
    Class to store the Nexis Uni API authentication credentials and any other
    universal configuration parameters.
    """

    # Nexis Uni credentials
    CLIENT_ID = parser.get("nu_api_credentials", "CLIENT_ID")
    CLIENT_SECRET = parser.get("nu_api_credentials", "SECRET")
    SCOPE = "http://oauth.lexisnexis.com/all"


class Collections:
    """
    URLs for different API collections.

    Currently only news and batch news included. More can be added as needed
    """

    BaseURL = "https://services-api.lexisnexis.com/v1/"
    News = BaseURL + "News"
    BatchNews = BaseURL + "BatchNews"

    def __iter__(self):
        class_vars = {
            k: getattr(cls, k)
            for k in dir(cls)
            if not callable(getattr(cls, k)) and not k.startswith("__")
        }
        for item in class_vars.items():
            yield item

    def __str__(self):
        printout = ""
        for collection, url in cls:
            printout += f"{collection}: {url}" + "\n"
        return printout


class Search:
    def __init__(
        self,
        client,
        collection,
        search_str,
        params=None,
        verbose=False,
        checkpoint=None,
    ):
        # Instance variables
        self.client = client
        self.payload = {"$search": search_str}
        self.rate_limit = {"minute": 5, "hour": 200, "day": 1000}
        if params:
            self.payload = dict(self.payload, **params)
        self.collection = collection
        if checkpoint:
            logger.info(f"Hot start search from checkpoint: {checkpoint}")
        self.checkpoint = checkpoint

        # Fetch init results and update metadata
        self._get_init_results()

        # Check search
        if self._raw_results.status_code == requests.codes.ok:
            logger.info("Successfully instantiated search.")
            self._update_data()

        if verbose:
            self._print_params()

    def _get_init_results(self):
        """Fetch initial search results.

        returns
        -------
        r: an HTTP request response, same as requests library
        json_results: the response converted to json format
        """
        # Run get request, update metadata, _next_page, _prev_page, rate_limitt
        try:
            if self.checkpoint:
                r = self.client.get(self.checkpoint)
            else:
                r = self.client.get(self.collection, params=self.payload)
            self._raw_results = r
            self._json_results = r.json()
            if "@odata.count" not in self._json_results.keys():
                self.n_pages = 0
            else:
                self.n_results = self._json_results["@odata.count"]
                self.n_pages = math.ceil(
                    self.n_results / len(self._json_results["value"])
                )
            for k in self.rate_limit.keys():
                self.rate_limit[k] -= 1
        except Exception as e:
            logger.error(f"Error encountered: {e}")

    def _update_data(self):
        """Store initial results."""
        self._next_link = (
            self._json_results["@odata.nextLink"]
            if "@odata.nextLink" in self._json_results.keys()
            else None
        )
        self._values = (
            self._json_results["value"]
            if "value" in self._json_results.keys()
            else None
        )
        remaining = self._raw_results.headers["X-RateLimit-Remaining"].split("/")
        for k, calls in zip(self.rate_limit.keys(), remaining):
            self.rate_limit[k] = int(calls)

    def _print_params(self):
        """Print search params."""
        logger.info("Lexis Nexis API Search Parameters")
        logger.info(f"Collection: {self.collection}")
        for k, v in self.payload.items():
            logger.info(f"{k}: {v}")

    @property
    def results(self):
        return self._values

    def next_page(self):
        if not self._next_link:
            logger.info("Already on last page.")
            return
        self._raw_results = self.client.get(self._next_link)
        if self._raw_results.status_code == requests.codes.ok:
            self._json_results = self._raw_results.json()
            self._update_data()
        else:
            raise requests.HTTPError(self._raw_results)
        return self.results

    def __iter__(self):
        # Not final page of results,
        # retrieve results,
        # fetch next page,
        # yield results, repeat
        while self._next_link:
            results = self.results
            next_link = self._next_link
            self.next_page()
            yield (results, next_link)
        # Last page already retrieved, yield results
        yield (self.results, None)


class OAuth2SessionBackoff(OAuth2Session):
    def __init__(
        self,
        client_id=None,
        client_secret=None,
        token_endpoint_auth_method=None,
        revocation_endpoint_auth_method=None,
        scope=None,
        redirect_uri=None,
        token=None,
        token_placement="header",
        update_token=None,
        **kwargs,
    ):
        OAuth2Session.__init__(
            self,
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method=token_endpoint_auth_method,
            revocation_endpoint_auth_method=revocation_endpoint_auth_method,
            scope=scope,
            redirect_uri=redirect_uri,
            token=token,
            token_placement=token_placement,
            update_token=update_token,
            **kwargs,
        )

    @backoff.on_exception(
        backoff.expo, requests.exceptions.RequestException, max_tries=8
    )
    def backoff_get(self, *args, **kwargs):
        return self.get(*args, **kwargs)


class API:
    """
    OAuth2 Client Session for connecting to Lexis Nexis API.

    params        # Replace client get with backoff get
        self.client.get = backoff_get
    ------
    path_to_env, str: absolute path to an .env file containing
        the OAuth Client ID and Client Secret
    """

    def __init__(self, config):
        self.client = self.create_client(config=config)
        self._fetch_token()
        self.collections = Collections()

    def create_client(self, config):
        return OAuth2SessionBackoff(
            config.CLIENT_ID, config.CLIENT_SECRET, scope=config.SCOPE
        )

    def _fetch_token(self):
        logger.info("Fetching token...")
        token_endpoint = "https://auth-api.lexisnexis.com/oauth/v2/token"
        self._token = self.client.fetch_token(
            token_endpoint, grant_type="client_credentials"
        )
        logger.info("Success")

    @property
    def token(self):
        """Returns LexisNexis API Client Access Token."""
        if self.token_status == "Expired":
            self._fetch_token()
        return self._token["access_token"]

    @property
    def token_status(self):
        """Returns token Status."""
        self._token_status = "Valid"
        if self._token["expires_at"] < time.time():
            self._token_status = "Expired"
        return self._token_status

    @property
    def token_metadata(self):
        """Returns token metadata."""
        self._token_metadata = {
            k: v for k, v in self._token.items() if k != "access_token"
        }
        return self._token_metadata


#
# API Call Helper Functions
#


def search_dict_to_filename(search_dict):
    assert set(["name", "search_str", "date", "n_requests"]) == set(
        search_dict.keys()
    ), "search dict missing required arguments"
    date_str = re.sub("\s", "_", search_dict["date"])
    dispute_name = search_dict["name"]
    return f"data/nu-api-data-raw/{dispute_name}?{date_str}.json"


def load_json_list(path):
    logger.info(f"Loading json list at {path}")
    with open(path, "rb") as infile:
        json_list = json.load(infile)
    return json_list


def save_json(path, data):
    logger.info(f"Saving json to {path}")
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False)


def search_and_concatenate(
    api_client,
    collection,
    search_str,
    top=50,
    filter=None,
    checkpoint=None,
    page_limit=None,
):
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
    checkpoint: str, optional, a url request to hot start a search in the event of prior failure.
    page_limit: int, optional, a limit on the number of pages to iterate through before stopping.

    Return
    ------
    all_results: list, dicts/json objects
    fetch_incomplete: boolean, whether the function successfully retrieved all pages of results
    """
    all_results = []
    n_requests = 0

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
        checkpoint=checkpoint,
    )

    # Find starting page number on hot start
    start_page = (
        int(re.findall(r"\$skip=(\d+)", checkpoint)[0]) // 50 if checkpoint else 1
    )

    # Exit early for no results
    if s.n_pages < 1:
        return all_results, None, s.rate_limit["day"]

    # Iterate through all results and combine in a single list
    logger.info("Concatenating results...")
    try:
        for i, payload in enumerate(s):
            n_requests = i + 1
            logger.info(f"Page: {i + start_page} / {s.n_pages}")
            page, next_link = payload
            all_results.extend(page)

            # Exit early if limit reached
            if s.rate_limit["day"] == 0:
                logger.info(
                    f"Request limit of 1000 reached for 24 hour periood. Stopping iteration."
                )
                break
            elif page_limit and n_requests == page_limit:
                logger.info(
                    f"Request limit of {page_limit} reached. Stopping iteration."
                )
                break

            time.sleep(18)  # Sleep request so as not to hit hourly rate limit
    except Exception as e:
        logger.error(f"Encountered exception when fetching data: {e}")
        # store file name as job bookmark
        return all_results, next_link, s.rate_limit["day"]
    else:
        # Check all results captured on success
        if len(all_results) != s.n_results:
            logger.warning("Did not successfully concatenate all results.")
            logger.warning(
                f"\tOnly {len(all_results)} / {s.n_results} successfully fetched."
            )
        return all_results, next_link, s.rate_limit["day"]


def main():
    """
    Workflow:
    1) Load in search log as list of JSON
    2) Begin search with item at tail end of list: List[-1]
    3)
    """
    # Instantiate API
    api = API(config=Config)

    # Load search log and bookmarks
    if os.path.exists(COMPLETE_SEARCHES_PATH):
        complete_searches = load_json_list(COMPLETE_SEARCHES_PATH)
    else:
        complete_searches = []
    incomplete_searches = load_json_list(INCOMPLETE_SEARCHES_PATH)
    bookmarks = load_json_list(BOOKMARKS_PATH)

    incomplete_results = []

    # Iterate through incomplete searches and perform search
    daily_request_remaining = 1000  # Assume 1000 by default, let API update
    while daily_request_remaining > 0:
        # If token is expired, refresh
        if api.token_status == "Expired":
            api._fetch_token()

        # Fetch search arguments
        params = incomplete_searches[-1]

        # If the job is a retry load incomplete results from last job (job bookmark)
        if bookmarks["next_page"] is not None:
            incomplete_results = load_json_list(bookmarks["incomplete_results_path"])

        # Exit early if next call will exceed limit
        # if params["n_requests"] + total_requests > MAX_REQUEST_PER_DAY:
        #     logger.info(f"Daily rate limit will be exceeded by next call. Exiting job.")
        #     break

        # Print status
        logger.info(f"Starting API calls for: {params['name']}\t{params['date']}")
        filename = search_dict_to_filename(search_dict=params)

        # Search and concatenate results
        results, next_link, daily_request_remaining = search_and_concatenate(
            api_client=api.client,
            collection=Collections.BatchNews,
            search_str=params["search_str"],
            filter=params["date"],
            checkpoint=bookmarks["next_page"],
            page_limit=MAX_REQUEST_PER_DISPUTE,
        )

        # Save results to s3 bucket with filename
        if bookmarks["next_page"] is not None:
            incomplete_results.extend(results)
            results = incomplete_results
        save_json(filename, results)

        # If search didn't finish, store filename as a job_bookmark
        # to be reloaded on job rety
        if next_link:
            logger.warning(f"Job incomplete. Updating bookmarks:")
            logger.info(f"next_page: {next_link}")
            logger.info(f"incomplete_results_path: {filename}")
            bookmarks = {"next_page": next_link, "incomplete_results_path": filename}
        else:
            # Update job state
            complete_searches.append(incomplete_searches.pop())
            bookmarks = {"next_page": None, "incomplete_results_path": ""}
            # Save job meta data
            save_json(path=COMPLETE_SEARCHES_PATH, data=complete_searches)
            save_json(path=INCOMPLETE_SEARCHES_PATH, data=incomplete_searches)
        save_json(path=BOOKMARKS_PATH, data=bookmarks)


if __name__ == "__main__":
    main()

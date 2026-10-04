"""Exit 0 once every TimeWarp server listed in TIMEWARP_SITES serves its start page."""

import os
import sys
import urllib.error
import urllib.request

START_URLS = {
    "wiki": "http://localhost:5000/",
    "news": "http://localhost:5001/",
    # WebShop loads its products and search index on the first request.
    "webshop": "http://localhost:5002/abc",
}


def main() -> int:
    sites = [site for site in os.environ["TIMEWARP_SITES"].split(",") if site]
    for site in sites:
        try:
            with urllib.request.urlopen(START_URLS[site], timeout=120) as response:
                if response.status >= 500:
                    print(f"{site}: HTTP {response.status}")
                    return 1
        except (urllib.error.URLError, OSError) as error:
            print(f"{site}: {error}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

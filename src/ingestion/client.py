import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


VELIB_NETWORK_ENDPOINT = "http://api.citybik.es/v2/networks/velib"


def fetch_velib_data() -> dict:
    try:
        with urlopen(VELIB_NETWORK_ENDPOINT, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"CityBikes HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"CityBikes network error: {exc.reason}") from exc

    return json.loads(payload)

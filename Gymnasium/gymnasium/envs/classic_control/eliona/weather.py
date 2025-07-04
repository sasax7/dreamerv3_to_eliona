import openmeteo_requests
import requests_cache
from retry_requests import retry


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def get_weather_data_with_timestamp(config, timestamp=None):
    weather_attributes = config["environment"].get("weather_attributes", {})
    if (
        not weather_attributes
        or weather_attributes.get("latitude") is None
        or weather_attributes.get("longitude") is None
    ):
        return None

    """
    Fetch weather data using Open-Meteo API for the given configuration and timestamp.

    :param config: Configuration dictionary containing latitude, longitude, and weather attributes to extract.
    :param timestamp: The timestamp to fetch weather data for. If None, fetches current weather data.
    :return: Dictionary containing requested weather attributes.
    """
    latitude = config["environment"]["weather_attributes"]["latitude"]
    longitude = config["environment"]["weather_attributes"]["longitude"]
    hourly_attrs = config["environment"]["weather_attributes"].get("hourly", [])
    daily_attrs = config["environment"]["weather_attributes"].get("daily", [])
    select_attrs = config["environment"]["weather_attributes"].get("select", set())

    def extract_selected(data, select_attrs, prefix):
        """
        Extract specific values based on the `select` configuration.
        :param data: List of values (hourly or daily).
        :param select_attrs: Set of selected attributes (e.g., "temperature_2m_1").
        :param prefix: Prefix of the attribute (e.g., "temperature_2m").
        :return: Filtered dictionary of selected values.
        """
        selected_data = {}
        for attr in select_attrs:
            if attr.startswith(prefix):
                index = (
                    int(attr.split("_")[-1]) - 1
                )  # Extract index from "temperature_2m_1"
                if 0 <= index < len(data):
                    selected_data[attr] = data[index]
        return selected_data

    if timestamp is None:
        # Fetch current weather data
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(hourly_attrs),
            "daily": ",".join(daily_attrs),
            "current_weather": True,
        }
        response = openmeteo.weather_api(
            "https://api.open-meteo.com/v1/forecast", params=params
        )
        weather_results = {}

        # Process hourly and daily data
        hourly = response[0].Hourly()
        for index, attr in enumerate(hourly_attrs):
            values = hourly.Variables(index).ValuesAsNumpy().tolist()
            weather_results.update(extract_selected(values, select_attrs, attr))

        daily = response[0].Daily()
        for index, attr in enumerate(daily_attrs):
            values = daily.Variables(index).ValuesAsNumpy().tolist()
            weather_results.update(extract_selected(values, select_attrs, attr))

        return weather_results

    # Historical weather
    date_str = timestamp.strftime("%Y-%m-%d")
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(hourly_attrs),
        "daily": ",".join(daily_attrs),
    }
    response = openmeteo.weather_api(
        "https://historical-forecast-api.open-meteo.com/v1/forecast", params=params
    )
    weather_results = {}

    # Process hourly data
    hourly = response[0].Hourly()
    for index, attr in enumerate(hourly_attrs):
        values = hourly.Variables(index).ValuesAsNumpy().tolist()
        weather_results.update(extract_selected(values, select_attrs, attr))

    # Process daily data
    daily = response[0].Daily()
    for index, attr in enumerate(daily_attrs):
        values = daily.Variables(index).ValuesAsNumpy().tolist()
        weather_results.update(extract_selected(values, select_attrs, attr))

    return weather_results


# Example configuration
config = {
    "asset_id": 3547,
    "environment": {
        "observation_attributes": {
            "temperatur": {"low": -5, "high": 50},
            "humidity": {"low": 0, "high": 100},
            "Aussentemperatur": {"low": -5, "high": 50},
        },
        "action_attributes": {
            "heating_on_off": {"type": "discrete", "values": [0, 1]},
            "humidifyer_on_of": {"type": "discrete", "values": [0, 1]},
        },
        "time_attributes": {
            "time_of_day": {"low": -1, "high": 25},
        },
        "weather_attributes": {
            "latitude": 52.52,
            "longitude": 13.405,
            "hourly": ["temperature_2m", "relative_humidity_2m"],
            "daily": ["temperature_2m_max"],
            "select": {
                "temperature_2m_1",
                "temperature_2m_6",
                "relative_humidity_2m_1",
                "relative_humidity_2m_6",
                "temperature_2m_max_1",
            },
        },
    },
    "reward_function": {
        "attributes": [
            {
                "name": "temperatur",
                "type": "range",
                "target_range": [19, 23],
                "reward_factor": 0.1,
                "timerange": {"time": "time_of_day", "range": [[6, 16], [17, 22]]},
            },
            {
                "name": "humidity",
                "type": "range",
                "target_range": [40, 60],
                "reward_factor": 0.1,
                "timerange": {"time": "time_of_day", "range": [[6, 16], [17, 22]]},
            },
        ]
    },
}

# # Example usage
# # Fetch current weather data
# current_weather = get_weather_data_with_timestamp(config)
# print("Current Weather:")
# print(json.dumps(current_weather, indent=2))

# # Fetch historical weather data for a specific timestamp
# timestamp = datetime.datetime(2023, 1, 1, 12)
# historical_weather = get_weather_data_with_timestamp(config, timestamp)
# print("\nHistorical Weather:")
# print(json.dumps(historical_weather, indent=2))

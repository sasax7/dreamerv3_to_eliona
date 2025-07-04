import datetime
from eliona.api_client2 import ApiClient, Configuration, DataApi, ApiException
from eliona.api_client2.models import Data
import os


ELIONA_API_KEY = os.getenv("API_TOKEN")
ELIONA_HOST = os.getenv("API_ENDPOINT")


def write_into_eliona(asset_id, data):
    configuration = Configuration(
        host=ELIONA_HOST, api_key={"ApiKeyAuth": ELIONA_API_KEY}
    )
    with ApiClient(configuration) as api_client:
        data_api = DataApi(api_client)

        data = Data(asset_id=asset_id, subtype="input", data=data)
        data_api.put_data(data)


def get_eliona_heap_data(asset_id):
    configuration = Configuration(
        host=ELIONA_HOST, api_key={"ApiKeyAuth": ELIONA_API_KEY}
    )
    with ApiClient(configuration) as api_client:
        data_api = DataApi(api_client)
        try:
            result = data_api.get_data(asset_id=asset_id)
            return result
        except ApiException as e:
            print(f"Exception when calling DataApi->get_data: {e}")
            return None


def get_trend_data(asset_id, start_date, end_date):
    configuration = Configuration(
        host=ELIONA_HOST, api_key={"ApiKeyAuth": ELIONA_API_KEY}
    )
    with ApiClient(configuration) as api_client:
        data_api = DataApi(api_client)
        from_date = start_date.isoformat()
        to_date = end_date.isoformat()
        try:
            print(f"Fetching data from {from_date} to {to_date} of asset {asset_id}")
            result = data_api.get_data_trends(
                from_date=from_date, to_date=to_date, asset_id=asset_id
            )
            return result
        except ApiException as e:
            print(f"Exception when calling DataApi->get_data_trends: {e}")
            return None


def fetch_data_in_chunks(asset_id, start_date, end_date):
    all_data = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + datetime.timedelta(days=5), end_date)
        data_chunk = get_trend_data(asset_id, current_start, current_end)
        if data_chunk:
            all_data.extend(data_chunk)
        current_start = current_end + datetime.timedelta(seconds=1)
    return all_data


def convert_to_dict(data):
    formatted_data = {}
    for entry in data:
        timestamp = entry.timestamp.isoformat()
        data_dict = entry.data
        if timestamp in formatted_data:
            formatted_data[timestamp].update(data_dict)
        else:
            formatted_data[timestamp] = data_dict
    return formatted_data


from collections import defaultdict



def get_eliona_trend_data(start_date, end_date, environment):
    """
    Args:
      start_date (datetime w/ tz) – first timestamp you actually want
      end_date   (datetime w/ tz) – last timestamp you actually want
      environment – list of blocks, each:
         {
           "asset_id": 4271,
           "observation_attributes": {"NA_TA": {...}, …},
           "action_attributes":      {"KO_SOLL": {...}, …}
         }

    Returns:
      List[{"timestamp": iso-str,
            "state": {obs_key: float, …},
            "action": {act_key: float, …}}]
      – one entry per *observation change*, with both obs+action forward-filled.
    """
    # 1) we fetch 7 days early so forward-fill has context
    buffer_start = start_date - datetime.timedelta(days=7)

    # 2) group all attributes by asset_id so we only call the API once per asset
    by_asset = defaultdict(lambda: {"obs": set(), "act": set()})
    for block in environment:
        aid = block["asset_id"]
        by_asset[aid]["obs"].update(block.get("observation_attributes", {}).keys())
        by_asset[aid]["act"].update(block.get("action_attributes", {}).keys())

    # This will accumulate every timestamp -> merged state+action
    merged_raw = defaultdict(lambda: {"timestamp": None, "state": {}, "action": {}})
    all_obs_keys = set()
    all_act_keys = set()

    # 3) fetch once per asset
    for asset_id, groups in by_asset.items():
        obs_keys = list(groups["obs"])
        act_keys = list(groups["act"])
        all_obs_keys.update(obs_keys)
        all_act_keys.update(act_keys)

        # fetch & prepare
        data = fetch_data_in_chunks(asset_id, buffer_start, end_date)
        flat    = convert_to_dict(data)
        times   = sorted(flat.keys())

        last_obs = {}
        last_act = {}

        # for each timestamp, update “last known” & emit if an obs changed
        for ts in times:
            row = flat[ts]
            obs_changed = False

            # update observations
            for k in obs_keys:
                if k in row:
                    if last_obs.get(k) != row[k]:
                        obs_changed = True
                    last_obs[k] = row[k]

            # update actions silently
            for k in act_keys:
                if k in row:
                    last_act[k] = row[k]

            # emit only on obs change
            if obs_changed:
                merged_raw[ts]["timestamp"] = ts
                for k in obs_keys:
                    # take the latest known
                    merged_raw[ts]["state"][k] = last_obs.get(k)
                for k in act_keys:
                    merged_raw[ts]["action"][k] = last_act.get(k)

    # 4) build sorted list and drop buffer timestamps
    final = []
    for ts in sorted(merged_raw):
        # keep only if ≥ real start_date
        dt = datetime.datetime.fromisoformat(ts)
        if dt < start_date:
            continue
        final.append({
            "timestamp": ts,
            "state":  merged_raw[ts]["state"].copy(),
            "action": merged_raw[ts]["action"].copy(),
        })

    # 5) forward-fill missing obs & actions
    for k in all_obs_keys:
        last = None
        for entry in final:
            if k in entry["state"]:
                last = entry["state"][k]
            elif last is not None:
                entry["state"][k] = last

    for k in all_act_keys:
        last = None
        for entry in final:
            if k in entry["action"]:
                last = entry["action"][k]
            elif last is not None:
                entry["action"][k] = last

    return final



# data = get_eliona_trend_data(
#     3547,
#     pytz.timezone("Europe/Berlin").localize(datetime.datetime(2025, 1, 13, 15, 11)),
#     pytz.timezone("Europe/Berlin").localize(datetime.datetime(2025, 1, 13, 15, 25)),
#     ["temperatur"],
#     ["heating_on_off"],
# )
# import pprint

# pprint.pprint(data)

import numpy as np



from gymnasium.envs.classic_control.eliona.helper_fuctions import datetime_to_time_attributes
from gymnasium.envs.classic_control.eliona.weather import get_weather_data_with_timestamp
import datetime

import dateutil.parser
from gymnasium.envs.classic_control.eliona.reward_calculation import calculate_reward


import numpy as np
import datetime

import gymnasium as gym

import numpy as np
def add_time_attributes_to_state(state, dt: datetime.datetime, config):
    """
    Add time attributes to 'state' if time_attributes exist in the config.
    """
    time_config = config["environment"].get("time_attributes", {})
    if not time_config:  # If empty or missing, just return
        return

    # dt is already a datetime object
    time_attributes = datetime_to_time_attributes(dt)
    for key, value in time_attributes.items():
        # Only add if it's in config["environment"]["time_attributes"]
        if key in time_config:
            state[key] = value


def add_weather_data_to_state(state, dt: datetime.datetime, config):
    """
    Add weather attributes to 'state' if weather_attributes exist in the config.
    """
    weather_config = config["environment"].get("weather_attributes", {})
    if not weather_config:  # If empty or missing, just return
        return

    # Attempt to fetch the weather data
    weather_data = get_weather_data_with_timestamp(config, dt)
    if weather_data is None:
        return

    # Only add the keys that appear in config["environment"]["weather_attributes"]["select"]
    selected_keys = weather_config.get("select", {})
    for key, value in weather_data.items():
        if key in selected_keys:
            state[key] = value


def enrich_data_with_time_and_weather(data, config):
    """
    For each data entry in the list, add time attributes and weather data
    into the 'state' dictionary, if they are configured.
    """
    for entry in data:
        # Each entry has 'timestamp' (as a string) and 'state' (dict)
        timestamp_str = entry['timestamp']
        # Convert the string to a datetime
        timestamp_dt = dateutil.parser.isoparse(timestamp_str)

        # Now add time attributes
        add_time_attributes_to_state(entry['state'], timestamp_dt, config)

        # Add weather data attributes
        add_weather_data_to_state(entry['state'], timestamp_dt, config)

    return data


import numpy as np

# Suppose we have the same enriched_data from your example
# enriched_data[i] = {
#    "timestamp": "...",
#    "state": {"temperatur":0.53, "humidity":48.91, ...},
#    "action": {"heating_on_off":0, "humidifyer_on_of":1}
# }

def flatten_observation(state_dict, obs_keys):
    """
    state_dict = {"temperatur":0.53, "humidity":48.91, ...}
    obs_keys = the list/order of keys to flatten into a vector
    returns e.g. np.array([0.53, 48.91, 0.58, 12.0, ...], dtype=float32)
    """
    return np.array([state_dict[k] for k in obs_keys], dtype=np.float32)



def build_expert_dataset_flat(enriched_data, config):
    """
    Creates a MultiDiscrete dataset for Behavior Cloning with normalized observations.
    Assumes that enriched_data[i]["action"] produced enriched_data[i+1]["state"].
    Therefore, we shift actions by one.
    """
    obs_keys = []
    
    obs_attrs = config["environment"].get("observation_attributes", {})
    obs_keys += list(obs_attrs.keys())

    time_attrs = config["environment"].get("time_attributes", {})
    obs_keys += list(time_attrs.keys())

    weather_select = config["environment"].get("weather_attributes", {}).get("select", {})
    obs_keys += list(weather_select.keys())

    action_attrs = config["environment"].get("action_attributes", {})

    X = []
    y = []
    
    # Iterate until the second last entry to allow shifting action from the next entry
    for i in range(len(enriched_data) - 1):
        # Flatten and normalize observation from the current state's "state"
        obs = []
        for key in obs_keys:
            value = enriched_data[i]["state"].get(key, 0.0)
            if key in obs_attrs:
                low = obs_attrs[key]["low"]
                high = obs_attrs[key]["high"]
                normalized_value = 2 * (value - low) / (high - low) - 1
                obs.append(normalized_value)
            else:
                obs.append(value)
        X.append(obs)
        
        # Use the action from the next entry as the label
        action_values = [enriched_data[i + 1]["action"][key] for key in action_attrs.keys()]
        y.append(action_values)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y

def build_expert_data_with_rewards(enriched_data, config):
    """
    Calculate and add rewards for each state using calculate_reward(), and then normalize
    the observation attributes in the state in the same way as build_expert_dataset_flat.
    """
    obs_attrs = config["environment"].get("observation_attributes", {})
    for entry in enriched_data:
        # Compute the reward using the original (unnormalized) state values
        reward = calculate_reward(entry["state"], config)
        entry["reward"] = reward
        
        # Normalize only the observation attributes (if defined) in the state
        for key, params in obs_attrs.items():
            if key in entry["state"]:
                low = params["low"]
                high = params["high"]
                entry["state"][key] = 2 * (entry["state"][key] - low) / (high - low) - 1
    return enriched_data

def normalize_data(data, config):
    """
    Normalize the observation attributes in the data using the low/high bounds in config.
    """
    obs_attrs = config["environment"].get("observation_attributes", {})
    for entry in data:
        for key, params in obs_attrs.items():
            if key in entry["state"]:
                low = params["low"]
                high = params["high"]
                entry["state"][key] = 2 * (entry["state"][key] - low) / (high - low) - 1
    return data
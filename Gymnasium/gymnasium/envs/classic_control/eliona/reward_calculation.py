from gymnasium.envs.classic_control.eliona.helper_fuctions import datetime_to_time_attributes
import numpy as np


def is_within_timerange(timerange, datetime_attributes):

    if not timerange:
        return True  # No timerange means always active
    time_attr = timerange["time"]
    time_ranges = timerange["range"]
    current_time = datetime_attributes[time_attr]

    # Check if the current time is within any of the specified ranges
    for time_range in time_ranges:
        if time_range[0] <= current_time <= time_range[1]:
            return True
    return False


def range_reward(state, attribute, target_range, reward_factor):
    value = state[attribute]
    reward = (
        10
        if target_range[0] <= value <= target_range[1]
        else -(abs(value - np.mean(target_range)) )
    )
    return reward * reward_factor


def minimize_reward(state, attribute, reward_factor):
    value = state[attribute]
    return -value * reward_factor


def maximize_reward(state, attribute, reward_factor):
    value = state[attribute]
    return value * reward_factor


def exact_reward(state, attribute, target_value, reward_factor):
    value = state[attribute]
    return -((abs(value - target_value) * reward_factor) ** 2)


def calculate_reward(state, config):
    reward_function_attributes = config["reward_function"]["attributes"]
    # Compute raw rewards for each attribute
    if "time" in state:
        datetime_attributes = datetime_to_time_attributes(state["time"])
    datetime_attributes = datetime_to_time_attributes()
    raw_rewards = calculate_reward1(state, config,datetime_attributes)
    # Compute per-attribute best (max) and worst (min) raw rewards
    max_rewards, min_rewards = calculate_max_min_rewards(config)
    
    scaled_rewards = []
    for raw, max_val, min_val in zip(raw_rewards, max_rewards, min_rewards):
        if max_val != min_val:
            scaled = 2 * (raw - min_val) / (max_val - min_val) - 1
        else:
            scaled = 0  # Avoid division by zero if best and worst values are equal.
        scaled_rewards.append(scaled)

    reward_function_attributes = config["reward_function"]["attributes"]
    
    weighted_sum = 0.0
    total_weight = 0.0
    for scaled, attr in zip(scaled_rewards, reward_function_attributes):
        weight = attr.get("reward_factor", 1)
        weighted_sum += scaled * weight
        total_weight += weight
    
    # The worst possible weighted sum is -total_weight,
    # and the best is total_weight. A linear mapping of weighted_sum:
    # scaled_total = 2 * (weighted_sum - (-total_weight)) / (2 * total_weight) - 1
    # which simplifies to:
    scaled_total = weighted_sum / total_weight if total_weight != 0 else 0
    return scaled_total
   

def calculate_reward1(state, config, datetime_attributes=None):
    reward_function_attributes = config["reward_function"]["attributes"]
    individual_rewards = []
    for attr in reward_function_attributes:
        name = attr["name"]
        reward_type = attr["type"]
        reward_factor = attr.get("reward_factor", 1)
        timerange = attr.get("timerange")

        # Check if the current time is within the specified time range
        if timerange and datetime_attributes:
            if not is_within_timerange(timerange, datetime_attributes):
                individual_rewards.append(0)  # Ignore reward if not within time range
             
                continue

        if reward_type == "range":
            r = range_reward(state, name, attr["target_range"], reward_factor)
        elif reward_type == "minimize":
            r = minimize_reward(state, name, reward_factor)
        elif reward_type == "maximize":
            r = maximize_reward(state, name, reward_factor)
        elif reward_type == "exact":
            r = exact_reward(state, name, attr["target_value"], reward_factor)
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")
        individual_rewards.append(r)
    return individual_rewards

def calculate_max_min_rewards(config):
    reward_function_attributes = config["reward_function"]["attributes"]
    observation_attributes = config["environment"]["observation_attributes"]

    max_state = {}
    min_state = {}

    for attr in reward_function_attributes:
        name = attr["name"]
        reward_type = attr["type"]
        target_range = attr.get("target_range")
        target_value = attr.get("target_value")
        obs_limits = observation_attributes[name]

        if reward_type == "range":
            # “Worst-case” (min_state) might be far from target_range center
            center = (target_range[0] + target_range[1]) / 2
            # Pick whichever limit is farther from the center to reflect largest penalty
            distance_low = abs(obs_limits["low"] - center)
            distance_high = abs(obs_limits["high"] - center)
            if distance_low > distance_high:
                min_state[name] = obs_limits["low"]
            else:
                min_state[name] = obs_limits["high"]
            max_state[name] = center

        elif reward_type == "minimize":
            max_state[name] = obs_limits["low"]
            min_state[name] = obs_limits["high"]
        elif reward_type == "maximize":
            max_state[name] = obs_limits["high"]
            min_state[name] = obs_limits["low"]
        elif reward_type == "exact":
            max_state[name] = target_value
            # For a squared penalty, pick whichever limit is farthest from target_value
            if abs(obs_limits["low"] - target_value) > abs(obs_limits["high"] - target_value):
                min_state[name] = obs_limits["low"]
            else:
                min_state[name] = obs_limits["high"]

    max_reward = calculate_reward1(max_state, config)
    min_reward = calculate_reward1(min_state, config)

    return max_reward, min_reward

def undo_normalization(state, config):
    observation_attributes = config["environment"]["observation_attributes"]
    original_state = {}
    for key, value in state.items():
        low = observation_attributes[key]["low"]
        high = observation_attributes[key]["high"]
        original_state[key] = value * (high - low) / 2 + (high + low) / 2
    return original_state

def calculate_normalized_reward(normalized_state, config):
    # Convert normalized_state (numpy array) to dictionary
    state_keys = list(config["environment"]["observation_attributes"].keys())
    state_dict = {key: normalized_state[i] for i, key in enumerate(state_keys)}
    
    # Undo normalization
    original_state = undo_normalization(state_dict, config)

    
    # Calculate reward
    reward = calculate_reward(original_state, config)
    
    return reward
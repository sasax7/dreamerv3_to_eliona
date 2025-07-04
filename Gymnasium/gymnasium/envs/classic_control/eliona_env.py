import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime, timezone


def datetime_to_time_attributes(dt=None, country="CH"):
    """
    Convert a datetime object into a dictionary of time attributes,
    including holiday information (using the `holidays` library).
    """
    import holidays

    if dt is None:
        dt = datetime.now()
    if not isinstance(dt, datetime):
        raise ValueError("Input must be a datetime object")

    holiday_calendar = holidays.country_holidays(country)

    time_attributes = {
        "time_of_day": dt.hour + dt.minute / 60 + dt.second / 3600,  # Fractional hours
        "day_of_week": dt.weekday(),
        "day_of_month": dt.day,
        "month_of_year": dt.month,
        "week_of_year": dt.isocalendar()[1],
        "year": dt.year,
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "is_holiday": 1 if dt in holiday_calendar else 0,
        "minute_of_hour": dt.minute,
    }
    return time_attributes


def create_observation_space(observation_attributes, time_attributes, weather_attributes):
    """
    Create a single flattened observation space as a `Box` with normalized bounds [-1, 1],
    matching the format used during PPO training.
    """
    num_obs = 0
    num_obs += len(observation_attributes)
    num_obs += len(weather_attributes)
    num_obs += len(time_attributes)
    return spaces.Box(low=-1.0, high=1.0, shape=(num_obs,), dtype=np.float32)


def create_action_space(action_attributes):
    """
    Always produce a single Box space in the range [-1, 1] with shape (num_attributes,).
    """
    num_actions = len(action_attributes)
    return spaces.Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32)


def custom_reward_function(state: dict, previous_state: dict, config: dict) -> float:
    reward = 0.0

    # Step 1: Get current time (used for optional timeranges)
    current_time = state.get("time_of_day", state.get("time", datetime.now()).hour)

    for rule in config["reward_function"]["attributes"]:
        name = rule["name"]
        rtype = rule["type"]
        factor = rule.get("reward_factor", 1.0)
        value = state.get(name, None)
        previous_value = previous_state.get(name, None)

        if value is None:
            continue

        # Step 2: Check timerange (optional)
        timerange_config = rule.get("timerange", {})
        allowed_ranges = timerange_config.get("range", [])
        in_timerange = (
            True if not allowed_ranges
            else any(start <= current_time < end for start, end in allowed_ranges)
        )
        if not in_timerange:
            continue

        # Step 3: Dynamic range type
        if rtype == "dynamic_range":
            ref_attr = rule.get("depends_on")
            ref_val = state.get(ref_attr, None)
            base_value = rule.get("base_value", 20)
            scale_above = rule.get("scale_above", 0.0)
            scale_below = rule.get("scale_below", 0.9)

            if ref_val is not None:
                delta = ref_val - base_value
                adjustment = scale_above * delta if delta > 0 else -scale_below * delta
                dynamic_target = base_value + adjustment
                gap = abs(value - dynamic_target)
                reward -= factor * gap
            continue

        # Step 4: Static range
        if rtype == "range":
            target_low, target_high = rule["target_range"]
            if target_low <= value <= target_high:
                reward += factor * 1.0
            else:
                gap = min(abs(value - target_low), abs(value - target_high))
                reward -= factor * gap

        # Step 5: Minimize
        elif rtype == "minimize":
            reward -= factor * value

        # Other types can be added here

    return reward



class ElionaEnvironment(gym.Env):
    """
    A Gymnasium environment that steps through a pre-fetched “enriched” Eliona dataset
    rather than simulating everything.  Observations are exposed as a Dict with a single
    "obs" key (a flat Box), and .sample() will return the dataset’s “action” at the current index.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, config, data: list, render_mode=None):
     
        if len(data) == 0:
            raise ValueError("ElionaEnvironment: trajectory is empty – cannot reset.")
        super().__init__()
        self.render_mode = render_mode
        self.config = config

        # 1) Store the data trajectory and initialize index pointer
        self.trajectory = data
        self._idx = 0

        # 2) Build obs_keys in the same order as in `create_observation_space`
        self.obs_keys = list(config["environment"]["observation_attributes"].keys())
        time_attrs = config["environment"].get("time_attributes", {})
        if time_attrs:
            self.obs_keys.extend(time_attrs.keys())
        weather_select = config["environment"].get("weather_attributes", {}).get("select", {})
        if weather_select:
            self.obs_keys.extend(weather_select.keys())

        # 3) Create the base Box for the flattened vector
        base_box = create_observation_space(
            config["environment"]["observation_attributes"],
            config["environment"].get("time_attributes", {}),
            config["environment"].get("weather_attributes", {}).get("select", {}),
        )

        # 4) Wrap that Box into a Dict under key "obs"
        self.observation_space = spaces.Dict({"obs": base_box})
        self.action_space = create_action_space(config["environment"]["action_attributes"])
        self.action_keys  = list(config["environment"]["action_attributes"])
        # 5) Placeholders for state, previous_state, reward, lastaction
        self.state = {}
        self.previous_state = {}
        self.reward = 0.0
        self.lastaction = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset index to the first row of the trajectory
        self._idx = 0
        first_row = self.trajectory[self._idx]

        # Parse timestamp and copy state dict
        ts = datetime.fromisoformat(first_row["timestamp"])
        state_values = first_row["state"].copy()

        # Merge in time attributes based on the timestamp
        time_attrs = datetime_to_time_attributes(ts)
        state_values.update(time_attrs)
        state_values["time"] = ts  # Keep the raw datetime

        # Initialize both state and previous_state to this first row
        self.state = state_values.copy()
        self.previous_state = state_values.copy()

        # No action yet
        self.lastaction = None
        self.reward = 0.0

        # Build and return the initial observation as a dict under "obs"
        raw_array = self._dict_to_obs(self.state)
        return {"obs": raw_array}, {}

    def step(self, action):
        # 1) Record the agent’s last action (no effect on next state—historical playback)
        if isinstance(action, np.ndarray):
            self.lastaction = float(action[0])
        else:
            self.lastaction = float(action)

        # 2) Advance index; if we've exhausted the trajectory, terminate (truncated)
        self._idx += 1
        if self._idx >= len(self.trajectory):
            return {"obs": self._dict_to_obs(self.state)}, 0.0, False, True, {}

        # 3) Load the next row from the trajectory
        row = self.trajectory[self._idx]
        ts = datetime.fromisoformat(row["timestamp"])
        raw_state = row["state"].copy()

        # 4) Compute time attributes and merge
        time_attrs = datetime_to_time_attributes(ts)
        raw_state.update(time_attrs)
        raw_state["time"] = ts

        # 5) Save old state for reward computation
        self.previous_state = self.state.copy()
        # 6) Overwrite current state
        self.state = raw_state

        # 7) Compute reward
        reward = custom_reward_function(self.state, self.previous_state, self.config)
        self.reward = reward

        # 8) Check out-of-bounds (if any observation is outside its configured [low, high])
        terminated = False
        for attr, params in self.config["environment"]["observation_attributes"].items():
            val = self.state.get(attr, None)
            if (val is None) or not (params["low"] <= val <= params["high"]):
                terminated = True
                break

        # 9) Build the new observation array and return as dict under "obs"
        raw_array = self._dict_to_obs(self.state)
        return {"obs": raw_array}, reward, terminated, False, {}

    def sample(self):
        """
        Return the “true” action (setpoint_vl) associated with the current index
        in self.trajectory, but normalized into [-1, +1] to match self.action_space.
        """
        print("getting sample")
        # Grab the raw setpoint (0–100) from the trajectory at index self._idx

        # Find the (first) action attribute name from the config
        action_key = next(iter(self.config["environment"]["action_attributes"]))
        
        raw_val = self.trajectory[self._idx]["action"][action_key]
        
        lo  = self.config["environment"]["action_attributes"][action_key]["low"]
        hi  = self.config["environment"]["action_attributes"][action_key]["high"]
        normalized = 2.0 * (raw_val - lo) / (hi - lo) - 1.0
        # Return as a numpy array of shape (1,), dtype float32, since action_space is Box((1,),)
        return np.array([normalized], dtype=np.float32)

    def _dict_to_obs(self, state_dict):
        """
        Normalize each key in `obs_keys`:
         - If key is in observation_attributes, normalize from [low..high] → [-1..1].
         - If key is "time_of_day", normalize [0..24] → [-1..1].
         - If key is "minute_of_hour", normalize [0..60] → [-1..1].
         - Otherwise (e.g. weather), append raw float value.
        """
        obs = []
        for key in self.obs_keys:
            value = state_dict.get(key, 0.0)

            if key in self.config["environment"]["observation_attributes"]:
                low = self.config["environment"]["observation_attributes"][key]["low"]
                high = self.config["environment"]["observation_attributes"][key]["high"]
                normalized = 2.0 * (value - low) / (high - low) - 1.0
                obs.append(normalized)

            elif key == "time_of_day":
                low, high = 0.0, 24.0
                v_clamped = max(min(value, high), low)
                normalized = 2.0 * (v_clamped - low) / (high - low) - 1.0
                obs.append(normalized)

            elif key == "minute_of_hour":
                low, high = 0.0, 60.0
                v_clamped = max(min(value, high), low)
                normalized = 2.0 * (v_clamped - low) / (high - low) - 1.0
                obs.append(normalized)

            else:
                # e.g. weather attributes or anything else
                obs.append(value)

        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        # Simple text-based render
        print(
            f"state={self.state} | "
            f"reward={self.reward:>8.5f} | "
            f"lastaction={self.lastaction}"
        )
        if mode in ["human", "rgb_array"]:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

            fig, ax = plt.subplots(figsize=(6, 4))
            
            categories = ["NA_TA", "NA_TV", "KO_IST target", "Agent action"]
            values = [
                self.state.get("NA_TA", 0),
                self.state.get("NA_TV", 0),
                self.state.get("KO_IST", 0),  # or whatever “target” column you logged
                self.lastaction if self.lastaction is not None else 0,
            ]
            ax.bar(categories, values, color=["blue", "orange", "green", "red"])
            ax.set_ylabel("Value")
            ax.set_title("Current Temps and Last Action")
            ax.set_ylim(-5, 100)

            canvas = FigureCanvas(fig)
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8").reshape(int(height), int(width), 4)
            plt.close(fig)
            return image
        else:
            return super().render(mode=mode)

    def close(self):
        print("Closing ElionaEnvironment.")

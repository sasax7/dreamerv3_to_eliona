"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from gymnasium.envs.registration import (
    load_env_plugins,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)


# Classic
# ----------------------------------------

register(
    id="CartPole-v0",
    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)
from datetime import datetime, timezone
from gymnasium.envs.classic_control.eliona.eliona_data import get_eliona_trend_data
from gymnasium.envs.classic_control.eliona.prepare_data import enrich_data_with_time_and_weather


static = {
    "start_date": "2025-01-01",
    "end_date": "2025-07-03",
    "environment": [
        {"asset_id": 4271, 
         "observation_attributes": {"NA_TA": {"low": -20, "high": 50}},
         "action_attributes":      {"KO_IST": {"type": "continuous","low":0,"high":100}},
        },
        {"asset_id": 4274,
         "observation_attributes": {"NA_TV": {"low": 0, "high": 80}},
         # no action for this asset
        },
      ],
    "reward_function": {
        "attributes": [
            {
                "name": "KO_IST",
                "type": "range",
                "target_range": [0, 50],
                "reward_factor": 5
            },
            {
                "name": "NA_TV",
                "type": "dynamic_range",
                "depends_on": "NA_TA",
                "base_value": 20,
                "scale_above": 0.0,
                "scale_below": 0.9,
                "reward_factor": 5
            }
        ]
    },
}
static["start_date"] = datetime.fromisoformat(static["start_date"]).replace(tzinfo=timezone.utc)
static["end_date"] = datetime.fromisoformat(static["end_date"]).replace(tzinfo=timezone.utc)
def merge_env_blocks(env_blocks: list):
    """Fasst alle observation_ / action_attributes der Asset-Blöcke zusammen."""
    obs_attrs, act_attrs = {}, {}
    for block in env_blocks:
        obs_attrs.update(block.get("observation_attributes", {}))
        act_attrs.update(block.get("action_attributes", {}))
    return obs_attrs, act_attrs

# ---- vor dem register() aufrufen -------------------------------
obs_attrs, act_attrs = merge_env_blocks(static["environment"])

env_cfg = {
    "observation_attributes": obs_attrs,
    "action_attributes": act_attrs,
    "time_attributes": {},          # falls du welche hast
    "weather_attributes": {},       # dito
}
raw_data = get_eliona_trend_data(
    static["start_date"],
    static["end_date"],
    static["environment"]
)
print("First action dict  ➜", raw_data[0]["action"])
print("First state dict   ➜", raw_data[0]["state"])
print("First data    ➜", raw_data[0])
print("second data    ➜", raw_data[1])
print("data    ➜", raw_data[1100])
print("➜ raw", len(raw_data))

def merge_env_blocks(env_blocks: list):
    """Fasst alle observation_ / action_attributes der Asset-Blöcke zusammen."""
    obs_attrs, act_attrs = {}, {}
    for block in env_blocks:
        obs_attrs.update(block.get("observation_attributes", {}))
        act_attrs.update(block.get("action_attributes", {}))
    return obs_attrs, act_attrs

# ---- vor dem register() aufrufen -------------------------------
obs_attrs, act_attrs = merge_env_blocks(static["environment"])

env_cfg = {
    "observation_attributes": obs_attrs,
    "action_attributes": act_attrs,
    "time_attributes": {},          # falls du welche hast
    "weather_attributes": {},       # dito
}
enriched = enrich_data_with_time_and_weather(raw_data, {"environment": env_cfg})
print("➜ enriched", len(enriched))


register(
    id="ElionaEnv",
    entry_point="gymnasium.envs.classic_control.eliona_env:ElionaEnvironment",
    kwargs={
        "config": {**static, "environment": env_cfg},   # !!! jetzt Dict statt Liste
        "data": enriched,
    },
    max_episode_steps=1000,
)
register(
    id="CartPole-v1",
    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="MountainCar-v0",
    entry_point="gymnasium.envs.classic_control.mountain_car:MountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="MountainCarContinuous-v0",
    entry_point="gymnasium.envs.classic_control.continuous_mountain_car:Continuous_MountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id="Pendulum-v1",
    entry_point="gymnasium.envs.classic_control.pendulum:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="Acrobot-v1",
    entry_point="gymnasium.envs.classic_control.acrobot:AcrobotEnv",
    reward_threshold=-100.0,
    max_episode_steps=500,
)


# Phys2d (jax classic control)
# ----------------------------------------

register(
    id="CartPoleJax-v0",
    entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="CartPoleJax-v1",
    entry_point="gymnasium.envs.phys2d.cartpole:CartPoleJaxEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id="PendulumJax-v0",
    entry_point="gymnasium.envs.phys2d.pendulum:PendulumJaxEnv",
    max_episode_steps=200,
)

# Box2d
# ----------------------------------------

register(
    id="LunarLander-v2",
    entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuous-v2",
    entry_point="gymnasium.envs.box2d.lunar_lander:LunarLander",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="BipedalWalker-v3",
    entry_point="gymnasium.envs.box2d.bipedal_walker:BipedalWalker",
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id="BipedalWalkerHardcore-v3",
    entry_point="gymnasium.envs.box2d.bipedal_walker:BipedalWalker",
    kwargs={"hardcore": True},
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id="CarRacing-v2",
    entry_point="gymnasium.envs.box2d.car_racing:CarRacing",
    max_episode_steps=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id="Blackjack-v1",
    entry_point="gymnasium.envs.toy_text.blackjack:BlackjackEnv",
    kwargs={"sab": True, "natural": False},
)

register(
    id="FrozenLake-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4"},
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)

register(
    id="FrozenLake8x8-v1",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,  # optimum = 0.91
)

register(
    id="CliffWalking-v0",
    entry_point="gymnasium.envs.toy_text.cliffwalking:CliffWalkingEnv",
)

register(
    id="Taxi-v3",
    entry_point="gymnasium.envs.toy_text.taxi:TaxiEnv",
    reward_threshold=8,  # optimum = 8.46
    max_episode_steps=200,
)

# Mujoco
# ----------------------------------------

# 2D

register(
    id="Reacher-v2",
    entry_point="gymnasium.envs.mujoco:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Reacher-v4",
    entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Pusher-v2",
    entry_point="gymnasium.envs.mujoco:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Pusher-v4",
    entry_point="gymnasium.envs.mujoco.pusher_v4:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="InvertedPendulum-v2",
    entry_point="gymnasium.envs.mujoco:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedPendulum-v4",
    entry_point="gymnasium.envs.mujoco.inverted_pendulum_v4:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id="InvertedDoublePendulum-v2",
    entry_point="gymnasium.envs.mujoco:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="InvertedDoublePendulum-v4",
    entry_point="gymnasium.envs.mujoco.inverted_double_pendulum_v4:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="HalfCheetah-v2",
    entry_point="gymnasium.envs.mujoco:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetah-v3",
    entry_point="gymnasium.envs.mujoco.half_cheetah_v3:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="HalfCheetah-v4",
    entry_point="gymnasium.envs.mujoco.half_cheetah_v4:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="Hopper-v2",
    entry_point="gymnasium.envs.mujoco:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v3",
    entry_point="gymnasium.envs.mujoco.hopper_v3:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Hopper-v4",
    entry_point="gymnasium.envs.mujoco.hopper_v4:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Swimmer-v2",
    entry_point="gymnasium.envs.mujoco:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v3",
    entry_point="gymnasium.envs.mujoco.swimmer_v3:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Swimmer-v4",
    entry_point="gymnasium.envs.mujoco.swimmer_v4:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Walker2d-v2",
    max_episode_steps=1000,
    entry_point="gymnasium.envs.mujoco:Walker2dEnv",
)

register(
    id="Walker2d-v3",
    max_episode_steps=1000,
    entry_point="gymnasium.envs.mujoco.walker2d_v3:Walker2dEnv",
)

register(
    id="Walker2d-v4",
    max_episode_steps=1000,
    entry_point="gymnasium.envs.mujoco.walker2d_v4:Walker2dEnv",
)

register(
    id="Ant-v2",
    entry_point="gymnasium.envs.mujoco:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Ant-v3",
    entry_point="gymnasium.envs.mujoco.ant_v3:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Ant-v4",
    entry_point="gymnasium.envs.mujoco.ant_v4:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Humanoid-v2",
    entry_point="gymnasium.envs.mujoco:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="Humanoid-v3",
    entry_point="gymnasium.envs.mujoco.humanoid_v3:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="Humanoid-v4",
    entry_point="gymnasium.envs.mujoco.humanoid_v4:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v2",
    entry_point="gymnasium.envs.mujoco:HumanoidStandupEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v4",
    entry_point="gymnasium.envs.mujoco.humanoidstandup_v4:HumanoidStandupEnv",
    max_episode_steps=1000,
)


# --- For shimmy compatibility
def _raise_shimmy_error():
    raise ImportError(
        "To use the gym compatibility environments, run `pip install shimmy[gym]`"
    )


# When installed, shimmy will re-register these environments with the correct entry_point
register(id="GymV22Environment-v0", entry_point=_raise_shimmy_error)
register(id="GymV26Environment-v0", entry_point=_raise_shimmy_error)


# Hook to load plugins from entry points
load_env_plugins()

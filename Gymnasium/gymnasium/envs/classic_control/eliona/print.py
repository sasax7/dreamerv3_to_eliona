#!/usr/bin/env python3
from __future__ import annotations
import pprint
from datetime import datetime, timezone

import numpy as np
import torch
from torch.distributions import Independent
from lightning.fabric import Fabric
from omegaconf import OmegaConf
from gymnasium.spaces import Box, Dict as SpaceDict

from eliona_data import get_eliona_trend_data
from prepare_data import enrich_data_with_time_and_weather, build_expert_dataset_flat




# -------------------------------------------------------------------------
# 2) Fetch & enrich Eliona data
# -------------------------------------------------------------------------
static = {
    "asset_id": 1651,
    "start_date": "2025-05-15",
    "end_date": "2025-05-30",
    "environment": {
        "observation_attributes": {
            "outside_temperature": {"low": -15, "high": 50},
            "outside_temperature_forecast": {"low": -15, "high": 50},
            "heatdemand": {"low": 0, "high": 60},
            "momentaner_tarif": {"low": 0, "high": 1},
            "momentane_kosten": {"low": 0, "high": 0.3},
            "indoor_temperature": {"low": -15, "high": 40},
        },
        "action_attributes": {"setpoint_vl": {"type": "continuous", "low": 0, "high": 100}},
    },
    "reward_function": {
        "attributes": [
            {"name": "indoor_temperature", "type": "range", "target_range": [19, 24], "reward_factor": 5},
            {"name": "momentane_kosten", "type": "minimize", "reward_factor": 1},
        ]
    },
}
static["start_date"] = datetime.fromisoformat(static["start_date"]).replace(tzinfo=timezone.utc)
static["end_date"] = datetime.fromisoformat(static["end_date"]).replace(tzinfo=timezone.utc)

raw = get_eliona_trend_data(
    static["asset_id"],
    static["start_date"],
    static["end_date"],
    list(static["environment"]["observation_attributes"]),
    list(static["environment"]["action_attributes"]),
)
enriched = enrich_data_with_time_and_weather(raw, static)

print("Enriched data sample:")
pprint.pprint(enriched[0])

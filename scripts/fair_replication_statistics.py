"""Complete three-seed descriptive statistics; never select or drop a seed."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def summarize_groups(rows, group_keys, metric_keys, expected_seeds=(7, 17, 27)):
    groups = defaultdict(dict)
    expected = set(expected_seeds)
    for row in rows:
        seed = int(row["seed"])
        if seed not in expected:
            raise ValueError(f"Unexpected seed {seed}")
        group = tuple(row[k] for k in group_keys)
        if seed in groups[group]:
            raise ValueError(f"Duplicate seed {seed} in group {group}")
        groups[group][seed] = row
    results = []
    for group, records in sorted(groups.items()):
        for metric in metric_keys:
            values, missing = [], []
            for seed in expected_seeds:
                value = records.get(seed, {}).get(metric)
                if value is None or value == "" or not math.isfinite(float(value)):
                    missing.append(seed)
                else:
                    values.append(float(value))
            complete = not missing
            results.append(
                {
                    **dict(zip(group_keys, group, strict=True)),
                    "metric": metric,
                    "expected_seeds": ",".join(map(str, expected_seeds)),
                    "recorded_seeds": ",".join(map(str, sorted(records))),
                    "n_available": len(values),
                    "all_three_finite": complete,
                    "missing_metric_seeds": ",".join(map(str, missing)),
                    "mean": float(np.mean(values)) if complete else None,
                    "sample_std": float(np.std(values, ddof=1)) if complete else None,
                }
            )
    return results

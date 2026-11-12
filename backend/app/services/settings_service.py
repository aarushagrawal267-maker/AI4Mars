"""User-adjustable presentation and simulation settings, persisted in SQLite."""
from __future__ import annotations

from ml.classes import CLASS_KEYS, DEFAULT_COLORS, hex_to_rgb

from ..database.db import Database
from ..database.models import Setting

# Illustrative defaults only. They are NOT measurements of real rover mobility.
DEFAULT_COSTS: dict[str, dict] = {
    "soil": {"cost": 1.0, "passable": True},
    "bedrock": {"cost": 2.0, "passable": True},
    "sand": {"cost": 5.0, "passable": True},
    "big_rock": {"cost": 20.0, "passable": True},
}
COST_DISCLAIMER = (
    "Movement costs are experimental assumptions chosen for demonstration. They are not validated measurements "
    "of Mars rover mobility. Terrain class alone cannot determine whether a rover can safely drive somewhere: "
    "slope, rock size, wheel sinkage, rover state and mission rules all matter. This simulator is not connected "
    "to any rover."
)


class SettingsService:
    def __init__(self, db: Database):
        self.db = db

    def _get(self, key: str, default: dict) -> dict:
        with self.db.session() as s:
            row = s.get(Setting, key)
            return dict(row.value) if row else dict(default)

    def _set(self, key: str, value: dict) -> None:
        with self.db.session() as s:
            row = s.get(Setting, key)
            if row:
                row.value = value
            else:
                s.add(Setting(key=key, value=value))

    def colors(self) -> dict[str, str]:
        return {**DEFAULT_COLORS, **self._get("colors", {})}

    def set_colors(self, colors: dict[str, str]) -> dict[str, str]:
        clean = {}
        for key in CLASS_KEYS:
            if key in colors:
                hex_to_rgb(colors[key])  # raises ValueError when malformed
                clean[key] = "#" + colors[key].strip().lstrip("#").upper()
        merged = {**self.colors(), **clean}
        self._set("colors", merged)
        return merged

    def reset_colors(self) -> dict[str, str]:
        self._set("colors", dict(DEFAULT_COLORS))
        return dict(DEFAULT_COLORS)

    def costs(self) -> dict[str, dict]:
        stored = self._get("nav_costs", {})
        return {k: {**DEFAULT_COSTS[k], **stored.get(k, {})} for k in CLASS_KEYS}

    def set_costs(self, costs: dict[str, dict]) -> dict[str, dict]:
        merged = self.costs()
        for k in CLASS_KEYS:
            if k in costs:
                merged[k] = {"cost": float(costs[k]["cost"]), "passable": bool(costs[k]["passable"])}
        self._set("nav_costs", merged)
        return merged

    def reset_costs(self) -> dict[str, dict]:
        self._set("nav_costs", {k: dict(v) for k, v in DEFAULT_COSTS.items()})
        return self.costs()

"""
backend/simulation/models.py

Pure data containers, no logic beyond TireSet.degrade(). Designed so the
shape doesn't have to change when Phase 5 wires in simulated (non-
historical) advancement -- it just starts populating fields that pure
replay leaves inert (degradation_rate) or approximate (gap_to_leader).
"""
from dataclasses import dataclass, field


@dataclass
class TireSet:
    """One set of tires currently fitted to a car.

    degradation_rate is unused in Phase 1 -- pure replay reads tire age
    straight from the ground-truth `tire_age` column in `laps` and never
    predicts anything. The field exists now so this dataclass's shape
    doesn't change when Phase 5 wires in Phase 2's tire degradation
    model: simulated advancement will populate and use it inside
    degrade(). Don't be surprised it's always 0.0 right now -- that's
    expected, not a bug.
    """

    compound: str
    age: int
    degradation_rate: float = 0.0

    def degrade(self) -> None:
        """Age this tire set by one lap. Never called in replay mode --
        age comes straight from the DB. Exists for Phase 5's simulated
        advancement, where there's no DB row to read tire_age from."""
        self.age += 1


@dataclass
class Driver:
    """One driver's state at a single point in the race (one lap)."""

    code: str                      # 3-letter FastF1 code, e.g. "VER"
    team: str
    position: int                  # on-track running position; 99 = unclassified/retired
    gap_to_leader: float           # seconds, APPROXIMATE -- see engine.py docstring for why
    current_tire: TireSet
    stint_history: list[TireSet] = field(default_factory=list)  # completed stints, oldest first
    lap_times: list[float] = field(default_factory=list)        # seconds, one per completed lap so far


@dataclass
class RaceState:
    """A single immutable snapshot of the whole field at one lap.

    SimulationEngine.load_historical() builds the full list of these up
    front; nothing downstream mutates a RaceState after it's created.
    That's what makes advance_lap_historical() a pure index lookup, and
    what will make Phase 5's side-by-side real-vs-simulated comparison
    straightforward -- you're indexing into two lists of snapshots, never
    asking "what does this object currently look like."
    """

    race_id: int
    lap: int
    drivers: dict[str, Driver]     # keyed by driver code, e.g. "VER"
    safety_car: bool
    vsc: bool
    weather: str                   # "dry" or "wet" -- coarse, derived from FastF1's Rainfall flag
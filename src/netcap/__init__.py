"""netcap: accelerator-time capacity accounting for large-scale AI training.

The package answers one question: for a given cluster and workload, which
marginal infrastructure investment produces the most productive accelerator
time, and under what conditions does that answer change.

Modules:

* :mod:`netcap.config`       parameter schema, YAML round-trip
* :mod:`netcap.collectives`  alpha-beta hierarchical collective timing
* :mod:`netcap.performance`  healthy-step compute and communication model
* :mod:`netcap.reliability`  failure, detection, recovery, checkpoint model
* :mod:`netcap.accounting`   four-fate accelerator-time ledger and invariants
* :mod:`netcap.metrics`      substitution metric, attribution, decision rules
"""

__version__ = "0.1.0"

from .config import (  # noqa: F401
    AcceleratorSpec,
    KernelSpec,
    ModelSpec,
    ParallelismSpec,
    ReliabilitySpec,
    ScenarioConfig,
    TopologySpec,
    load_scenario,
    replace_nested,
    save_scenario,
)
from .accounting import CapacityLedger, build_ledger, mfu  # noqa: F401
from .metrics import (  # noqa: F401
    Intervention,
    break_even_cost,
    rank_interventions,
    scaling_curve,
    shapley_attribution,
    substitution_equivalent_accelerators,
)

__all__ = [
    "AcceleratorSpec",
    "CapacityLedger",
    "Intervention",
    "KernelSpec",
    "ModelSpec",
    "ParallelismSpec",
    "ReliabilitySpec",
    "ScenarioConfig",
    "TopologySpec",
    "break_even_cost",
    "build_ledger",
    "load_scenario",
    "mfu",
    "rank_interventions",
    "replace_nested",
    "save_scenario",
    "scaling_curve",
    "shapley_attribution",
    "substitution_equivalent_accelerators",
    "__version__",
]

"""Installed, working-directory-independent entry point for scenario accounting."""
import argparse
import json
import sys
from .config import load_scenario
from .accounting import build_ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scenario', help='Path to a scenario YAML file')
    parser.add_argument('--fidelity', choices=('analytical', 'monte_carlo'), default='analytical')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--replicates', type=int, default=200)
    args = parser.parse_args(argv)
    try:
        if args.replicates <= 0:
            raise ValueError('replicates must be positive')
        ledger = build_ledger(load_scenario(args.scenario), args.fidelity, args.seed, args.replicates)
        print(json.dumps({'evidence_class': 'simulated' if args.fidelity == 'monte_carlo' else 'derived',
                          'scenario': ledger.scenario_name,
                          'useful_capacity_fraction': ledger.useful_capacity_fraction,
                          'productive_accelerator_seconds': ledger.productive,
                          'nominal_accelerator_seconds': ledger.total,
                          'fidelity': args.fidelity, 'seed': args.seed,
                          'assumption': 'Model output, not measured fleet performance'}, allow_nan=False, indent=2))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f'netcap: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

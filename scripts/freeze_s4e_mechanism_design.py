"""Build reviewed S4E3 offline data/asset locks. Does not authorize inference."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s4e_mechanism_design import build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('collection', 'low-policy', 'core-lock', 'datastore', 'external-manifest',
                 'external-csv', 'raw-csv', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(build(**vars(args)), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

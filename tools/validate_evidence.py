"""Validate portable experiment evidence manifests without importing repo code."""
import argparse
import hashlib
import json
import math
from pathlib import Path

CLASSES = {'measured','external-observed','calibrated','simulated','derived','scenario','sanity-bound'}
SOURCE_TYPES = {'primary-publication','vendor-specification','public-dataset','operator-measurement',
                'model-assumption','analytical-identity','experiment-output'}


def validate(document, root):
    root = Path(root)
    errors = []
    if document.get('schema_version') != 'dimaggi-evidence/v1': errors.append('unsupported schema_version')
    sources = document.get('sources', {})
    for name, source in sources.items():
        if source.get('source_type') not in SOURCE_TYPES: errors.append(f'{name}: invalid source_type')
        for field in ('location','version_date','license_or_reuse','scope'):
            if not source.get(field): errors.append(f'{name}: missing {field}')
        location=source.get('location','')
        if location and not location.startswith(('https://','http://')):
            local=(root/location.split('#')[0]).resolve()
            if not local.is_relative_to(root.resolve()) or not local.is_file():
                errors.append(f'{name}: missing/unsafe local source')
    for section in ('inputs','results'):
        if not isinstance(document.get(section), list):
            errors.append(f'missing {section}'); continue
        for row in document[section]:
            name = row.get('name','unnamed')
            for field in ('name','unit','transformation','uncertainty','affected_outputs'):
                if not row.get(field): errors.append(f'{name}: missing {field}')
            if row.get('evidence_class') not in CLASSES: errors.append(f'{name}: invalid evidence_class')
            refs = row.get('sources', [])
            if not refs or any(ref not in sources for ref in refs): errors.append(f'{name}: unresolved source')
            if 'statement' in row:
                if not isinstance(row['statement'],str) or not row['statement'].strip():
                    errors.append(f'{name}: empty statement')
                if 'value' in row or 'range' in row: errors.append(f'{name}: ambiguous statement/numeric result')
                continue
            values = row.get('range', [row.get('value')])
            if not isinstance(values,list) or not values or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
                errors.append(f'{name}: values must be finite numbers'); continue
            if 'range' in row and (len(values)!=2 or values[0]>values[1]): errors.append(f'{name}: invalid range')
            bounds = row.get('valid_bounds')
            if (not isinstance(bounds,list) or len(bounds)!=2
                    or any(type(x) not in (int,float) or not math.isfinite(x) for x in bounds)
                    or bounds[0]>bounds[1]):
                errors.append(f'{name}: finite ordered valid_bounds required')
            elif any(x<bounds[0] or x>bounds[1] for x in values): errors.append(f'{name}: outside valid bounds')
            if 'range' in row and not row.get('range_basis'): errors.append(f'{name}: unsupported range')
    for artifact in document.get('artifacts', []):
        path = (root / artifact['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f'missing/unsafe artifact: {artifact["path"]}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
            errors.append(f'changed artifact: {artifact["path"]}')
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    errors = validate(json.loads(args.manifest.read_text()), args.root or args.manifest.parent)
    print(json.dumps({'passed':not errors,'errors':errors},indent=2))
    raise SystemExit(bool(errors))

#!/usr/bin/env python3
"""Read-only audit of the 35 unique M1101 FoldX BuildModel failures.

This script does not run FoldX, modify PDB/TXT files, or overwrite results.
It checks whether the expected mutation-list TXT, repaired WT PDB, and any
previously generated mutant PDBs are present for each failed task.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Optional

DATA_CSV = Path('/data/data_split/raw/M1101.csv')
WT_DIR = Path('/data/data_split/splits/M1101/wt')
TXT_DIR = Path('/data/external/M1101/pdb/txt')
DEAL_DIR = Path('/data/data_split/splits/M1101/deal')

# Unique failed task IDs derived from the uploaded M1101_foldx_ddg.csv.
# The CSV contains 36 failed rows but 35 unique IDs because 1N8Z_YH100AF
# occurs twice in the source records.
FAILED_IDS = [
    '1N8Z_HL91A,YH100AA',
    '1N8Z_NL30S,HL91F,YL92W,TL94S,DH98W,YH100AF,YH102V',
    '1N8Z_WH95A,YH100AA',
    '1N8Z_YH100AA',
    '1N8Z_YH100AF',
    '2NY7_PH97E,YH98W,SH99G,DH100AD,DH100BQ,SH100CR,PH100DR,QH100EY',
    '2NY7_PH97E,YH98W,SH99G,DH100AD,DH100BQ,SH100CV,PH100DR,QH100EY',
    '2NY7_PH97E,YH98W,SH99G,DH100AE,DH100BM,SH100CF,PH100DR,QH100EY',
    '2NY7_PH97E,YH98W,SH99G,DH100AE,DH100BM,SH100CR,PH100DR,QH100EF',
    '2NY7_PH97E,YH98W,SH99G,DH100AE,DH100BQ,SH100CF,PH100DR,QH100EF',
    '2NY7_PH97E,YH98W,SH99G,DH100AH,DH100BQ,SH100CR,PH100DR,QH100EY',
    '2NY7_PH97E,YH98W,SH99G,DH100AT,DH100BQ,SH100CR,PH100DR,QH100EF',
    '2NY7_SL28K,IL28AE,RL29F,SL30G',
    '2NY7_SL28P,IL28AL,RL29H,SL30R,RL31A',
    '2NY7_SL28Q,IL28AL,RL29D,SL30G,RL31S',
    '2NY7_SL28T,IL28AV,RL29Y,SL30R,RL31D',
    '3BDY_DL27AA',
    '3BDY_IL27BA',
    '3BDY_PL27CA',
    '3BDY_RL27DA',
    '3BDY_YH100AA',
    '3BE1_DL27AA',
    '3BE1_IL27BA',
    '3BE1_PL27CA',
    '3BE1_RL27DA',
    '3BE1_YH100AA',
    '3BN9_DA60AA',
    '3BN9_DA60BA',
    '3BN9_FA60EA',
    '3BN9_QA221AA',
    '3BN9_QH100AV',
    '3BN9_RA60CA',
    '3BN9_RA60FA',
    '3BN9_YA60GA',
    'HM_3BN9_QH100AV',
]


def norm_text(value: str) -> str:
    return ''.join(str(value or '').replace('\ufeff', '').split())


def find_case_insensitive(directory: Path, filename: str) -> Optional[Path]:
    direct = directory / filename
    if direct.is_file():
        return direct
    if not directory.is_dir():
        return None
    target = filename.casefold()
    for p in directory.iterdir():
        if p.is_file() and p.name.casefold() == target:
            return p
    return None


def read_source_rows() -> Dict[str, Dict[str, str]]:
    if not DATA_CSV.is_file():
        raise FileNotFoundError(f'M1101 source CSV not found: {DATA_CSV}')
    with DATA_CSV.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'PDB', 'ID'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'M1101.csv is missing columns: {sorted(missing)}')
        rows: Dict[str, Dict[str, str]] = {}
        conflicts = []
        for row in reader:
            task_id = (row.get('ID') or '').strip()
            if not task_id or task_id not in FAILED_IDS:
                continue
            record = {
                'PDB': (row.get('PDB') or '').strip(),
                'ID': task_id,
                'Mutation': (row.get('Mutation') or '').strip(),
            }
            if task_id in rows:
                old = rows[task_id]
                if (old['PDB'], old['Mutation']) != (record['PDB'], record['Mutation']):
                    conflicts.append(task_id)
            else:
                rows[task_id] = record
        if conflicts:
            raise ValueError('Conflicting PDB/Mutation mappings in M1101.csv: ' + ', '.join(sorted(set(conflicts))))
        return rows


def main() -> int:
    source = read_source_rows()
    print('M1101 FoldX recovery input audit (READ ONLY)')
    print(f'Source CSV : {DATA_CSV}')
    print(f'WT folder  : {WT_DIR}')
    print(f'TXT folder : {TXT_DIR}')
    print(f'Deal folder: {DEAL_DIR}')
    print(f'Failed unique IDs from FoldX result CSV: {len(FAILED_IDS)}')
    print(f'IDs found in M1101.csv: {len(source)}')
    print()
    print('PDB\tTXT\tTXT_CONTENT\tWT_PDB\tEXISTING_MUTANT_OUTPUTS\tID')

    counts = {'txt_ok': 0, 'txt_missing': 0, 'txt_mismatch': 0,
              'wt_ok': 0, 'wt_missing': 0, 'out_found': 0, 'source_missing': 0}

    for task_id in FAILED_IDS:
        row = source.get(task_id)
        if row is None:
            counts['source_missing'] += 1
            print(f'NA\tMISSING\tNA\tMISSING\t0\t{task_id}')
            continue

        pdb_id = row['PDB']
        mutation = row['Mutation'] or (task_id[len(pdb_id) + 1:] if task_id.startswith(pdb_id + '_') else '')
        expected_txt = f'individual_list_{task_id}.txt'
        txt_path = find_case_insensitive(TXT_DIR, expected_txt)
        wt_path = find_case_insensitive(WT_DIR, f'{pdb_id}.pdb')

        if txt_path is None:
            txt_status = 'MISSING'
            txt_content = 'NA'
            counts['txt_missing'] += 1
        else:
            content = txt_path.read_text(encoding='utf-8-sig', errors='replace')
            compact = norm_text(content).rstrip(';')
            expected = norm_text(mutation).rstrip(';')
            txt_content = content.strip().replace('\t', ' ')[:100]
            if expected and compact == expected:
                txt_status = 'MATCH'
                counts['txt_ok'] += 1
            else:
                txt_status = 'MISMATCH' if expected else 'NO_SOURCE_MUTATION'
                counts['txt_mismatch'] += 1
                txt_content = f'expected={mutation}; | actual={txt_content}'

        if wt_path is None:
            wt_status = 'MISSING'
            counts['wt_missing'] += 1
        else:
            wt_status = wt_path.name
            counts['wt_ok'] += 1

        # Check likely outputs without changing anything. The exact layout can vary.
        outputs = []
        if DEAL_DIR.is_dir():
            direct = DEAL_DIR / f'{task_id}.pdb'
            if direct.is_file():
                outputs.append(str(direct))
            task_dir = DEAL_DIR / task_id
            if task_dir.is_dir():
                outputs.extend(str(p) for p in sorted(task_dir.glob('*.pdb')))
            outputs.extend(str(p) for p in sorted(DEAL_DIR.glob(f'{pdb_id}_1_*.pdb')) if p.is_file())
        if outputs:
            counts['out_found'] += 1
        output_status = str(len(set(outputs)))
        print(f'{pdb_id}\t{txt_status}\t{txt_content}\t{wt_status}\t{output_status}\t{task_id}')

    print('\nSUMMARY')
    for key, value in counts.items():
        print(f'{key}={value}')
    print('\nNo FoldX command was executed and no input/output file was modified.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)

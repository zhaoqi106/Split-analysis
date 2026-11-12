#!/usr/bin/env python3
"""Recompute only the 35 previously missing M1101 FoldX tasks.

Default mode performs a full preflight only. Add --execute to run FoldX 5.1.
The script uses the audited renumbered WT PDB and mapped mutation list for each
of the 35 tasks. It never modifies source CSV, WT PDBs, source TXT files, or the
remote-server result CSV. Each task is calculated once per unique ID, with
three BuildModel mutant/paired-WT-reference pairs and explicit partner groups.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from Bio.PDB import PDBParser

ROOT = Path('/data/data_split')
CSV_PATH = ROOT / 'raw' / 'M1101.csv'
WT_DIR = ROOT / 'splits' / 'M1101' / 'wt'
SOURCE_TXT_DIR = Path('/data/external/M1101/pdb/txt')
PREPARED_DIR = Path('/data/foldx/M1101/foldx_recovery_inputs')
FOLDX_DEFAULT = Path('/data/tools/foldx')
OUT_ROOT = Path('/data/foldx/FoldX_recomputed_M1101_missing35_20261105_40proc')
NUM_RUNS = 3
COMMAND_TIMEOUT = 3600
EXPECTED_SOURCE_ROWS = 36

# Exactly the 35 unique IDs previously missing from the remote M1101 run.
MISSING_IDS = [
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

TOKEN_RE = re.compile(r'^([A-Z])([A-Z0-9])(\d+)([A-Z])$')
AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M',
    'SEC':'U','PYL':'O',
}


def now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def norm(value: Any) -> str:
    return str(value if value is not None else '').strip()


def safe_name(value: str) -> str:
    value = norm(value)
    safe = re.sub(r'[^A-Za-z0-9._,+-]+', '_', value)
    if not safe or safe in {'.', '..'}:
        raise ValueError(f'Unsafe task ID: {value!r}')
    return safe


def pdb_stem(value: Any) -> str:
    s = norm(value)
    if s.lower().endswith('.pdb'):
        s = s[:-4]
    if not s:
        raise ValueError('Empty PDB identifier')
    return s


def parse_partners(value: Any) -> Tuple[str, str]:
    s = norm(value)
    if s.count('_') != 1:
        raise ValueError(f'Invalid Partners value {s!r}; expected GROUP1_GROUP2')
    left, right = s.split('_', 1)
    left = re.sub(r'[\s,;:#|/+]+', '', left)
    right = re.sub(r'[\s,;:#|/+]+', '', right)
    if not left or not right or set(left) & set(right):
        raise ValueError(f'Empty or overlapping partner groups in {s!r}')
    return left, right


def parse_tokens(text: str) -> List[str]:
    raw = norm(text).rstrip(';').strip()
    tokens = [x.strip().upper() for x in re.split(r'[,;\s]+', raw) if x.strip()]
    if not tokens:
        raise ValueError('Empty mutation list')
    for token in tokens:
        if not TOKEN_RE.fullmatch(token):
            raise ValueError(f'Expected mapped standard FoldX token, got {token!r}')
    return tokens


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_model(path: Path):
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    try:
        return next(structure.get_models())
    except StopIteration:
        raise ValueError(f'No coordinate model in {path}')


def validate_mapped_tokens(tokens: List[str], model, group1: str, group2: str, task_id: str) -> None:
    chains = {c.id.strip() for c in model}
    missing = sorted((set(group1) | set(group2)) - chains)
    if missing:
        raise ValueError(f'{task_id}: partner chains absent from prepared PDB: {missing}')
    residue_index: Dict[Tuple[str, int], List[str]] = {}
    for chain in model:
        cid = chain.id.strip()
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            aa = AA3_TO_1.get(residue.resname.strip().upper())
            if aa is not None:
                residue_index.setdefault((cid, int(residue.id[1])), []).append(aa)
    seen = set()
    for token in tokens:
        m = TOKEN_RE.fullmatch(token)
        if not m:
            raise ValueError(f'{task_id}: invalid mapped token {token}')
        wt, chain, pos_s, mut = m.groups()
        pos = int(pos_s)
        if chain not in set(group1 + group2):
            raise ValueError(f'{task_id}: mutation chain {chain} is outside Partners')
        key = (chain, pos)
        if key in seen:
            raise ValueError(f'{task_id}: duplicate mutation position {key}')
        seen.add(key)
        observed = residue_index.get(key, [])
        if len(observed) != 1:
            raise ValueError(f'{task_id}: expected one standard residue at {chain}:{pos}, found {observed}')
        if observed[0] != wt:
            raise ValueError(f'{task_id}: WT mismatch for {token}; prepared PDB has {observed[0]}')
        if wt == mut:
            raise ValueError(f'{task_id}: no-op mutation remained in prepared list: {token}')


def load_prepared_task(row: Dict[str, Any], source_group: pd.DataFrame) -> dict:
    task_id = norm(row['ID'])
    task_dir = PREPARED_DIR / task_id
    pdb_base = pdb_stem(row['PDB'])
    pdb_path = task_dir / 'wt_renumbered.pdb'
    txt_path = task_dir / 'individual_list.txt'
    mapping_path = task_dir / 'mutation_mapping.csv'
    manifest_path = task_dir / 'manifest.json'
    original_txt = SOURCE_TXT_DIR / f'individual_list_{task_id}.txt'
    if not all(p.is_file() for p in (pdb_path, txt_path, mapping_path, manifest_path, original_txt)):
        missing = [str(p) for p in (pdb_path, txt_path, mapping_path, manifest_path, original_txt) if not p.is_file()]
        raise FileNotFoundError(f'{task_id}: prepared inputs missing: {missing}')

    partners = norm(row['Partners'])
    group1, group2 = parse_partners(partners)
    with manifest_path.open(encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('status') != 'READY_FOR_FOLDX_PREFLIGHT':
        raise ValueError(f'{task_id}: invalid preparation manifest status {manifest.get("status")!r}')

    original_tokens = [x.upper() for x in re.split(r'[,;\s]+', original_txt.read_text(encoding='ascii').strip().rstrip(';')) if x.strip()]
    mapping = pd.read_csv(mapping_path, dtype=str).fillna('')
    required = {'txt_token', 'mapped_token', 'included_in_foldx_list'}
    if not required.issubset(mapping.columns):
        raise ValueError(f'{task_id}: mapping table missing {sorted(required - set(mapping.columns))}')
    mapped_source = [norm(x).upper() for x in mapping['txt_token'].tolist()]
    if sorted(mapped_source) != sorted(original_tokens):
        raise ValueError(f'{task_id}: source TXT differs from audited mapping table')

    effective = []
    no_ops = []
    for _, mrow in mapping.iterrows():
        included = norm(mrow['included_in_foldx_list']).lower() in {'true', '1', 'yes'}
        token = norm(mrow['mapped_token']).upper()
        if included:
            effective.append(token)
        else:
            no_ops.append(norm(mrow['txt_token']).upper())
    prepared_tokens = parse_tokens(txt_path.read_text(encoding='ascii'))
    if prepared_tokens != effective:
        raise ValueError(f'{task_id}: prepared mutation TXT differs from mapping table')

    csv_tokens = [x.upper() for x in re.split(r'[,;\s]+', norm(row['Mutation']).strip().rstrip(';')) if x.strip()]
    if sorted(csv_tokens) != sorted(original_tokens):
        raise ValueError(f'{task_id}: CSV Mutation differs from original source TXT')

    model = read_model(pdb_path)
    validate_mapped_tokens(effective, model, group1, group2, task_id)
    return {
        'task_id': task_id, 'pdb': pdb_base, 'PDB': pdb_base,
        'partners': partners, 'Partners': partners, 'Mutation': norm(row['Mutation']),
        'group1': group1, 'group2': group2, 'input_pdb': str(pdb_path),
        'input_txt': str(txt_path), 'source_txt': str(original_txt),
        'mapping_csv': str(mapping_path), 'manifest': str(manifest_path),
        'input_pdb_sha256': sha256_file(pdb_path),
        'input_txt_sha256': sha256_file(txt_path),
        'original_mutation_count': len(original_tokens),
        'effective_mutation_count': len(effective),
        'no_op_mutations_removed': len(no_ops),
        'effective_tokens': effective, 'removed_no_ops': no_ops,
        'source_record_count': int(len(source_group)),
        'input_mode': 'audited_renumbered',
    }


def canonical_group(value: str) -> str:
    return ''.join(sorted(re.findall(r'[A-Za-z0-9]', str(value).upper())))


def run_command(cmd: Sequence[str], cwd: Path, log, label: str) -> None:
    cmd = [str(x) for x in cmd]
    log.write(f'\n[{now()}] {label}: {" ".join(cmd)}\n')
    log.flush()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                              check=False, timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        log.write(f'{label}: TIMEOUT after {COMMAND_TIMEOUT} seconds\n')
        raise RuntimeError(f'{label} timed out after {COMMAND_TIMEOUT} seconds') from exc
    log.write(f'[{label}] returncode={proc.returncode}\n')
    if proc.stdout:
        log.write(f'[{label}] STDOUT:\n{proc.stdout}\n')
    if proc.stderr:
        log.write(f'[{label}] STDERR:\n{proc.stderr}\n')
    log.flush()
    if proc.returncode != 0:
        raise RuntimeError(f'{label} failed with return code {proc.returncode}')


def find_mutant_pairs(task_dir: Path, pdb_base: str) -> List[Tuple[Path, Path]]:
    pattern = re.compile(rf'^{re.escape(pdb_base)}_\d+_\d+\.pdb$', re.IGNORECASE)
    mutants = sorted(p for p in task_dir.glob('*.pdb') if pattern.fullmatch(p.name))
    pairs = []
    for mutant in mutants:
        wt_ref = task_dir / f'WT_{mutant.name}'
        if not wt_ref.is_file():
            raise RuntimeError(f'Missing paired WT reference for {mutant.name}')
        pairs.append((mutant, wt_ref))
    if len(pairs) != NUM_RUNS:
        raise RuntimeError(f'Expected {NUM_RUNS} mutant/WT pairs, found {len(pairs)}')
    return pairs


def parse_interaction_fxout(path: Path, group1: str, group2: str) -> float:
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    target1, target2 = canonical_group(group1), canonical_group(group2)
    for i, line in enumerate(lines):
        if 'interaction energy' not in line.lower():
            continue
        header = [x.strip() for x in line.split('\t')]
        norm_header = [re.sub(r'[^a-z0-9]', '', x.lower()) for x in header]
        required = {'interactionenergy', 'group1', 'group2'}
        if not required.issubset(set(norm_header)):
            continue
        eidx = norm_header.index('interactionenergy')
        g1idx, g2idx = norm_header.index('group1'), norm_header.index('group2')
        matched = []
        for data_line in lines[i + 1:]:
            if not data_line.strip() or data_line.lstrip().startswith('#'):
                continue
            values = [x.strip() for x in data_line.split('\t')]
            if len(values) <= max(eidx, g1idx, g2idx):
                continue
            try:
                energy = float(values[eidx])
            except ValueError:
                continue
            g1, g2 = canonical_group(values[g1idx]), canonical_group(values[g2idx])
            if ((g1 == target1 and g2 == target2) or (g1 == target2 and g2 == target1)) and math.isfinite(energy):
                matched.append(energy)
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            raise RuntimeError(f'{path.name}: multiple rows match {group1},{group2}: {matched}')
    raise RuntimeError(f'{path.name}: no unique Group1/Group2 match for {group1},{group2}')


def analyse_one(foldx: Path, task_dir: Path, pdb_path: Path, tag: str,
                group1: str, group2: str, log, label: str) -> float:
    run_command([
        str(foldx), '--command=AnalyseComplex', f'--pdb={pdb_path.name}',
        f'--analyseComplexChains={group1},{group2}', f'--output-file={tag}'
    ], task_dir, log, label)
    candidates = sorted(task_dir.glob(f'Interaction_{tag}*_AC.fxout'))
    if not candidates:
        candidates = sorted(task_dir.glob(f'Summary_{tag}*.fxout'))
    if len(candidates) != 1:
        raise RuntimeError(f'{label}: expected exactly one FXOUT, found {[p.name for p in candidates]}')
    return parse_interaction_fxout(candidates[0], group1, group2)


def get_existing_success(task_root: Path) -> Optional[dict]:
    if not task_root.is_dir():
        return None
    for attempt in sorted(task_root.glob('attempt_*')):
        result_path = attempt / 'task_result.json'
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if result.get('status') == 'success' and result.get('n_pairs') == NUM_RUNS:
                    result['skipped_existing_success'] = True
                    return result
            except Exception:
                pass
    return None


def new_attempt_dir(task_root: Path) -> Path:
    task_root.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in task_root.glob('attempt_*'):
        m = re.fullmatch(r'attempt_(\d+)', p.name)
        if m:
            nums.append(int(m.group(1)))
    dest = task_root / f'attempt_{max(nums, default=0) + 1:03d}'
    dest.mkdir(parents=False, exist_ok=False)
    return dest


def run_task(info: dict, foldx_exe: str, output_root: str) -> dict:
    task_root = Path(output_root) / 'M1101' / safe_name(info['task_id'])
    prior = get_existing_success(task_root)
    if prior is not None:
        return prior
    task_dir = new_attempt_dir(task_root)
    log_path = task_dir / f'{safe_name(info["task_id"])}.log'
    result = {
        'dataset': 'M1101', 'task_id': info['task_id'], 'pdb': info['pdb'],
        'PDB': info['PDB'], 'Partners': info['Partners'], 'Mutation': info['Mutation'],
        'partners': info['partners'], 'group1': info['group1'], 'group2': info['group2'],
        'input_mode': info['input_mode'], 'original_mutation_count': info['original_mutation_count'],
        'effective_mutation_count': info['effective_mutation_count'],
        'no_op_mutations_removed': info['no_op_mutations_removed'],
        'source_record_count': info['source_record_count'],
        'status': 'failed', 'n_pairs': 0, 'predicted_ddG': None,
        'mean_wt_reference_energy': None, 'mean_mutant_energy': None,
        'ddg_sd_population': None, 'pair_ddgs': None, 'pair_details': None,
        'error': None, 'log_path': str(log_path), 'output_dir': str(task_dir),
    }
    with log_path.open('w', encoding='utf-8') as log:
        log.write(f'=== FoldX missing-35 recomputation {info["task_id"]} at {now()} ===\n')
        log.write(json.dumps({k: v for k, v in info.items() if k != 'effective_tokens'}, ensure_ascii=False, indent=2) + '\n')
        try:
            foldx = Path(foldx_exe)
            if not foldx.is_file() or not os.access(foldx, os.X_OK):
                raise FileNotFoundError(f'FoldX executable missing/not executable: {foldx}')
            local_pdb = task_dir / f'{info["pdb"]}.pdb'
            shutil.copy2(info['input_pdb'], local_pdb)
            shutil.copy2(info['input_txt'], task_dir / 'source_prepared_individual_list.txt')
            shutil.copy2(info['source_txt'], task_dir / 'source_original_individual_list.txt')
            (task_dir / 'individual_list.txt').write_text(','.join(info['effective_tokens']) + ';\n', encoding='ascii')
            log.write(f'WT input: {info["input_pdb"]}\n')
            log.write(f'Prepared mutation list: {info["input_txt"]}\n')
            log.write(f'Effective mutations: {info["effective_tokens"]}\n')
            log.write(f'Removed no-op mutations: {info["removed_no_ops"]}\n')
            run_command([
                str(foldx), '--command=BuildModel', f'--pdb={local_pdb.name}',
                '--mutant-file=individual_list.txt', f'--numberOfRuns={NUM_RUNS}',
                '--out-pdb=true', '--output-file=buildmodel'
            ], task_dir, log, 'BuildModel')
            pairs = find_mutant_pairs(task_dir, info['pdb'])
            pair_details, pair_ddgs, wt_energies, mut_energies = [], [], [], []
            for idx, (mutant_pdb, wt_ref_pdb) in enumerate(pairs, start=1):
                wt_energy = analyse_one(foldx, task_dir, wt_ref_pdb, f'pair{idx:02d}_wt',
                                        info['group1'], info['group2'], log, f'Analyse_WT_reference_{idx}')
                mut_energy = analyse_one(foldx, task_dir, mutant_pdb, f'pair{idx:02d}_mut',
                                         info['group1'], info['group2'], log, f'Analyse_mutant_{idx}')
                ddg = mut_energy - wt_energy
                wt_energies.append(wt_energy)
                mut_energies.append(mut_energy)
                pair_ddgs.append(ddg)
                pair_details.append({
                    'pair': idx, 'mutant_pdb': mutant_pdb.name, 'wt_reference_pdb': wt_ref_pdb.name,
                    'wt_interaction_energy': wt_energy, 'mut_interaction_energy': mut_energy,
                    'paired_ddG': ddg,
                })
                log.write(f'PAIR {idx}: {mut_energy} - {wt_energy} = {ddg}\n')
            mean_ddg = sum(pair_ddgs) / len(pair_ddgs)
            variance = sum((x - mean_ddg) ** 2 for x in pair_ddgs) / len(pair_ddgs)
            result.update({
                'status': 'success', 'n_pairs': len(pair_ddgs), 'predicted_ddG': mean_ddg,
                'mean_wt_reference_energy': sum(wt_energies) / len(wt_energies),
                'mean_mutant_energy': sum(mut_energies) / len(mut_energies),
                'ddg_sd_population': math.sqrt(variance),
                'pair_ddgs': ';'.join(f'{x:.8f}' for x in pair_ddgs),
                'pair_details': json.dumps(pair_details, separators=(',', ':')),
            })
            (task_dir / 'pair_details.json').write_text(json.dumps(pair_details, indent=2) + '\n', encoding='utf-8')
            log.write(f'MEAN paired ddG = {mean_ddg}\n=== SUCCESS ===\n')
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            log.write(f'ERROR: {result["error"]}\n=== FAILED ===\n')
    (task_dir / 'task_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Recompute only the 35 missing unique M1101 FoldX tasks.')
    parser.add_argument('--foldx', default=str(FOLDX_DEFAULT))
    parser.add_argument('--workers', type=int, default=35)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be >= 1')
    if len(MISSING_IDS) != 35 or len(set(MISSING_IDS)) != 35:
        raise RuntimeError('Internal missing-ID list is not exactly 35 unique tasks')
    foldx = Path(args.foldx)
    if not foldx.is_file() or not os.access(foldx, os.X_OK):
        print(f'ERROR: FoldX executable missing/not executable: {foldx}', file=sys.stderr)
        return 2
    if not CSV_PATH.is_file():
        print(f'ERROR: M1101 source CSV missing: {CSV_PATH}', file=sys.stderr)
        return 2
    if not PREPARED_DIR.is_dir() or not SOURCE_TXT_DIR.is_dir():
        print('ERROR: prepared-input or source-TXT directory missing', file=sys.stderr)
        return 2

    source_df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    required = {'ID', 'PDB', 'Partners', 'Mutation'}
    if not required.issubset(source_df.columns):
        print(f'ERROR: M1101.csv missing columns {sorted(required - set(source_df.columns))}', file=sys.stderr)
        return 2
    source_df['ID'] = source_df['ID'].map(norm)
    subset = source_df[source_df['ID'].isin(MISSING_IDS)].copy()
    found_ids = set(subset['ID'])
    missing_ids = sorted(set(MISSING_IDS) - found_ids)
    extra_ids = sorted(found_ids - set(MISSING_IDS))
    if missing_ids or extra_ids:
        print(f'ERROR: ID-list mismatch. missing={missing_ids}; extra={extra_ids}', file=sys.stderr)
        return 2
    if subset['ID'].nunique() != 35 or len(subset) != EXPECTED_SOURCE_ROWS:
        print(f'ERROR: expected 35 unique IDs and {EXPECTED_SOURCE_ROWS} source rows; found {subset.ID.nunique()} IDs and {len(subset)} rows', file=sys.stderr)
        return 2

    tasks = []
    errors = []
    for task_id in MISSING_IDS:
        group = subset[subset['ID'] == task_id]
        first = group.iloc[0].to_dict()
        for col in ('PDB', 'Partners', 'Mutation'):
            if len({norm(x) for x in group[col].tolist()}) != 1:
                errors.append({'task_id': task_id, 'error': f'Conflicting duplicate-ID values in {col}'})
                break
        if any(e['task_id'] == task_id for e in errors):
            continue
        try:
            tasks.append(load_prepared_task(first, group))
        except Exception as exc:
            errors.append({'task_id': task_id, 'error': f'{type(exc).__name__}: {exc}'})

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if errors or len(tasks) != 35:
        pd.DataFrame(errors).to_csv(OUT_ROOT / 'preflight_failures.csv', index=False)
        print(f'PREFLIGHT FAILED: prepared {len(tasks)}/35; errors={len(errors)}. No FoldX command executed.')
        for e in errors:
            print(f'FAIL {e["task_id"]}: {e["error"]}')
        return 2

    task_manifest = [{k: v for k, v in t.items() if k not in {'effective_tokens', 'removed_no_ops'}} for t in tasks]
    pd.DataFrame(task_manifest).to_csv(OUT_ROOT / 'preflight_tasks.csv', index=False)
    manifest = {
        'created_at': now(), 'dataset': 'M1101', 'scope': '35 previously missing unique IDs only',
        'source_rows_for_patch': len(subset), 'unique_tasks': len(tasks),
        'foldx_executable': str(foldx), 'foldx_sha256': sha256_file(foldx),
        'foldx_version_expected': '5.1', 'buildmodel_runs': NUM_RUNS,
        'analyse_complex_chains': 'explicit Partners groups', 'workers': args.workers,
        'prepared_input_dir': str(PREPARED_DIR), 'source_txt_dir': str(SOURCE_TXT_DIR),
    }
    (OUT_ROOT / 'preflight_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('M1101 missing-35 FoldX recomputation')
    print(f'Mode       : {"EXECUTE" if args.execute else "PREFLIGHT ONLY"}')
    print(f'FoldX      : {foldx}')
    print(f'Output     : {OUT_ROOT}')
    print(f'Workers    : {args.workers}')
    print('Protocol   : BuildModel x3; paired WT references; explicit AnalyseComplex partner groups')
    print(f'Source rows: {len(subset)}; unique missing IDs: {len(tasks)}; preflight errors: 0')
    print('PREFLIGHT PASSED')
    if not args.execute:
        print('No FoldX command executed. Add --execute to run only these 35 tasks.')
        return 0

    started = time.time()
    results = []
    pending = []
    for info in tasks:
        old = get_existing_success(OUT_ROOT / 'M1101' / safe_name(info['task_id']))
        if old is not None:
            results.append(old)
        else:
            pending.append(info)
    print(f'Tasks total=35; already successful={len(results)}; pending={len(pending)}')
    with cf.ProcessPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as executor:
        future_map = {executor.submit(run_task, info, str(foldx), str(OUT_ROOT)): info for info in pending}
        completed = 0
        for future in cf.as_completed(future_map):
            info = future_map[future]
            completed += 1
            try:
                res = future.result()
            except Exception as exc:
                res = {'dataset': 'M1101', 'task_id': info['task_id'], 'status': 'failed',
                       'predicted_ddG': None, 'n_pairs': 0, 'error': f'WorkerError: {type(exc).__name__}: {exc}'}
            results.append(res)
            if completed % 5 == 0 or completed == len(pending):
                ok = sum(r.get('status') == 'success' for r in results)
                print(f'Progress {completed}/{len(pending)} new tasks; total success={ok}; failed={len(results)-ok}', flush=True)

    by_id = {r['task_id']: r for r in results}
    ordered = [by_id[x] for x in MISSING_IDS if x in by_id]
    unique_df = pd.DataFrame(ordered)
    unique_df.to_csv(OUT_ROOT / 'predictions_by_unique_id.csv', index=False)
    patch = subset.merge(
        unique_df[['task_id', 'status', 'predicted_ddG', 'n_pairs', 'pair_ddgs', 'ddg_sd_population', 'log_path', 'output_dir', 'error']],
        left_on='ID', right_on='task_id', how='left', validate='many_to_one'
    )
    patch = patch.rename(columns={'predicted_ddG': 'foldx_patch_predicted_ddG',
                                  'status': 'foldx_patch_status', 'n_pairs': 'foldx_patch_n_pairs',
                                  'pair_ddgs': 'foldx_patch_pair_ddgs', 'ddg_sd_population': 'foldx_patch_ddg_sd_population',
                                  'log_path': 'foldx_patch_log_path', 'output_dir': 'foldx_patch_output_dir',
                                  'error': 'foldx_patch_error'})
    patch.to_csv(OUT_ROOT / 'predictions_record_level.csv', index=False)
    failures = unique_df[unique_df['status'] != 'success'] if not unique_df.empty else unique_df
    failures.to_csv(OUT_ROOT / 'failed_tasks.csv', index=False)
    success = int((unique_df['status'] == 'success').sum()) if not unique_df.empty else 0
    summary = pd.DataFrame([{
        'dataset': 'M1101', 'scope': 'missing_35_only', 'source_rows': len(subset),
        'unique_tasks': 35, 'success': success, 'failed': 35 - success,
        'finite_unique_predictions': int(pd.to_numeric(unique_df.get('predicted_ddG'), errors='coerce').notna().sum()) if not unique_df.empty else 0,
        'record_level_rows': len(patch),
    }])
    summary.to_csv(OUT_ROOT / 'run_summary.csv', index=False)
    (OUT_ROOT / 'run_completion.json').write_text(json.dumps({
        'completed_at': now(), 'elapsed_seconds': round(time.time() - started, 2),
        'summary': summary.iloc[0].to_dict(),
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'M1101 missing-35: success={success}/35; failed={35-success}; source rows patched={len(patch)}')
    print(f'Final outputs: {OUT_ROOT}')
    return 0 if success == 35 else 3


if __name__ == '__main__':
    raise SystemExit(main())

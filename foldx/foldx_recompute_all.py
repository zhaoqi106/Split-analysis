#!/usr/bin/env python3
"""Unified, non-destructive FoldX 5.1 recomputation for S1131 and M1101.

Default: full preflight only.  Use --execute to run all unique mutation tasks.
The script validates all inputs before launching any FoldX process, uses the
M1101 audited renumbered inputs for the 35 insertion-code tasks, and uses the
already-Repaired WT structures plus source Individual Lists for all other tasks.
Each BuildModel mutant is paired with its own WT reference.  AnalyseComplex is
run with explicit partner-chain groups, and the fxout Group1/Group2 fields must
match those groups exactly.  Existing source files and old result CSVs are never
modified.  Completed tasks are skipped on rerun; failed/incomplete attempts are
preserved and retried in a new attempt directory.
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
FOLDX_DEFAULT = Path('/data/tools/foldx')
OUT_ROOT = Path('/data/foldx/FoldX_recomputed_20261105')
NUM_RUNS = 3
COMMAND_TIMEOUT = 3600

DATASETS = {
    'S1131': {
        'csv': ROOT / 'raw' / 'S1131.csv',
        'wt_dir': ROOT / 'splits' / 'S1131' / 'wt',
        'txt_dir': Path('/data/external/S1131/pdb/txt'),
        'prepared_dir': None,
        'expected_rows': 1131,
        'expected_unique': 1131,
    },
    'M1101': {
        'csv': ROOT / 'raw' / 'M1101.csv',
        'wt_dir': ROOT / 'splits' / 'M1101' / 'wt',
        'txt_dir': Path('/data/external/M1101/pdb/txt'),
        'prepared_dir': Path('/data/foldx/M1101/foldx_recovery_inputs'),
        'expected_rows': 1101,
        'expected_unique': 1096,
    },
}

TOKEN_RE = re.compile(r'^([A-Z])([A-Z0-9])(\d+)([A-Z])$')
INSERTION_TOKEN_RE = re.compile(r'^([A-Z])([A-Z0-9])(\d+)([A-Z])([A-Z])$')
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
        raise ValueError(f'Invalid Partners value {s!r}, expected GROUP1_GROUP2')
    left, right = s.split('_', 1)
    left = re.sub(r'[\s,;:#|/+]+', '', left)
    right = re.sub(r'[\s,;:#|/+]+', '', right)
    if not left or not right or set(left) & set(right):
        raise ValueError(f'Empty or overlapping partner groups in {s!r}')
    return left, right


def parse_tokens_text(text: str) -> List[str]:
    raw = norm(text).rstrip(';').strip()
    tokens = [x.strip().upper() for x in re.split(r'[,;\s]+', raw) if x.strip()]
    if not tokens:
        raise ValueError('Empty mutation list')
    for token in tokens:
        if not TOKEN_RE.fullmatch(token) and not INSERTION_TOKEN_RE.fullmatch(token):
            raise ValueError(f'Unsupported mutation token {token!r}; expected standard or insertion-coded FoldX token')
    return tokens


def parse_mutation_column(value: Any) -> List[str]:
    return parse_tokens_text(norm(value))


def read_structure(path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    try:
        return next(structure.get_models())
    except StopIteration:
        raise ValueError(f'No coordinate model in {path}')


def locate_case_insensitive(directory: Path, filename: str) -> Optional[Path]:
    direct = directory / filename
    if direct.is_file():
        return direct
    if not directory.is_dir():
        return None
    target = filename.lower()
    for p in directory.iterdir():
        if p.is_file() and p.name.lower() == target:
            return p
    return None


def chain_residue_index(model) -> Tuple[set, Dict[Tuple[str, int], List[Tuple[str, str]]]]:
    chain_ids = set()
    index: Dict[Tuple[str, int], List[Tuple[str, str]]] = {}
    for chain in model:
        chain_id = chain.id.strip()
        chain_ids.add(chain_id)
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            seq = int(residue.id[1])
            icode = str(residue.id[2]).strip()
            aa = AA3_TO_1.get(residue.resname.strip().upper())
            if aa is None:
                continue
            index.setdefault((chain_id, seq), []).append((icode, aa))
    return chain_ids, index


def validate_tokens_against_model(tokens: List[str], model, group1: str, group2: str,
                                   task_id: str) -> Tuple[List[str], List[str]]:
    chain_ids, residue_index = chain_residue_index(model)
    missing_partner = sorted((set(group1) | set(group2)) - chain_ids)
    if missing_partner:
        raise ValueError(f'{task_id}: partner chains absent from WT PDB: {missing_partner}')
    effective: List[str] = []
    no_ops: List[str] = []
    seen = set()
    for token in tokens:
        m = TOKEN_RE.fullmatch(token)
        if m is None:
            raise ValueError(f'{task_id}: insertion-coded mutation {token!r} requires audited renumbered input')
        wt, chain, pos_s, mut = m.groups()
        pos = int(pos_s)
        if chain not in chain_ids:
            raise ValueError(f'{task_id}: mutation chain {chain!r} absent from WT PDB')
        if chain not in set(group1 + group2):
            raise ValueError(f'{task_id}: mutation chain {chain!r} is outside Partners={group1}_{group2}')
        key = (chain, pos)
        if key in seen:
            raise ValueError(f'{task_id}: duplicate mutation position {key}')
        seen.add(key)
        candidates = residue_index.get(key, [])
        blank = [aa for icode, aa in candidates if not icode]
        if len(blank) != 1:
            if candidates:
                detail = candidates
                raise ValueError(f'{task_id}: ambiguous/nonblank insertion-code residue at {chain}:{pos}, candidates={detail}; use audited renumbered input')
            raise ValueError(f'{task_id}: residue not found at {chain}:{pos} for token {token}')
        observed = blank[0]
        if observed != wt:
            raise ValueError(f'{task_id}: WT mismatch for {token}, PDB has {observed} at {chain}:{pos}')
        if wt == mut:
            no_ops.append(token)
        else:
            effective.append(token)
    if not effective:
        raise ValueError(f'{task_id}: all mutations are no-op substitutions')
    return effective, no_ops


def read_prepared_mapping(prep_dir: Path, original_tokens: List[str], task_id: str) -> Tuple[List[str], List[str]]:
    mapping_path = prep_dir / 'mutation_mapping.csv'
    manifest_path = prep_dir / 'manifest.json'
    if not mapping_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f'{task_id}: prepared input missing mutation_mapping.csv or manifest.json')
    mapping = pd.read_csv(mapping_path, dtype=str).fillna('')
    required = {'txt_token', 'mapped_token', 'included_in_foldx_list'}
    if not required.issubset(mapping.columns):
        raise ValueError(f'{task_id}: mapping table missing columns {sorted(required - set(mapping.columns))}')
    mapped_original = [norm(x).upper() for x in mapping['txt_token'].tolist()]
    if sorted(mapped_original) != sorted(original_tokens):
        raise ValueError(f'{task_id}: original mutation list differs from audited mapping table')
    effective, no_ops = [], []
    for _, row in mapping.iterrows():
        token = norm(row['mapped_token']).upper()
        flag = norm(row['included_in_foldx_list']).lower()
        included = flag in {'true', '1', 'yes'}
        if included:
            effective.append(token)
        else:
            no_ops.append(norm(row['txt_token']).upper())
    prepared_txt = prep_dir / 'individual_list.txt'
    prepared_tokens = parse_tokens_text(prepared_txt.read_text(encoding='ascii'))
    if prepared_tokens != effective:
        raise ValueError(f'{task_id}: prepared TXT differs from mapping table effective tokens')
    with manifest_path.open(encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('status') != 'READY_FOR_FOLDX_PREFLIGHT':
        raise ValueError(f'{task_id}: unexpected preparation manifest status {manifest.get("status")!r}')
    return effective, no_ops


def prepare_dataset_tasks(dataset: str, cfg: dict) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    csv_path = cfg['csv']
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = {'ID', 'PDB', 'Partners', 'Mutation'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{dataset} CSV missing required columns: {sorted(missing)}')
    if len(df) != cfg['expected_rows']:
        raise ValueError(f'{dataset}: expected {cfg["expected_rows"]} source rows, found {len(df)}')
    df['ID'] = df['ID'].map(norm)
    if (df['ID'] == '').any():
        raise ValueError(f'{dataset}: empty ID in source CSV')
    unique_count = df['ID'].nunique()
    if unique_count != cfg['expected_unique']:
        raise ValueError(f'{dataset}: expected {cfg["expected_unique"]} unique IDs, found {unique_count}')

    # One calculation per unique mutation identifier.  Duplicate rows must share
    # the same structure, partner groups, and mutation definition.
    unique_rows = []
    errors: List[dict] = []
    for task_id, group in df.groupby('ID', sort=True):
        for col in ('PDB', 'Partners', 'Mutation'):
            vals = {norm(v) for v in group[col].tolist()}
            if len(vals) != 1:
                errors.append({'dataset': dataset, 'task_id': task_id,
                               'error': f'Conflicting duplicate-ID values in {col}: {sorted(vals)}'})
        if errors and errors[-1].get('task_id') == task_id:
            continue
        unique_rows.append(group.iloc[0].to_dict())

    # Cache parsed structures by resolved input path to avoid repeated parsing.
    model_cache: Dict[str, Any] = {}
    tasks: List[dict] = []
    for row in unique_rows:
        task_id = norm(row['ID'])
        try:
            pdb_id = pdb_stem(row['PDB'])
            group1, group2 = parse_partners(row['Partners'])
            source_txt = cfg['txt_dir'] / f'individual_list_{task_id}.txt'
            if not source_txt.is_file():
                raise FileNotFoundError(f'Mutation TXT missing: {source_txt}')
            original_tokens = parse_tokens_text(source_txt.read_text(encoding='ascii'))
            csv_tokens = parse_mutation_column(row['Mutation'])
            if sorted(original_tokens) != sorted(csv_tokens):
                raise ValueError(f'{task_id}: source CSV Mutation differs from Individual List TXT')

            input_mode = 'original_repaired'
            input_pdb = locate_case_insensitive(cfg['wt_dir'], f'{pdb_id}.pdb')
            input_txt = source_txt
            prep_dir = None
            if cfg['prepared_dir'] is not None:
                candidate_dir = cfg['prepared_dir'] / task_id
                candidate_pdb = candidate_dir / 'wt_renumbered.pdb'
                candidate_txt = candidate_dir / 'individual_list.txt'
                if candidate_pdb.is_file() and candidate_txt.is_file():
                    input_pdb, input_txt, prep_dir = candidate_pdb, candidate_txt, candidate_dir
                    input_mode = 'audited_renumbered'
            if input_pdb is None or not input_pdb.is_file():
                raise FileNotFoundError(f'WT PDB missing for {task_id}: {cfg["wt_dir"] / (pdb_id + ".pdb")}')

            cache_key = str(input_pdb.resolve())
            if cache_key not in model_cache:
                model_cache[cache_key] = read_structure(input_pdb)
            model = model_cache[cache_key]

            if prep_dir is not None:
                effective_tokens, no_ops = read_prepared_mapping(prep_dir, original_tokens, task_id)
                # Validate the renumbered structure against the effective mapped tokens.
                effective_tokens, extra_no_ops = validate_tokens_against_model(
                    effective_tokens, model, group1, group2, task_id)
                if extra_no_ops:
                    raise ValueError(f'{task_id}: no-op remained in prepared mutation list: {extra_no_ops}')
            else:
                effective_tokens, no_ops = validate_tokens_against_model(
                    original_tokens, model, group1, group2, task_id)

            tasks.append({
                'dataset': dataset, 'task_id': task_id, 'pdb': pdb_id,
                'partners': norm(row['Partners']), 'group1': group1, 'group2': group2,
                'original_mutation_count': len(original_tokens),
                'effective_mutation_count': len(effective_tokens),
                'no_op_mutations_removed': len(no_ops),
                'original_tokens': original_tokens,
                'effective_tokens': effective_tokens,
                'input_mode': input_mode,
                'input_pdb': str(input_pdb), 'input_txt': str(input_txt),
                'source_txt': str(source_txt), 'source_csv': str(csv_path),
                'source_record_count': int(len(group)),
            })
        except Exception as exc:
            errors.append({'dataset': dataset, 'task_id': task_id, 'error': str(exc)})

    tasks_df = pd.DataFrame([{k: v for k, v in t.items() if k not in {'original_tokens', 'effective_tokens'}} for t in tasks])
    (OUT_ROOT / dataset).mkdir(parents=True, exist_ok=True)
    if not tasks_df.empty:
        tasks_df.to_csv(OUT_ROOT / dataset / 'preflight_tasks.csv', index=False)
    if errors:
        pd.DataFrame(errors).to_csv(OUT_ROOT / dataset / 'preflight_failures.csv', index=False)
    return df, tasks, errors


def canonical_group(value: str) -> str:
    chars = re.findall(r'[A-Za-z0-9]', str(value).upper())
    return ''.join(sorted(chars))


def run_command(cmd: Sequence[str], cwd: Path, log, label: str) -> None:
    cmd = [str(x) for x in cmd]
    log.write(f'\n[{now()}] {label}: {" ".join(cmd)}\n')
    log.flush()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                              check=False, timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        log.write(f'[{label}] TIMEOUT after {COMMAND_TIMEOUT} seconds\n')
        if exc.stdout:
            log.write(str(exc.stdout) + '\n')
        if exc.stderr:
            log.write(str(exc.stderr) + '\n')
        log.flush()
        raise RuntimeError(f'{label} timed out after {COMMAND_TIMEOUT} seconds')
    log.write(f'[{label}] returncode={proc.returncode}\n')
    if proc.stdout:
        log.write(f'[{label}] STDOUT:\n{proc.stdout}\n')
    if proc.stderr:
        log.write(f'[{label}] STDERR:\n{proc.stderr}\n')
    log.flush()
    if proc.returncode != 0:
        raise RuntimeError(f'{label} failed with return code {proc.returncode}')


def find_pairs(task_dir: Path, pdb_base: str) -> List[Tuple[Path, Path]]:
    pat = re.compile(rf'^{re.escape(pdb_base)}_\d+_\d+\.pdb$', re.IGNORECASE)
    mutants = sorted([p for p in task_dir.glob('*.pdb') if pat.fullmatch(p.name)])
    pairs = []
    for mutant in mutants:
        wt_ref = task_dir / f'WT_{mutant.name}'
        if not wt_ref.is_file():
            raise RuntimeError(f'Missing paired WT reference for {mutant.name}: expected {wt_ref.name}')
        pairs.append((mutant, wt_ref))
    if len(pairs) != NUM_RUNS:
        pdb_files = sorted(p.name for p in task_dir.glob('*.pdb'))
        raise RuntimeError(f'Expected {NUM_RUNS} mutant/WT-reference pairs, found {len(pairs)}; PDB files={pdb_files}')
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
            if (g1 == target1 and g2 == target2) or (g1 == target2 and g2 == target1):
                if math.isfinite(energy):
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
        raise RuntimeError(f'{label}: expected exactly one fxout, found {[p.name for p in candidates]}')
    return parse_interaction_fxout(candidates[0], group1, group2)


def next_attempt_dir(task_root: Path) -> Tuple[Path, Optional[dict]]:
    task_root.mkdir(parents=True, exist_ok=True)
    for p in sorted(task_root.glob('attempt_*')):
        result_path = p / 'task_result.json'
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if result.get('status') == 'success':
                    result['skipped_existing_success'] = True
                    return p, result
            except Exception:
                pass
    existing_nums = []
    for p in task_root.glob('attempt_*'):
        m = re.fullmatch(r'attempt_(\d+)', p.name)
        if m:
            existing_nums.append(int(m.group(1)))
    n = max(existing_nums, default=0) + 1
    dest = task_root / f'attempt_{n:03d}'
    dest.mkdir(parents=False, exist_ok=False)
    return dest, None


def run_task(info: dict, foldx_exe: str, output_root: str) -> dict:
    dataset, task_id = info['dataset'], info['task_id']
    task_root = Path(output_root) / dataset / safe_name(task_id)
    task_dir, prior = next_attempt_dir(task_root)
    if prior is not None:
        return prior
    log_path = task_dir / f'{safe_name(task_id)}.log'
    result = {
        'dataset': dataset, 'task_id': task_id, 'pdb': info['pdb'],
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
        log.write(f'=== FoldX recomputation {dataset}/{task_id} at {now()} ===\n')
        log.write(json.dumps({k: v for k, v in info.items() if k not in {'original_tokens', 'effective_tokens'}},
                             ensure_ascii=False, indent=2) + '\n')
        try:
            foldx = Path(foldx_exe)
            if not foldx.is_file() or not os.access(foldx, os.X_OK):
                raise FileNotFoundError(f'FoldX executable missing/not executable: {foldx}')
            pdb_base = info['pdb']
            local_pdb = task_dir / f'{pdb_base}.pdb'
            shutil.copy2(info['input_pdb'], local_pdb)
            shutil.copy2(info['source_txt'], task_dir / 'source_individual_list.txt')
            (task_dir / 'individual_list.txt').write_text(','.join(info['effective_tokens']) + ';\n', encoding='ascii')
            log.write(f'WT input: {info["input_pdb"]}\n')
            log.write(f'Mutation TXT: {info["source_txt"]}\n')
            log.write(f'Effective mutations: {info["effective_tokens"]}\n')
            log.write(f'No-op mutations removed: {info["no_op_mutations_removed"]}\n')

            run_command([
                str(foldx), '--command=BuildModel', f'--pdb={local_pdb.name}',
                '--mutant-file=individual_list.txt', f'--numberOfRuns={NUM_RUNS}',
                '--out-pdb=true', '--output-file=buildmodel'
            ], task_dir, log, 'BuildModel')

            pairs = find_pairs(task_dir, pdb_base)
            pair_details = []
            wt_energies, mut_energies, pair_ddgs = [], [], []
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
            mean_wt = sum(wt_energies) / len(wt_energies)
            mean_mut = sum(mut_energies) / len(mut_energies)
            variance = sum((x - mean_ddg) ** 2 for x in pair_ddgs) / len(pair_ddgs)
            result.update({
                'status': 'success', 'n_pairs': len(pair_ddgs), 'predicted_ddG': mean_ddg,
                'mean_wt_reference_energy': mean_wt, 'mean_mutant_energy': mean_mut,
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_existing_successes(dataset: str, tasks: List[dict], output_root: Path) -> List[dict]:
    results = []
    for info in tasks:
        task_root = output_root / dataset / safe_name(info['task_id'])
        found = None
        for p in sorted(task_root.glob('attempt_*')):
            rp = p / 'task_result.json'
            if rp.is_file():
                try:
                    r = json.loads(rp.read_text(encoding='utf-8'))
                    if r.get('status') == 'success':
                        found = r
                        break
                except Exception:
                    pass
        if found is not None:
            results.append(found)
    return results


def write_dataset_outputs(dataset: str, source_df: pd.DataFrame, task_results: List[dict], output_root: Path) -> None:
    ds_dir = output_root / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(task_results)
    if not result_df.empty:
        result_df.to_csv(ds_dir / 'predictions_by_unique_id.csv', index=False)
    merged = source_df.merge(
        result_df[['task_id', 'status', 'predicted_ddG', 'n_pairs', 'input_mode', 'log_path', 'error']]
            if not result_df.empty else pd.DataFrame(columns=['task_id','status','predicted_ddG','n_pairs','input_mode','log_path','error']),
        left_on='ID', right_on='task_id', how='left', validate='many_to_one'
    )
    merged = merged.rename(columns={
        'status': 'foldx_status', 'predicted_ddG': 'foldx_predicted_ddG',
        'n_pairs': 'foldx_n_pairs', 'input_mode': 'foldx_input_mode',
        'log_path': 'foldx_log_path', 'error': 'foldx_error',
    })
    merged.to_csv(ds_dir / 'predictions_record_level.csv', index=False)
    if not result_df.empty:
        failures = result_df[result_df['status'] != 'success']
        failures.to_csv(ds_dir / 'failed_tasks.csv', index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description='Unified non-destructive FoldX recomputation for S1131 and M1101.')
    parser.add_argument('--datasets', nargs='+', choices=sorted(DATASETS), default=['S1131', 'M1101'])
    parser.add_argument('--foldx', default=str(FOLDX_DEFAULT))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--execute', action='store_true', help='Run FoldX after full-dataset preflight succeeds.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be >= 1')
    foldx = Path(args.foldx)
    if not foldx.is_file() or not os.access(foldx, os.X_OK):
        print(f'ERROR: FoldX executable missing/not executable: {foldx}', file=sys.stderr)
        return 2

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print('Unified FoldX 5.1 recomputation')
    print(f'Mode       : {"EXECUTE" if args.execute else "PREFLIGHT ONLY"}')
    print(f'FoldX      : {foldx}')
    print(f'Output     : {OUT_ROOT}')
    print(f'Workers    : {args.workers}')
    print('Protocol   : BuildModel x3; paired WT references; explicit AnalyseComplex partner groups')

    all_data = {}
    all_tasks = []
    all_errors = []
    for dataset in args.datasets:
        cfg = DATASETS[dataset]
        print(f'\n[{dataset}] CSV={cfg["csv"]}')
        print(f'[{dataset}] WT={cfg["wt_dir"]}')
        print(f'[{dataset}] TXT={cfg["txt_dir"]}')
        if not cfg['txt_dir'].is_dir():
            all_errors.append({'dataset': dataset, 'task_id': '', 'error': f'TXT directory missing: {cfg["txt_dir"]}'})
            continue
        if not cfg['wt_dir'].is_dir():
            all_errors.append({'dataset': dataset, 'task_id': '', 'error': f'WT directory missing: {cfg["wt_dir"]}'})
            continue
        try:
            source_df, tasks, errors = prepare_dataset_tasks(dataset, cfg)
            all_data[dataset] = source_df
            all_tasks.extend(tasks)
            all_errors.extend(errors)
            print(f'[{dataset}] source records={len(source_df)}; unique tasks={len(tasks)}; preflight errors={len(errors)}')
        except Exception as exc:
            all_errors.append({'dataset': dataset, 'task_id': '', 'error': f'{type(exc).__name__}: {exc}'})

    if all_errors:
        error_path = OUT_ROOT / 'preflight_failures_all.csv'
        pd.DataFrame(all_errors).to_csv(error_path, index=False)
        print('\nPREFLIGHT FAILED. No FoldX command was executed.')
        print(pd.DataFrame(all_errors).head(60).to_string(index=False))
        print(f'Full failure report: {error_path}')
        return 2

    task_counts = {ds: sum(1 for x in all_tasks if x['dataset'] == ds) for ds in args.datasets}
    preflight_manifest = {
        'created_at': now(), 'foldx_executable': str(foldx), 'foldx_sha256': sha256_file(foldx),
        'foldx_version_expected': '5.1', 'num_runs': NUM_RUNS,
        'analyse_complex_chains': 'explicit Partners groups', 'datasets': args.datasets,
        'source_rows': {ds: len(all_data[ds]) for ds in args.datasets},
        'unique_tasks': task_counts, 'total_unique_tasks': len(all_tasks),
        'total_source_rows': sum(len(all_data[ds]) for ds in args.datasets),
        'workers': args.workers,
    }
    (OUT_ROOT / 'preflight_manifest.json').write_text(json.dumps(preflight_manifest, indent=2) + '\n', encoding='utf-8')
    print('\nPREFLIGHT PASSED')
    for ds in args.datasets:
        print(f'  {ds}: {len(all_data[ds])} source rows; {task_counts[ds]} unique tasks')
    print(f'  TOTAL: {preflight_manifest["total_source_rows"]} source rows; {len(all_tasks)} unique tasks')
    print(f'Preflight manifest: {OUT_ROOT / "preflight_manifest.json"}')

    if not args.execute:
        print('\nNo FoldX command was executed. Add --execute to run all unique tasks.')
        return 0

    start = time.time()
    results_by_dataset: Dict[str, List[dict]] = {ds: [] for ds in args.datasets}
    # Reuse completed successful tasks from earlier attempts and run all remaining tasks.
    pending = []
    for info in all_tasks:
        task_root = OUT_ROOT / info['dataset'] / safe_name(info['task_id'])
        existing_success = None
        for p in sorted(task_root.glob('attempt_*')):
            rp = p / 'task_result.json'
            if rp.is_file():
                try:
                    old = json.loads(rp.read_text(encoding='utf-8'))
                    if old.get('status') == 'success':
                        old['skipped_existing_success'] = True
                        existing_success = old
                        break
                except Exception:
                    pass
        if existing_success is not None:
            results_by_dataset[info['dataset']].append(existing_success)
        else:
            pending.append(info)

    print(f'\nTasks total={len(all_tasks)}, already successful={len(all_tasks)-len(pending)}, pending={len(pending)}')
    with cf.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run_task, info, str(foldx), str(OUT_ROOT)): info for info in pending}
        completed = 0
        for future in cf.as_completed(future_map):
            info = future_map[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                result = {'dataset': info['dataset'], 'task_id': info['task_id'], 'status': 'failed',
                          'predicted_ddG': None, 'error': f'WorkerError: {type(exc).__name__}: {exc}'}
            results_by_dataset[info['dataset']].append(result)
            if completed % 10 == 0 or completed == len(pending):
                ok = sum(1 for arr in results_by_dataset.values() for r in arr if r.get('status') == 'success')
                fail = sum(1 for arr in results_by_dataset.values() for r in arr if r.get('status') != 'success')
                print(f'Progress {completed}/{len(pending)} new tasks; total success={ok}; failed={fail}', flush=True)

    summary_rows = []
    for ds in args.datasets:
        # Include results for all tasks in the deterministic source task order.
        by_id = {r['task_id']: r for r in results_by_dataset[ds]}
        ordered = [by_id[t['task_id']] for t in all_tasks if t['dataset'] == ds and t['task_id'] in by_id]
        write_dataset_outputs(ds, all_data[ds], ordered, OUT_ROOT)
        success = sum(1 for r in ordered if r.get('status') == 'success')
        failed = len(ordered) - success
        summary_rows.append({'dataset': ds, 'source_records': len(all_data[ds]),
                             'unique_tasks': task_counts[ds], 'success': success, 'failed': failed,
                             'finite_unique_predictions': success})
        print(f'[{ds}] success={success}/{task_counts[ds]}, failed={failed}')

    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / 'run_summary.csv', index=False)
    (OUT_ROOT / 'run_completion.json').write_text(json.dumps({
        'completed_at': now(), 'elapsed_seconds': round(time.time() - start, 2),
        'summary': summary_rows,
    }, indent=2) + '\n', encoding='utf-8')
    print(f'\nFinal outputs: {OUT_ROOT}')
    return 0 if all(x['failed'] == 0 for x in summary_rows) else 3


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare FoldX-compatible M1101 inputs from the completed insertion-code audit.

This is a PREPARATION-ONLY script. It does not execute FoldX and never edits
source WT PDBs, source mutation TXT files, or existing prediction CSVs.

For each of the 35 audited failed tasks it:
  1. validates the audit tables against M1101.csv and the original mutation TXT;
  2. creates a separate WT PDB copy with standard amino-acid residues numbered
     sequentially within each chain and insertion codes cleared;
  3. creates a matching FoldX Individual List using the mapped positions;
  4. omits exact no-op substitutions (WT amino acid == mutant amino acid),
     while recording them explicitly in the task manifest and mapping table;
  5. validates the written PDB and mutation list before marking a task READY.

No FoldX command is run. Existing output files are not overwritten unless
--overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from Bio.PDB import PDBParser

BASE = Path('/data/data_split/splits/M1101')
WT_DIR = BASE / 'wt'
CSV_PATH = Path('/data/data_split/raw/M1101.csv')
TXT_DIR = Path('/data/external/M1101/pdb/txt')
AUDIT_DIR = BASE / 'foldx_recovery_mapping_audit'
OUT_DIR = BASE / 'foldx_recovery_inputs'

AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M',
}
TOKEN_RE = re.compile(r'^([A-Z])([A-Z0-9])(\d+)([A-Za-z]?)([A-Z])$')
ATOM_LIKE_RECORDS = {'ATOM', 'HETATM', 'ANISOU', 'SIGATM', 'SIGUIJ'}
WATER_NAMES = {'HOH', 'WAT', 'DOD'}


def norm_text(x) -> str:
    if pd.isna(x):
        return ''
    return str(x).strip()


def parse_partners(value: str) -> Tuple[str, str]:
    value = norm_text(value)
    if value.count('_') != 1:
        raise ValueError(f'Invalid Partners value: {value!r}')
    g1, g2 = value.split('_')
    g1 = re.sub(r'[\s,;:#|/+]+', '', g1)
    g2 = re.sub(r'[\s,;:#|/+]+', '', g2)
    if not g1 or not g2 or set(g1) & set(g2):
        raise ValueError(f'Invalid/overlapping partner groups: {value!r}')
    return g1, g2


def parse_original_tokens(txt_path: Path) -> List[str]:
    raw = txt_path.read_text(encoding='utf-8-sig').strip().rstrip(';').strip()
    tokens = [x.strip() for x in raw.split(',') if x.strip()]
    if not tokens:
        raise ValueError(f'Empty mutation list: {txt_path}')
    for token in tokens:
        if not TOKEN_RE.fullmatch(token):
            raise ValueError(f'Cannot parse original mutation token {token!r} in {txt_path}')
    return tokens


def pdb_key_from_line(line: str):
    if len(line) < 27:
        return None
    try:
        chain = line[21:22].strip()
        resseq = int(line[22:26])
        icode = line[26:27].strip().upper()
    except Exception:
        return None
    return chain, resseq, icode


def _rewrite_reference_record(line: str, rec: str, mapping) -> Tuple[str, bool]:
    """Update residue references in common PDB connectivity/annotation records."""
    # Each tuple is (chain slice, resSeq slice, iCode slice, resName slice).
    layouts = {
        'SSBOND': [(15,16,17,21,21,22,11,14), (29,30,31,35,35,36,25,28)],
        'CISPEP': [(15,16,17,21,21,22,11,14), (29,30,31,35,35,36,25,28)],
        'LINK':   [(21,22,22,26,26,27,17,20), (51,52,52,56,56,57,47,50)],
        'MODRES': [(16,17,18,22,22,23,12,15)],
    }
    if rec not in layouts:
        return line, False
    changed = False
    for chain_a, chain_b, seq_a, seq_b, icode_a, icode_b, name_a, name_b in layouts[rec]:
        if len(line) < max(seq_b, icode_b, name_b):
            continue
        chain = line[chain_a:chain_b].strip()
        try:
            old_num = int(line[seq_a:seq_b])
        except ValueError:
            continue
        icode = line[icode_a:icode_b].strip().upper()
        key = (chain, old_num, icode)
        mapped = mapping.get(key)
        if mapped is None:
            continue
        new_num, expected_resname, _ = mapped
        record_resname = line[name_a:name_b].strip().upper()
        if record_resname and record_resname != expected_resname:
            raise ValueError(f'{rec} residue-name mismatch for {key}: {record_resname} != {expected_resname}')
        line = line[:seq_a] + f'{new_num:4d}' + line[seq_b:icode_a] + ' ' + line[icode_b:]
        changed = True
    return line, changed


def rewrite_pdb(source: Path, destination: Path, map_rows: pd.DataFrame) -> dict:
    """Rewrite mapped amino-acid residue numbers; preserve all other PDB lines.

    ATOM records that cannot be mapped are fatal. Unmapped HETATM records are
    retained unchanged, but any collision with a newly numbered polymer
    residue is fatal rather than silently changing ligand/water identifiers.
    """
    # map: (chain, old_resseq, old_icode) -> (new_resseq, resname, aa)
    mapping: Dict[Tuple[str, int, str], Tuple[int, str, str]] = {}
    for _, r in map_rows.iterrows():
        key = (norm_text(r['chain']), int(r['original_resseq']), norm_text(r['insertion_code']).upper())
        val = (int(r['renumbered_resseq']), norm_text(r['resname']).upper(), norm_text(r['one_letter']).upper())
        if key in mapping and mapping[key] != val:
            raise ValueError(f'Duplicate/conflicting residue mapping for {source.name}: {key}')
        mapping[key] = val

    # Output residue IDs for mapped polymer residues, used to detect HETATM collisions.
    mapped_output_ids = {(chain, new_num, '') for (chain, _, _), (new_num, _, _) in mapping.items()}
    output_lines: List[str] = []
    mapped_atom_residues = set()
    unmapped_het = set()
    unmapped_atom = set()
    changed_records = 0
    changed_reference_records = defaultdict(int)

    with source.open('r', encoding='ascii', errors='replace') as handle:
        for line_no, line in enumerate(handle, start=1):
            rec = line[0:6].strip().upper() if len(line) >= 6 else ''
            if rec in {'SSBOND', 'CISPEP', 'LINK', 'MODRES'}:
                line, ref_changed = _rewrite_reference_record(line, rec, mapping)
                if ref_changed:
                    changed_reference_records[rec] += 1
            if rec in ATOM_LIKE_RECORDS:
                key = pdb_key_from_line(line)
                if key is None:
                    raise ValueError(f'Malformed {rec} record at {source}:{line_no}')
                chain, old_num, old_icode = key
                resname = line[17:20].strip().upper()
                m = mapping.get(key)
                if m is not None:
                    new_num, expected_resname, _ = m
                    if resname != expected_resname:
                        raise ValueError(
                            f'Residue-name mismatch at {source.name}:{line_no}: '
                            f'{resname} != audit {expected_resname}, key={key}'
                        )
                    if not (1 <= new_num <= 9999):
                        raise ValueError(f'New residue number out of PDB range: {new_num}')
                    # PDB fixed columns: residue sequence [22:26], insertion code [26].
                    line = line[:22] + f'{new_num:4d}' + ' ' + line[27:]
                    changed_records += 1
                    if rec in {'ATOM', 'HETATM'}:
                        mapped_atom_residues.add((chain, new_num, resname))
                elif rec == 'ATOM':
                    unmapped_atom.add((chain, old_num, old_icode, resname))
                elif rec == 'HETATM':
                    # Non-polymer HETATM is retained, but cannot collide with renumbered AA.
                    if (chain, old_num, old_icode) in mapped_output_ids:
                        raise ValueError(
                            f'Unmapped HETATM collides with renumbered amino-acid residue: '
                            f'{source.name} chain={chain!r} res={old_num}{old_icode} {resname}'
                        )
                    if resname not in WATER_NAMES:
                        unmapped_het.add((chain, old_num, old_icode, resname))
            elif rec == 'TER' and len(line) >= 27:
                key = pdb_key_from_line(line)
                if key in mapping:
                    new_num, _, _ = mapping[key]
                    line = line[:22] + f'{new_num:4d}' + ' ' + line[27:]
                    changed_records += 1
            output_lines.append(line)

    if unmapped_atom:
        preview = sorted(unmapped_atom)[:10]
        raise ValueError(f'Unmapped ATOM residues in {source.name}: {preview}')

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='ascii', errors='replace', newline='') as handle:
        handle.writelines(output_lines)

    # Re-parse and verify every audited amino-acid residue exists at its new ID.
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(destination.stem, str(destination))
    model = next(structure.get_models())
    observed = {}
    for chain_obj in model:
        for residue in chain_obj:
            resname = residue.resname.strip().upper()
            aa = AA3_TO_1.get(resname)
            if aa is None:
                continue
            hetflag, resseq, icode = residue.id
            key = (chain_obj.id.strip(), int(resseq), (icode or '').strip().upper())
            if key in observed and observed[key] != (resname, aa):
                raise ValueError(f'Duplicate output residue ID in {destination.name}: {key}')
            observed[key] = (resname, aa)

    missing_after = []
    for (chain, _, _), (new_num, resname, aa) in mapping.items():
        key = (chain, new_num, '')
        if observed.get(key) != (resname, aa):
            missing_after.append((key, (resname, aa), observed.get(key)))
    if missing_after:
        raise ValueError(f'Post-write PDB validation failed for {destination.name}: {missing_after[:10]}')

    return {
        'mapped_residue_count': len(mapping),
        'changed_record_count': changed_records,
        'changed_reference_records': dict(changed_reference_records),
        'unmapped_nonwater_hetatm_count': len(unmapped_het),
        'unmapped_nonwater_hetatm': sorted([list(x) for x in unmapped_het]),
        'output_atom_residue_count': len(observed),
    }


def prepare_task(task_id: str, source_row: pd.Series, summary_row: pd.Series,
                 detail: pd.DataFrame, chain_map: pd.DataFrame, overwrite: bool) -> dict:
    pdb_id = norm_text(source_row['PDB'])
    wt_path = WT_DIR / f'{pdb_id}.pdb'
    txt_path = TXT_DIR / f'individual_list_{task_id}.txt'
    task_out = OUT_DIR / task_id
    out_pdb = task_out / 'wt_renumbered.pdb'
    out_txt = task_out / 'individual_list.txt'
    out_original_txt = task_out / 'original_individual_list.txt'
    out_mapping = task_out / 'mutation_mapping.csv'
    out_manifest = task_out / 'manifest.json'

    if not wt_path.is_file():
        raise FileNotFoundError(f'WT PDB missing: {wt_path}')
    if not txt_path.is_file():
        raise FileNotFoundError(f'Mutation TXT missing: {txt_path}')

    existing = [p for p in (out_pdb, out_txt, out_original_txt, out_mapping, out_manifest) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output exists for {task_id}; use --overwrite to replace: {existing}')
    if overwrite and task_out.exists():
        shutil.rmtree(task_out)
    task_out.mkdir(parents=True, exist_ok=True)

    original_tokens = parse_original_tokens(txt_path)
    task_detail = detail[detail['task_id'].astype(str) == task_id].copy()
    if len(task_detail) != len(original_tokens):
        raise ValueError(f'{task_id}: audit detail rows ({len(task_detail)}) != TXT tokens ({len(original_tokens)})')
    if set(task_detail['txt_token'].astype(str)) != set(original_tokens):
        raise ValueError(f'{task_id}: audit tokens do not exactly match original TXT tokens')

    # Validate source CSV rows and partner chain identities.
    g1, g2 = parse_partners(source_row['Partners'])
    available_chains = set(chain_map['chain'].fillna('').astype(str))
    missing_chains = sorted((set(g1) | set(g2)) - available_chains)
    if missing_chains:
        raise ValueError(f'{task_id}: partner chains absent from WT mapping: {missing_chains}')

    # Validate every mapping and construct the FoldX Individual List.
    mapped_tokens: List[str] = []
    detail_rows = []
    no_ops = []
    seen_positions = set()
    for _, r in task_detail.iterrows():
        token = norm_text(r['txt_token'])
        if norm_text(r['status']) != 'MAPPED':
            raise ValueError(f'{task_id}: non-MAPPED mutation row {token}: {r["status"]}')
        wt_aa = norm_text(r['wt_aa']).upper()
        pdb_aa = norm_text(r['pdb_one_letter']).upper()
        mut_aa = norm_text(r['mut_aa']).upper()
        chain = norm_text(r['chain'])
        new_num = int(r['renumbered_resseq'])
        if wt_aa != pdb_aa:
            raise ValueError(f'{task_id}: WT amino-acid mismatch for {token}: {wt_aa} vs {pdb_aa}')
        if not (1 <= new_num <= 9999):
            raise ValueError(f'{task_id}: mapped position out of range for {token}: {new_num}')
        pos = (chain, new_num)
        if pos in seen_positions:
            raise ValueError(f'{task_id}: duplicate mutation position after mapping: {pos}')
        seen_positions.add(pos)
        is_noop = wt_aa == mut_aa
        mapped_token = f'{wt_aa}{chain}{new_num}{mut_aa}'
        if is_noop:
            no_ops.append({'original_token': token, 'mapped_token': mapped_token, 'reason': 'WT amino acid equals mutant amino acid'})
        else:
            mapped_tokens.append(mapped_token)
        row = r.to_dict()
        row.update({'mapped_token': mapped_token, 'included_in_foldx_list': not is_noop,
                    'omission_reason': 'no_op_substitution' if is_noop else ''})
        detail_rows.append(row)

    if not mapped_tokens:
        raise ValueError(f'{task_id}: no effective mutations remain after no-op filtering')

    # Generate PDB copy from the exact mapping table for this PDB.
    pdb_map_rows = chain_map[chain_map['pdb'].astype(str) == pdb_id].copy()
    if pdb_map_rows.empty:
        raise ValueError(f'No per-chain residue map for PDB {pdb_id}')
    pdb_report = rewrite_pdb(wt_path, out_pdb, pdb_map_rows)

    # Confirm all mutation targets exist in the rewritten PDB with expected WT residue.
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(out_pdb.stem, str(out_pdb))
    model = next(structure.get_models())
    observed = {}
    for chain_obj in model:
        for residue in chain_obj:
            aa = AA3_TO_1.get(residue.resname.strip().upper())
            if aa:
                observed[(chain_obj.id.strip(), int(residue.id[1]))] = aa
    for _, r in task_detail.iterrows():
        key = (norm_text(r['chain']), int(r['renumbered_resseq']))
        expected = norm_text(r['pdb_one_letter']).upper()
        if observed.get(key) != expected:
            raise ValueError(f'{task_id}: output mutation target validation failed at {key}: expected {expected}, observed {observed.get(key)}')

    out_txt.write_text(','.join(mapped_tokens) + ';\n', encoding='ascii')
    shutil.copy2(txt_path, out_original_txt)
    pd.DataFrame(detail_rows).to_csv(out_mapping, index=False)

    manifest = {
        'task_id': task_id,
        'pdb': pdb_id,
        'partners': norm_text(source_row['Partners']),
        'partner_group_1': g1,
        'partner_group_2': g2,
        'source_wt_pdb': str(wt_path),
        'source_mutation_txt': str(txt_path),
        'output_wt_pdb': str(out_pdb),
        'output_mutation_txt': str(out_txt),
        'original_mutation_count': len(original_tokens),
        'effective_mutation_count': len(mapped_tokens),
        'omitted_no_op_mutations': no_ops,
        'mapped_mutations': mapped_tokens,
        'pdb_rewrite_validation': pdb_report,
        'status': 'READY_FOR_FOLDX_PREFLIGHT',
        'foldx_executed': False,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return {
        'task_id': task_id, 'pdb': pdb_id, 'status': 'READY_FOR_FOLDX_PREFLIGHT',
        'original_mutations': len(original_tokens), 'effective_mutations': len(mapped_tokens),
        'no_op_removed': len(no_ops), 'mapped_residues': pdb_report['mapped_residue_count'],
        'unmapped_hetatm': pdb_report['unmapped_nonwater_hetatm_count'],
        'output_dir': str(task_out),
    }


def main():
    ap = argparse.ArgumentParser(description='Prepare renumbered M1101 FoldX inputs; does not run FoldX.')
    ap.add_argument('--overwrite', action='store_true', help='Replace only this script output directory for each task.')
    ap.add_argument('--limit', type=int, default=None, help='Prepare only the first N tasks for a pilot.')
    args = ap.parse_args()

    required = [CSV_PATH, AUDIT_DIR/'task_mapping_summary.csv', AUDIT_DIR/'residue_mapping_detail.csv', AUDIT_DIR/'per_chain_renumbering_map.csv']
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f'Required input not found: {p}')

    source = pd.read_csv(CSV_PATH, dtype=str)
    for col in ('ID', 'PDB', 'Partners'):
        if col not in source.columns:
            raise ValueError(f'M1101.csv missing required column {col!r}')
    summary = pd.read_csv(AUDIT_DIR/'task_mapping_summary.csv', dtype=str).fillna('')
    detail = pd.read_csv(AUDIT_DIR/'residue_mapping_detail.csv', dtype=str).fillna('')
    chain_map = pd.read_csv(AUDIT_DIR/'per_chain_renumbering_map.csv', dtype=str).fillna('')

    # Only tasks explicitly present in the completed audit and not marked for mapping review.
    tasks = summary[summary['status'].isin(['OK', 'MAPPED_WITH_NO_OP_MUTATION_REVIEW'])].copy()
    if args.limit is not None:
        tasks = tasks.head(args.limit)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    for _, srow in tasks.iterrows():
        task_id = norm_text(srow['task_id'])
        rows = source[source['ID'].astype(str).str.strip() == task_id]
        if rows.empty:
            failures.append({'task_id': task_id, 'error': 'ID not found in M1101.csv'})
            continue
        # Duplicate source IDs are permitted only if structural fields agree.
        for col in ('PDB', 'Partners', 'Mutation'):
            if col in rows.columns and rows[col].fillna('').astype(str).str.strip().nunique() > 1:
                failures.append({'task_id': task_id, 'error': f'Duplicate source rows disagree in {col}'})
                break
        else:
            try:
                results.append(prepare_task(task_id, rows.iloc[0], srow, detail, chain_map, args.overwrite))
            except Exception as exc:
                failures.append({'task_id': task_id, 'error': str(exc)})

    if results:
        pd.DataFrame(results).to_csv(OUT_DIR/'preparation_summary.csv', index=False)
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR/'preparation_failures.csv', index=False)

    print('M1101 FoldX renumbered-input preparation (NO FOLDX EXECUTION)')
    print(f'Source CSV : {CSV_PATH}')
    print(f'WT folder  : {WT_DIR}')
    print(f'Audit dir  : {AUDIT_DIR}')
    print(f'TXT folder : {TXT_DIR}')
    print(f'Output     : {OUT_DIR}')
    print(f'Tasks selected: {len(tasks)}')
    print(f'Prepared: {len(results)}; failed: {len(failures)}')
    for r in results:
        print(f"{r['status']:30s} {r['task_id']} | mutations {r['original_mutations']}->{r['effective_mutations']} | no-op={r['no_op_removed']}")
    if failures:
        print('\nFAILURES:')
        for f in failures:
            print(f"{f['task_id']}: {f['error']}")
    print('\nNo FoldX command was executed.')
    print('Source WT PDBs, source TXT files, and existing prediction CSVs were not modified.')
    if failures:
        sys.exit(2)


if __name__ == '__main__':
    main()

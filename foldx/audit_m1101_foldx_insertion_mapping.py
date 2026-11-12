#!/usr/bin/env python3
"""Read-only audit of FoldX mutation positions containing PDB insertion codes.

This script does NOT modify WT PDBs, mutation TXT files, or run FoldX.
It maps each original (chain, residue number, insertion code) to a sequential
per-chain residue number that can be used in a separate FoldX-compatible copy.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
from Bio.PDB import PDBParser

BASE = Path('/data/data_split/splits/M1101')
WT_DIR = BASE / 'wt'
TXT_DIR = Path('/data/external/M1101/pdb/txt')
OUT_DIR = BASE / 'foldx_recovery_mapping_audit'

PDB_IDS = ['HM_3BN9', '1N8Z', '2NY7', '3BDY', '3BE1', '3BN9']

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
    '3BDY_DL27AA', '3BDY_IL27BA', '3BDY_PL27CA', '3BDY_RL27DA', '3BDY_YH100AA',
    '3BE1_DL27AA', '3BE1_IL27BA', '3BE1_PL27CA', '3BE1_RL27DA', '3BE1_YH100AA',
    '3BN9_DA60AA', '3BN9_DA60BA', '3BN9_FA60EA', '3BN9_QA221AA',
    '3BN9_QH100AV', '3BN9_RA60CA', '3BN9_RA60FA', '3BN9_YA60GA',
    'HM_3BN9_QH100AV',
]

AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'MSE':'M',
}
# FoldX Individual List token: WT amino acid + chain + integer position + mutant amino acid.
# Existing source tokens encode insertion code between the integer position and mutant aa.
TOKEN_RE = re.compile(r'^([A-Z])([A-Z0-9])(\d+)([A-Za-z]?)([A-Z])$')


def parse_tokens(txt_path: Path):
    raw = txt_path.read_text(encoding='utf-8-sig').strip()
    raw = raw.rstrip(';').strip()
    if not raw:
        return raw, []
    tokens = [x.strip() for x in raw.split(',') if x.strip()]
    parsed = []
    for token in tokens:
        m = TOKEN_RE.fullmatch(token)
        if not m:
            parsed.append({'token': token, 'parse_status': 'PARSE_FAIL'})
            continue
        wt, chain, resseq, icode, mut = m.groups()
        parsed.append({
            'token': token,
            'wt_aa': wt,
            'chain': chain,
            'pdb_resseq': int(resseq),
            'icode': (icode or '').upper(),
            'mut_aa': mut,
            'parse_status': 'OK',
        })
    return raw, parsed


def build_residue_maps(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    exact = {}
    sequential = {}
    chain_counts = defaultdict(int)
    residue_rows = []

    for chain in model:
        for residue in chain:
            resname = residue.resname.strip().upper()
            aa = AA3_TO_1.get(resname)
            if aa is None:
                continue
            hetflag, resseq, icode = residue.id
            icode = (icode or '').strip().upper()
            chain_counts[chain.id] += 1
            new_resseq = chain_counts[chain.id]
            key = (chain.id, int(resseq), icode)
            exact[key] = (aa, resname, new_resseq)
            sequential[(chain.id, new_resseq)] = key
            residue_rows.append({
                'pdb': pdb_path.stem,
                'chain': chain.id,
                'original_resseq': int(resseq),
                'insertion_code': icode,
                'resname': resname,
                'one_letter': aa,
                'renumbered_resseq': new_resseq,
                'hetero_flag': hetflag,
            })
    return exact, residue_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = PDBParser(QUIET=True)
    all_rows = []
    task_rows = []
    pdb_maps = {}

    for task_id in FAILED_IDS:
        pdb_id = next((p for p in sorted(PDB_IDS, key=len, reverse=True) if task_id.startswith(p + '_')), None)
        if pdb_id is None:
            task_rows.append({'task_id': task_id, 'status': 'PDB_ID_PARSE_FAIL'})
            continue
        pdb_path = WT_DIR / f'{pdb_id}.pdb'
        txt_path = TXT_DIR / f'individual_list_{task_id}.txt'
        task_status = 'OK'
        if not pdb_path.is_file():
            task_rows.append({'task_id': task_id, 'pdb': pdb_id, 'status': 'WT_MISSING'})
            continue
        if not txt_path.is_file():
            task_rows.append({'task_id': task_id, 'pdb': pdb_id, 'status': 'TXT_MISSING'})
            continue
        if pdb_id not in pdb_maps:
            try:
                exact, residue_rows = build_residue_maps(pdb_path)
                pdb_maps[pdb_id] = (exact, residue_rows)
            except Exception as exc:
                task_rows.append({'task_id': task_id, 'pdb': pdb_id, 'status': f'PDB_PARSE_FAIL:{exc}'})
                continue
        exact, _ = pdb_maps[pdb_id]
        raw, parsed = parse_tokens(txt_path)
        mapped_tokens = []
        task_insertion_codes = 0
        task_noops = 0
        task_failures = []

        for item in parsed:
            row = {'task_id': task_id, 'pdb': pdb_id, 'txt_token': item['token']}
            if item.get('parse_status') != 'OK':
                row.update({'status': item.get('parse_status')})
                all_rows.append(row)
                task_failures.append(item['token'])
                continue

            chain = item['chain']
            resseq = item['pdb_resseq']
            icode = item['icode']
            if icode:
                task_insertion_codes += 1
            hit = exact.get((chain, resseq, icode))
            if hit is None:
                row.update({
                    'wt_aa': item['wt_aa'], 'chain': chain,
                    'pdb_resseq': resseq, 'insertion_code': icode,
                    'mut_aa': item['mut_aa'], 'status': 'PDB_RESIDUE_NOT_FOUND',
                })
                all_rows.append(row)
                task_failures.append(item['token'])
                continue

            pdb_aa, pdb_resname, new_resseq = hit
            wt_match = pdb_aa == item['wt_aa']
            noop = item['wt_aa'] == item['mut_aa']
            status = 'MAPPED' if wt_match else 'WT_AA_MISMATCH'
            if noop:
                task_noops += 1
            if wt_match:
                mapped_tokens.append(f"{item['wt_aa']}{chain}{new_resseq}{item['mut_aa']}")
            else:
                task_failures.append(item['token'])

            row.update({
                'wt_aa': item['wt_aa'], 'chain': chain,
                'pdb_resseq': resseq, 'insertion_code': icode,
                'pdb_resname': pdb_resname, 'pdb_one_letter': pdb_aa,
                'renumbered_resseq': new_resseq, 'mut_aa': item['mut_aa'],
                'is_noop': noop, 'status': status,
            })
            all_rows.append(row)

        if task_failures:
            task_status = 'MAPPING_REQUIRES_REVIEW'
        elif task_noops:
            task_status = 'MAPPED_WITH_NO_OP_MUTATION_REVIEW'
        task_rows.append({
            'task_id': task_id,
            'pdb': pdb_id,
            'status': task_status,
            'mutation_count': len(parsed),
            'insertion_code_count': task_insertion_codes,
            'no_op_count': task_noops,
            'mapped_individual_list_preview': ','.join(mapped_tokens) + ';' if mapped_tokens else '',
            'original_txt': raw,
        })

    pd.DataFrame(all_rows).to_csv(OUT_DIR / 'residue_mapping_detail.csv', index=False)
    pd.DataFrame(task_rows).to_csv(OUT_DIR / 'task_mapping_summary.csv', index=False)

    residue_rows_all = []
    for _, residue_rows in pdb_maps.values():
        residue_rows_all.extend(residue_rows)
    pd.DataFrame(residue_rows_all).to_csv(OUT_DIR / 'per_chain_renumbering_map.csv', index=False)

    print('M1101 FoldX insertion-code mapping audit (READ ONLY)')
    print(f'WT directory : {WT_DIR}')
    print(f'TXT directory: {TXT_DIR}')
    print(f'Output       : {OUT_DIR}')
    print(f'Unique tasks : {len(FAILED_IDS)}')
    print('\nTask summary:')
    for row in task_rows:
        print(f"{row.get('status','?'):38s} {row.get('task_id','')}")
    print('\nFiles written:')
    print(f'  {OUT_DIR / "task_mapping_summary.csv"}')
    print(f'  {OUT_DIR / "residue_mapping_detail.csv"}')
    print(f'  {OUT_DIR / "per_chain_renumbering_map.csv"}')
    print('\nNo WT PDB, mutation TXT, or existing FoldX result was modified.')
    print('No FoldX command was executed.')


if __name__ == '__main__':
    main()

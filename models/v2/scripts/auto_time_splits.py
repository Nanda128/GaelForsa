from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd


def _detect_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def _read_one(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        # Support both comma-separated and semicolon-separated raw exports.
        return pd.read_csv(path, sep=None, engine='python')
    return pd.read_parquet(path)


def _find_raw_files(raw_dir: str) -> list[str]:
    files: list[str] = []
    for root, _, fns in os.walk(raw_dir):
        for fn in fns:
            if fn.endswith((".csv", ".parquet", ".pq")):
                files.append(os.path.join(root, fn))
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser(description='Suggest auto time split config; no manual ranges required.')
    ap.add_argument('--raw_dir', default='data/raw')
    ap.add_argument('--ratios', default='0.7,0.15,0.15', help='train,val,test ratios')
    args = ap.parse_args()

    ratios = [float(x.strip()) for x in args.ratios.split(',')]
    if len(ratios) != 3:
        raise ValueError('ratios must have 3 comma-separated values')

    if not os.path.isdir(args.raw_dir):
        raise ValueError(f'raw_dir not found: {args.raw_dir}')

    files = _find_raw_files(args.raw_dir)
    if not files:
        raise ValueError(f'No raw files found in {args.raw_dir}')

    rows = []
    for p in sorted(files):
        df = _read_one(p)
        tcol = _detect_col(list(df.columns), ['timestamp', 'time_stamp', 'time', 'datetime'])
        idcol = _detect_col(list(df.columns), ['turbine_id', 'asset_id', 'asset', 'unit_id', 'unit'])
        if tcol is None:
            continue
        if idcol is None:
            idcol = '__file__'
            df[idcol] = os.path.basename(p)
        tt = pd.to_datetime(df[tcol], errors='coerce')
        m = tt.notna()
        g = df.loc[m].groupby(idcol)
        for tid, gg in g:
            t = pd.to_datetime(gg[tcol], errors='coerce').dropna().sort_values()
            if len(t) == 0:
                continue
            rows.append(
                {
                    'source_file': os.path.basename(p),
                    'turbine_id': str(tid),
                    'start': t.iloc[0],
                    'end': t.iloc[-1],
                    'n_rows': int(len(t)),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError('No parseable timestamps found in raw files')

    print('Detected per-file/per-turbine time spans:')
    print(out.to_string(index=False))
    print()
    print('Use these overrides (no manual time ranges needed):')
    print(f"data.splits.method=time_auto data.splits.ratios=[{ratios[0]},{ratios[1]},{ratios[2]}]")


if __name__ == '__main__':
    main()

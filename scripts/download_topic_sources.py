#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_topic_sources.py

Baixa as fontes públicas (Hugging Face, formato Parquet) usadas na construção do
dataset de classificação temática (gatekeeper de saúde do chatbot).

Os arquivos são salvos em data/raw/<nome>.parquet (diretório ignorado pelo git).
Downloads já existentes são pulados (use --force para rebaixar).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import requests

HF = 'https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet'

# nome_local -> (repo, [splits])
SOURCES = {
    'folha': ('emdemor/news-of-the-brazilian-newspaper', ['train']),
    'fakerecogna': ('recogna-nlp/FakeRecogna', ['train']),
    'mumin_pt': ('ju-resplande/MuMiN-PT', ['train']),
    'medpt': ('AKCIT/MedPT', ['train']),
    'mental_health_pt': ('orion-research/mental-health-pt_BR', ['train']),
    'pt_tweets': ('fpaulino/portuguese-tweets', ['train', 'validation', 'test']),
    'portuguesechat': ('rishiraj/portuguesechat', ['train', 'test']),
    'olid_br': ('dougtrajano/olid-br', ['train', 'test']),
    'told_br': ('mteb/told-br', ['train']),
    'hatebr': ('franciellevargas/HateBR', ['train']),
    'b2w_reviews': ('ruanchaves/b2w-reviews01', ['train']),
}


def download(name, repo, split, out_dir: Path, force: bool):
    dest = out_dir / f'{name}__{split}.parquet'
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return f'[skip] {dest.name}'
    url = HF.format(repo=repo, split=split)
    tmp = dest.with_suffix('.part')
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return f'[ok]   {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)'


def main():
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(repo_root / 'data' / 'raw'))
    ap.add_argument('--only', nargs='*', help=f'Subconjunto de fontes: {", ".join(SOURCES)}')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(n, repo, s) for n, (repo, splits) in SOURCES.items()
            if not args.only or n in args.only for s in splits]

    errors = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(download, n, repo, s, out_dir, args.force) for n, repo, s in jobs]
        for (n, _, s), f in zip(jobs, futs):
            try:
                print(f.result())
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f'[erro] {n}/{s}: {e}', file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()

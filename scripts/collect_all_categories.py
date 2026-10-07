#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_all_categories.py

Orquestrador para coletar amostras balanceadas de cada tema a partir de:
  - Boatos.org (para fake news de esporte, entretenimento, tecnologia, politica)
  - Agência Brasil / EBC (para notícias verdadeiras de esportes, cultura, economia, geral, politica)

Executa de forma limpa e sequencial com workers paralelos.
"""

import sys
import time
from pathlib import Path
from scrape_boatos_categories import scrape_category as scrape_boatos
from scrape_ebc_categories import scrape_category as scrape_ebc

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"

BOATOS_TARGETS = [
    ("esporte", 250),
    ("entretenimento", 250),
    ("tecnologia", 250),
    ("politica", 250),
]

EBC_TARGETS = [
    ("esportes", 250),
    ("cultura", 250),
    ("economia", 250),
    ("geral", 250),
    ("politica", 250),
]


def run_collection():
    print("=" * 75)
    print("INICIANDO COLETA BALANCEADA DE CATEGORIAS TEMÁTICAS")
    print("=" * 75)

    # 1. Coleta Boatos.org
    for cat, limit in BOATOS_TARGETS:
        out = DATASETS_DIR / f"boatos_{cat}.tsv"
        print(f"\n>>> [Boatos.org] Processando categoria: {cat} (meta: {limit})...")
        try:
            scrape_boatos(category=cat, output_file=out, limit=limit, workers=8)
        except Exception as e:
            print(f"Erro ao coletar boatos_{cat}: {e}")
        time.sleep(1.0)

    # 2. Coleta EBC
    for cat, limit in EBC_TARGETS:
        out = DATASETS_DIR / f"ebc_{cat}.tsv"
        print(f"\n>>> [EBC Agência Brasil] Processando categoria: {cat} (meta: {limit})...")
        try:
            scrape_ebc(category=cat, output_file=out, limit=limit, workers=8)
        except Exception as e:
            print(f"Erro ao coletar ebc_{cat}: {e}")
        time.sleep(1.0)

    print("\n" + "=" * 75)
    print("[CONCLUÍDO] Todas as coletas temáticas foram finalizadas com sucesso!")
    print("=" * 75)


if __name__ == "__main__":
    run_collection()


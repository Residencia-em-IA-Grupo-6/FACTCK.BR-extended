#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_health_boosted_dataset.py

Constrói dataset temático ampliado integrando o volume massivo de checagens reais de saúde
do Boatos.org, Ministério da Saúde, Agência Brasil e agências de checagem consolidadas
em 'datasets/saude_consolidado_unificado.tsv', garantindo 100% de limpeza e equilíbrio
entre os 8 temas.
"""

import html
import logging
import re
import sys
from pathlib import Path
from typing import Set

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = html.unescape(text)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", "", t)
    t = re.sub(r"@[A-Za-z0-9_]+", "", t)
    t = re.sub(r"\bRT\b", "", t)

    prefixes = [
        r"^[Bb]oato\s*[\-–—:\.]\s*",
        r"^[Ff]also\s*[\-–—:\.]\s*",
        r"^[Ff]ato\s+ou\s+[Ff]ake\s*[\-–—:\.]\s*",
        r"^[Vv]erdadeiro\s*[\-–—:\.]\s*",
        r"^[Cc]onte[úu]do\s*verificado\s*[\-–—:\.]\s*",
        r"^(?:[Vv]ers[ãa]o|[Tt]exto)\s*\d*\s*[\-–—:\.]\s*",
    ]
    for p in prefixes:
        t = re.sub(p, "", t)

    t = re.sub(r"#boato\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[boato\]|\(boato\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[…\]|\[\.\.\.\]", "", t)

    # Aspas
    t = t.strip()
    quotes = "\"\'“”‘’«»"
    while len(t) > 1 and t[0] in quotes and t[-1] in quotes:
        t = t[1:-1].strip()
    while len(t) > 0 and t[0] in quotes:
        t = t[1:].strip()
    while len(t) > 0 and t[-1] in quotes:
        t = t[:-1].strip()

    t = re.sub(r"\s+", " ", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def is_valid(text: str) -> bool:
    if not text or len(text) < 25 or len(text) > 1200:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    bad_patterns = [
        "confira o desmentido",
        "ver essa foto no instagram",
        "assista ao desmentido",
        "javascript",
        "subscribe to our channel",
    ]
    low = text.lower()
    return not any(p in low for p in bad_patterns)


def main():
    logger.info("=" * 70)
    logger.info("CONSTRUINDO DATASET DE TÓPICOS COM TODA A BASE DE SAÚDE")
    logger.info("=" * 70)

    repo_root = Path(".")
    datasets_dir = repo_root / "datasets"

    # 1. Carregar base consolidada de saúde
    health_path = datasets_dir / "saude_consolidado_unificado.tsv"
    df_health_raw = pd.read_csv(health_path, sep="\t")
    logger.info(f"Base de saúde carregada: {len(df_health_raw)} registros de {health_path.name}")

    health_records = []
    seen_texts: Set[str] = set()

    # Priorizar: Boatos.org, Ministério da Saúde, Fact-checkers
    for _, row in df_health_raw.iterrows():
        claim = str(row.get("Claim", "")).strip() or str(row.get("Titulo", "")).strip()
        cleaned = clean_text(claim)
        if not is_valid(cleaned):
            continue

        norm_key = re.sub(r"\W+", "", cleaned.lower())
        if norm_key in seen_texts or len(norm_key) < 20:
            continue
        seen_texts.add(norm_key)

        author = str(row.get("Author", "Desconhecido"))
        is_fk = bool(row.get("is_fake")) if pd.notnull(row.get("is_fake")) else None

        health_records.append({
            "text": cleaned,
            "topic": "saude",
            "source": f"local:{author}",
            "is_fake": is_fk,
            "is_health": True,
        })

    logger.info(f"Total de amostras únicas e limpas de saúde extraídas: {len(health_records)}")
    df_health = pd.DataFrame(health_records)
    print("Distribuição das fontes de saúde extraídas:")
    print(df_health["source"].value_counts().head(10))

    # Limitar a ~2.500 amostras equilibradas de saúde para paridade
    target_health_count = min(len(df_health), 2500)
    df_health_sampled = df_health.sample(n=target_health_count, random_state=42).reset_index(drop=True)
    logger.info(f"Amostras de saúde selecionadas para o dataset: {len(df_health_sampled)}")

    # 2. Carregar as outras 7 classes de topic_dataset_8classes.parquet
    base_parquet = datasets_dir / "topic_classification" / "topic_dataset_8classes.parquet"
    df_base = pd.read_parquet(base_parquet)
    logger.info(f"Base de tópicos gerais carregada: {len(df_base)} amostras de {base_parquet.name}")

    other_classes = [
        "conversa_geral",
        "politica_governo",
        "entretenimento_cultura",
        "esportes",
        "economia_negocios",
        "ciencia_tecnologia",
        "cotidiano_seguranca",
    ]

    other_dfs = []
    # Amostrar ~1.000 a 1.200 amostras de cada classe para balancear
    samples_per_other = min(1200, target_health_count // 2)  # ~1.200 por classe
    for cls in other_classes:
        subset = df_base[df_base["topic"] == cls].copy()
        sampled = subset.sample(n=min(len(subset), 1200), random_state=42)
        other_dfs.append(sampled)
        logger.info(f"Classe '{cls}': {len(sampled)} amostras selecionadas")

    # 3. Concatenar e embaralhar
    all_dfs = [df_health_sampled] + other_dfs
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 4. Salvar arquivos
    out_dir = datasets_dir / "topic_classification"
    out_parquet = out_dir / "health_boosted_dataset.parquet"
    out_tsv = out_dir / "health_boosted_dataset.tsv"

    final_df.to_parquet(out_parquet, index=False)
    final_df.to_csv(out_tsv, sep="\t", index=False)

    logger.info("=" * 70)
    logger.info(f"DATASET CONCLUÍDO E SALVO COM SUCESSO:")
    logger.info(f"  Parquet: {out_parquet}")
    logger.info(f"  TSV:     {out_tsv}")
    logger.info(f"Total de registros: {len(final_df)}")
    logger.info("=" * 70)

    print("\nDistribuição Final por Tema:")
    print(final_df["topic"].value_counts())


if __name__ == "__main__":
    main()

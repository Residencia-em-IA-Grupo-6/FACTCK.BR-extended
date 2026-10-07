#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_theme_dataset.py

Gera o dataset consolidado e balanceado para treinamento de modelos de classificação
de notícias por tema (7 classes):
  - politica
  - saude
  - esporte
  - entretenimento
  - economia
  - tecnologia
  - seguranca_publica

Atributos de saída:
  - Claim: Texto limpo da alegação/notícia (higienizado contra vieses de ML)
  - tema: Rótulo de classe temática (uma das 7 classes)
  - URL: Link permanente do artigo/checagem
  - Titulo: Título original da publicação
  - Author: Agência/veículo de checagem ou notícia
  - is_fake: Booleano de veracidade (True = falso/boato, False = fato/notícia autêntica)
"""

import os
import re
import csv
import html
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from classify_themes import classify_theme

THEMES = [
    "politica",
    "saude",
    "esporte",
    "entretenimento",
    "economia",
    "tecnologia",
    "seguranca_publica",
]


def clean_claim(text: str) -> str:
    """Higieniza o texto da alegação removendo ruídos e vieses de ML."""
    if not text or not isinstance(text, str):
        return ""

    t = html.unescape(text)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+|pic\.twitter\.com/\S+|t\.co/\S+", "", t)

    # Remover prefixos de veredito
    t = re.sub(r"^[Bb]oato\s*[\-–—:\.]\s*", "", t)
    t = re.sub(r"^[Cc]onte[úu]do\s*verificado\s*[\-–—:\.]\s*", "", t)
    t = re.sub(r"^(?:[Vv]ers[ãa]o|[Tt]exto)\s*\d*\s*[\-–—:\.]\s*", "", t)
    t = re.sub(r"^[Ff]also\s*[\-–—:]\s*", "", t)

    t = re.sub(r"#boato\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[boato\]|\(boato\)", "", t, flags=re.IGNORECASE)

    # Emojis
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\u200d\ufe0f]",
        flags=re.UNICODE,
    )
    t = emoji_pattern.sub("", t)

    # Hashtags e menções
    t = re.sub(r"#([A-Za-z0-9_À-ÿ]+)", r"\1", t)
    t = re.sub(r"@\w+", "", t)

    t = re.sub(r"\[…\]|\[\.\.\.\]", "", t)
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"\?{2,}", "?", t)

    t = t.strip()
    quote_chars = "\"\'“”‘’«»"
    while len(t) > 1 and t[0] in quote_chars and t[-1] in quote_chars:
        t = t[1:-1].strip()
    while len(t) > 0 and t[0] in quote_chars:
        t = t[1:].strip()
    while len(t) > 0 and t[-1] in quote_chars:
        t = t[:-1].strip()

    t = re.sub(r"\s+", " ", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]

    return t


def is_invalid_claim_text(s: str) -> bool:
    """Verifica se o texto é marcador técnico web em vez da alegação."""
    if not s or len(s.strip()) < 15:
        return True
    s_low = s.lower()
    bad_patterns = [
        "confira o desmentido",
        "ver essa foto no instagram",
        "assista ao desmentido",
        "assista ao vídeo",
        "data-mce-type",
    ]
    return any(p in s_low for p in bad_patterns)


def map_author(author_str: str) -> str:
    if not author_str:
        return "Desconhecido"
    s = str(author_str).lower()
    if "boatos" in s:
        return "Boatos.org"
    if "ebc" in s or "agência brasil" in s or "agencia brasil" in s:
        return "Agência Brasil"
    if "aosfatos" in s:
        return "Aos Fatos"
    if "lupa" in s:
        return "Agência Lupa"
    if "afp" in s or "checamos" in s:
        return "AFP Checamos"
    if "estadao" in s:
        return "Estadão Verifica"
    if "uol" in s or "bol" in s:
        return "UOL Confere"
    if "comprova" in s:
        return "Projeto Comprova"
    if "apublica" in s:
        return "Agência Pública"
    if "observador" in s:
        return "Observador"
    if "folha" in s:
        return "Folha de S.Paulo"
    if "globo" in s:
        return "O Globo"
    if "tatu" in s:
        return "Agência Tatu"
    if "nexo" in s:
        return "Nexo Jornal"
    if "saúde" in s or "saude" in s:
        return "Ministério da Saúde"
    return str(author_str).strip()


def classify_veracity(alt_name: str) -> bool:
    s = str(alt_name).lower().strip()
    if re.search(
        r"\b(fals[oa]|enganos[oa]|enganador|errad[oa]|distorcido|insustent[aá]vel|sem contexto|fora de contexto|fake|s[aá]tira|imposs[ií]vel provar|impreciso|não é bem assim)\b",
        s,
    ):
        return True
    if re.search(r"\b(verdadeir[oa]|certo|fato|procede|aut[eê]ntico)\b", s):
        return False
    return True


def load_tsv_safe(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append({k: (v or "").strip() for k, v in row.items() if k})
    return records


def build_dataset(
    repo_root: Path,
    output_file: Path,
    max_per_theme: int = 700,
    seed: int = 42,
):
    random.seed(seed)
    datasets_dir = repo_root / "datasets"

    print("=" * 75)
    print("CONSTRUÇÃO DO DATASET DE CLASSIFICAÇÃO DE NOTÍCIAS POR TEMAS")
    print(f"Taxonomia: {THEMES}")
    print(f"Meta máxima por classe: {max_per_theme} amostras")
    print("=" * 75)

    theme_pools: Dict[str, List[Dict]] = {t: [] for t in THEMES}
    seen_urls: Set[str] = set()
    seen_claims: Set[str] = set()

    def add_candidate(item: Dict, theme: str):
        if not theme or theme not in theme_pools:
            return
        url = item.get("URL", "").strip()
        claim = item.get("Claim", "").strip()
        if not claim or len(claim) < 20:
            return
        claim_norm = claim.lower()
        if (url and url in seen_urls) or (claim_norm in seen_claims):
            return

        seen_urls.add(url)
        seen_claims.add(claim_norm)
        theme_pools[theme].append(item)

    # 1. Carregar Base Consolidada de Saúde (datasets/factckbr_boatos_saude_consolidado.tsv)
    health_file = datasets_dir / "factckbr_boatos_saude_consolidado.tsv"
    if health_file.exists():
        print(f"Processando Base Consolidada de Saúde ({health_file.name})...")
        h_rows = load_tsv_safe(health_file)
        # Separar falsos e verdadeiros para balanceamento interno de saúde
        fakes = [r for r in h_rows if str(r.get("is_fake", "")).lower() == "true"]
        trues = [r for r in h_rows if str(r.get("is_fake", "")).lower() == "false"]
        random.shuffle(fakes)
        random.shuffle(trues)

        for r in fakes + trues:
            claim = clean_claim(r.get("Claim", ""))
            add_candidate(
                {
                    "Claim": claim,
                    "tema": "saude",
                    "URL": r.get("URL", ""),
                    "Titulo": r.get("Titulo", ""),
                    "Author": map_author(r.get("Author", "")),
                    "is_fake": str(r.get("is_fake", "")).lower() == "true",
                },
                "saude",
            )
        print(f"  -> Total em 'saude' após ingestão: {len(theme_pools['saude'])}")

    # 2. Carregar Checagens Gerais FACTCK.BR (FACTCKBR_updated.tsv e FACTCKBR_old.tsv)
    for fname in ["FACTCKBR_updated.tsv", "FACTCKBR_old.tsv"]:
        fpath = datasets_dir / fname
        if fpath.exists():
            print(f"\nClassificando e ingerindo {fname}...")
            f_rows = load_tsv_safe(fpath)
            added_count = 0
            for r in f_rows:
                claim_raw = r.get("claimReviewed", "").strip() or r.get("title", "").strip()
                title_raw = r.get("title", "").strip()
                text_to_classify = f"{claim_raw} {title_raw}"
                theme = classify_theme(text_to_classify)
                if theme:
                    claim_clean = clean_claim(claim_raw)
                    is_fake = classify_veracity(r.get("alternativeName", ""))
                    add_candidate(
                        {
                            "Claim": claim_clean,
                            "tema": theme,
                            "URL": r.get("URL", ""),
                            "Titulo": title_raw,
                            "Author": map_author(r.get("Author", "")),
                            "is_fake": is_fake,
                        },
                        theme,
                    )
                    added_count += 1
            print(f"  -> {added_count} registros classificados de {fname}.")

    # 3. Carregar Datasets Temáticos do Boatos.org (is_fake=True)
    boatos_map = {
        "esporte": "esporte",
        "entretenimento": "entretenimento",
        "tecnologia": "tecnologia",
        "politica": "politica",
    }
    for cat_name, target_theme in boatos_map.items():
        bpath = datasets_dir / f"boatos_{cat_name}.tsv"
        if bpath.exists():
            print(f"\nIngerindo Boatos.org ({bpath.name}) para o tema '{target_theme}'...")
            b_rows = load_tsv_safe(bpath)
            b_count = 0
            for r in b_rows:
                c = r.get("Texto_Falso_Original", "").strip()
                if is_invalid_claim_text(c):
                    c = r.get("Resumo_Boato", "").strip()
                if is_invalid_claim_text(c):
                    c = r.get("Titulo", "").strip()
                c_clean = clean_claim(c)

                add_candidate(
                    {
                        "Claim": c_clean,
                        "tema": target_theme,
                        "URL": r.get("URL", ""),
                        "Titulo": r.get("Titulo", ""),
                        "Author": "Boatos.org",
                        "is_fake": True,
                    },
                    target_theme,
                )
                b_count += 1
            print(f"  -> {b_count} matérias adicionadas de {bpath.name}.")

    # 4. Carregar Datasets Temáticos da EBC / Agência Brasil (is_fake=False)
    ebc_map = {
        "esportes": "esporte",
        "cultura": "entretenimento",
        "economia": "economia",
        "politica": "politica",
    }
    for ebc_cat, target_theme in ebc_map.items():
        epath = datasets_dir / f"ebc_{ebc_cat}.tsv"
        if epath.exists():
            print(f"\nIngerindo EBC ({epath.name}) para o tema '{target_theme}'...")
            e_rows = load_tsv_safe(epath)
            e_count = 0
            for r in e_rows:
                tit = r.get("Titulo", "").strip()
                sub = r.get("Subtitulo", "").strip()
                if sub and len(sub) >= 20:
                    claim_raw = f"{tit}. {sub}"
                else:
                    claim_raw = tit or sub
                claim_clean = clean_claim(claim_raw)
                add_candidate(
                    {
                        "Claim": claim_clean,
                        "tema": target_theme,
                        "URL": r.get("URL", ""),
                        "Titulo": tit,
                        "Author": "Agência Brasil",
                        "is_fake": False,
                    },
                    target_theme,
                )
                e_count += 1
            print(f"  -> {e_count} matérias adicionadas de {epath.name}.")

    # 5. Ingerir datasets suplementares da EBC com classificação temática (ebc_geral e ebc_direitos-humanos)
    for supp_file in ["ebc_geral.tsv", "ebc_direitos-humanos.tsv"]:
        supp_path = datasets_dir / supp_file
        if supp_path.exists():
            print(f"\nClassificando e ingerindo {supp_file}...")
            s_rows = load_tsv_safe(supp_path)
            s_count = 0
            for r in s_rows:
                tit = r.get("Titulo", "").strip()
                sub = r.get("Subtitulo", "").strip()
                conteudo_prefix = r.get("Conteudo", "").strip()[:400]
                theme = classify_theme(f"{tit} {sub} {conteudo_prefix}")
                if theme and theme in THEMES:
                    if sub and len(sub) >= 20:
                        claim_raw = f"{tit}. {sub}"
                    else:
                        claim_raw = tit or sub
                    claim_clean = clean_claim(claim_raw)
                    add_candidate(
                        {
                            "Claim": claim_clean,
                            "tema": theme,
                            "URL": r.get("URL", ""),
                            "Titulo": tit,
                            "Author": "Agência Brasil",
                            "is_fake": False,
                        },
                        theme,
                    )
                    s_count += 1
            print(f"  -> {s_count} matérias classificadas e adicionadas de {supp_file}.")

    # 6. Balanceamento e Consolidação Final
    print("\n" + "=" * 75)
    print("BALANCEAMENTO E COMPOSIÇÃO FINAL:")
    final_records = []
    for theme in THEMES:
        pool = theme_pools[theme]
        # Embaralhar para balancear fontes e veracidade
        random.shuffle(pool)
        selected = pool[:max_per_theme]
        final_records.extend(selected)
        fake_count = sum(1 for item in selected if item["is_fake"] is True)
        true_count = len(selected) - fake_count
        pct_fake = (fake_count / len(selected) * 100) if selected else 0
        print(
            f"  - {theme:<18}: {len(selected):>4} amostras "
            f"(is_fake: {fake_count} falso, {true_count} verdadeiro | {pct_fake:.1f}% fake)"
        )

    # Embaralhar o conjunto completo antes de salvar
    random.shuffle(final_records)

    # 7. Salvar TSV Consolidado
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Claim", "tema", "URL", "Titulo", "Author", "is_fake"]
    with open(output_file, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(final_records)

    print("\n" + "=" * 75)
    print(f"DATASET FINAL DE TEMAS GERADO COM SUCESSO!")
    print(f"Total de registros : {len(final_records)}")
    print(f"Arquivo de saída   : {output_file}")
    print("=" * 75)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_output = repo_root / "datasets" / "dataset_classificacao_temas.tsv"

    parser = argparse.ArgumentParser(
        description="Prepara o dataset balanceado para classificação de notícias por temas (7 classes)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(default_output),
        help=f"Caminho do arquivo TSV de saída (padrão: {default_output.name})",
    )
    parser.add_argument(
        "--max-per-theme", "-m",
        type=int,
        default=700,
        help="Limite máximo de amostras por classe temática (padrão: 700)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade (padrão: 42)",
    )

    args = parser.parse_args()
    build_dataset(
        repo_root=repo_root,
        output_file=Path(args.output),
        max_per_theme=args.max_per_theme,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

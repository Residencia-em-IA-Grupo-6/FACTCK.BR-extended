#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_health.py

Script para classificar e extrair notícias e checagens relacionadas à SAÚDE
a partir de datasets tabulares do FACTCK.BR (como new_factCkBR_old.tsv, FACTCKBR_old.tsv, etc.).

Funcionalidades:
  1. Filtra com alta precisão termos médicos, doenças, vacinas, medicamentos, SUS/saúde pública.
  2. Desambigua metáforas políticas ("síndrome do esquecimento"), termos policiais ("tráfico de drogas")
     e vírus de informática ("vírus no Telegram").
  3. Adiciona colunas de classificação binária (is_health: 1/0) e categorias temáticas (health_categories).
  4. Exporta um arquivo exclusivo contendo apenas as notícias de saúde.
"""

import os
import sys
import re
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Taxonomia de Saúde
HEALTH_CATEGORIES: Dict[str, List[str]] = {
    "doenca_sintoma": [
        r"\b(c[âa]ncer|tumor(es)?|leucemia|quimioterapia|radioterapia)\b",
        r"\b(diabete[st]?|insulina|glicemia)\b",
        r"\b(infarto|parada card[ií]aca|card[ií]ac[oa]s?|avc|derrame cerebral)\b",
        r"\b(sarampo|febre amarela|dengue|zika|chikungunya|mal[aá]ria|leptospirose|tuberculose|lepra|hansen[ií]ase)\b",
        r"\b(hiv|aids|s[ií]filis|gonorreia|hepatite[s]?)\b",
        r"\b(autismo|autistas?|s[ií]ndrome de down|microcefalia)\b",
        r"\b(alzheimer|parkinson|depress[ãa]o cl[ií]nica|ansiedade)\b",
        r"\b(pneumonia|asma|bronquite|meningite)\b",
        r"\b(epilepsia|convuls[ãa]o|hemodi[áa]lise|doen[çc]a renal)\b",
    ],
    "vacina_imunizacao": [
        r"\b(vacina(s|[çc][ãa]o|do[sa]?|r)?|imuniza([çc][ãa]o|do[sa]?|r)?)\b",
    ],
    "medicamento_tratamento": [
        r"\b(medicamento[s]?|rem[ée]dio[s]?|f[áa]rmaco[s]?|antibi[oó]tico[s]?|analg[ée]sico[s]?)\b",
        r"\b(tratamento m[ée]dico|cura (d[aeo]|para) (c[âa]ncer|doen[çc]a|aids|hiv))\b",
        r"\b(cirurgia[s]?|cir[úu]rgic[oa]s?|transplante[s]? de [oó]rg[ãa]os?)\b",
        r"\b(dipirona|paracetamol|ibuprofeno|morfina)\b",
    ],
    "saude_publica_sistema": [
        r"\b(sistema [úu]nico de sa[úu]de|\bSUS\b)\b",
        r"\b(minist[ée]rio da sa[úu]de|anvisa|organiza[çc][ãa]o mundial da sa[úu]de|\bOMS\b)\b",
        r"\b(m[ée]dic[oa]s?|enfermeir[oa]s?|hospital(ar|ares)?|\bUTI\b|\bUTIs\b|leitos? de hospital|postos? de sa[úu]de|mais m[ée]dicos)\b",
        r"\b(sa[úu]de p[úu]blica)\b",
    ],
    "epidemia_infeccao": [
        r"\b(infec[çc][ãa]o|infeccios[oa]s?|epidemia[s]?|pandemia[s]?|surto da doen[çc]a)\b",
        r"\b(v[ií]rus (hiv|h1n1|ebola|zika|chikungunya|da gripe|sincicial|da febre))\b",
    ],
}

COMPILED_CATEGORIES = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in HEALTH_CATEGORIES.items()
}

# Expressões para desambiguação de falsos positivos
NEGATIVE_PATTERNS = [
    re.compile(r"\b(tr[áa]fico de drogas?|apreens[ãa]o de drogas?|porte de drogas?|coca[ií]na|maconha)\b", re.IGNORECASE),
    re.compile(r"\b(v[ií]rus de computador|hacker[es]?|malware)\b", re.IGNORECASE),
    re.compile(r"\bs[ií]ndrome do esquecimento\b", re.IGNORECASE),
    re.compile(r"\bpolicial.*esfaquead[oa]\b", re.IGNORECASE),
]


def classify_text(text: str) -> Tuple[int, str]:
    """
    Avalia se o texto é relacionado à saúde e retorna (is_health, categorias).
    """
    if not text:
        return 0, ""

    matched_cats = set()
    for cat, regexes in COMPILED_CATEGORIES.items():
        for r in regexes:
            if r.search(text):
                matched_cats.add(cat)
                break

    if not matched_cats:
        return 0, ""

    # Checagem de negativos / falsos positivos
    has_negative = any(np.search(text) for np in NEGATIVE_PATTERNS)
    has_disease_or_vax = bool(matched_cats.intersection({"doenca_sintoma", "vacina_imunizacao", "medicamento_tratamento"}))

    if has_negative and not has_disease_or_vax:
        return 0, ""

    return 1, ",".join(sorted(matched_cats))


def process_dataset(input_file: Path, output_health_file: Path, annotate_file: Path = None):
    """Lê o dataset, classifica as notícias e salva as saídas."""
    print(f"Lendo dataset: {input_file}...")
    df = pd.read_csv(input_file, sep="\t", dtype=str, keep_default_na=False)

    print(f"Total de registros carregados: {len(df)}")

    # Classificação linha a linha
    is_health_list = []
    categories_list = []

    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        claim = str(row.get("claimReviewed", ""))
        review = str(row.get("reviewBody", ""))
        text = f"{title} {claim} {review}"

        is_h, cats = classify_text(text)
        is_health_list.append(is_h)
        categories_list.append(cats)

    df["is_health"] = is_health_list
    df["health_categories"] = categories_list

    health_subset = df[df["is_health"] == 1].copy()
    non_health_count = len(df) - len(health_subset)

    print("=" * 70)
    print("RESULTADOS DA CLASSIFICAÇÃO:")
    print(f"  Notícias relacionadas à saúde  (is_health=1): {len(health_subset)}")
    print(f"  Outras notícias                (is_health=0): {non_health_count}")
    print("=" * 70)

    # Salvar base filtrada de saúde
    output_health_file.parent.mkdir(parents=True, exist_ok=True)
    health_subset.to_csv(output_health_file, sep="\t", index=False)
    print(f"[OK] Base de saúde exportada: {output_health_file}")

    # Salvar base completa anotada com as colunas is_health se solicitado
    if annotate_file:
        annotate_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(annotate_file, sep="\t", index=False)
        print(f"[OK] Base completa com anotação is_health exportada: {annotate_file}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_input = repo_root / "datasets" / "new_factCkBR_old.tsv"
    default_output = repo_root / "datasets" / "new_factCkBR_old_saude.tsv"

    parser = argparse.ArgumentParser(
        description="Classificador de notícias relacionadas à saúde para datasets FACTCK.BR."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(default_input),
        help=f"Arquivo TSV de entrada (padrão: {default_input.name}).",
    )
    parser.add_argument(
        "--output-health", "-o",
        type=str,
        default=str(default_output),
        help=f"Arquivo de saída apenas com notícias de saúde (padrão: {default_output.name}).",
    )
    parser.add_argument(
        "--annotate-full", "-a",
        type=str,
        default=None,
        help="Caminho opcional para salvar a base completa anotada com as colunas is_health e health_categories.",
    )

    args = parser.parse_args()

    process_dataset(
        input_file=Path(args.input),
        output_health_file=Path(args.output_health),
        annotate_file=Path(args.annotate_full) if args.annotate_full else None,
    )


if __name__ == "__main__":
    main()


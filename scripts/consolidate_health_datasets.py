#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consolidate_health_datasets.py

Consolida os datasets de saúde do repositório (boatos_saude.tsv e FACTCKBR_updated_saude.tsv)
em uma base única e padronizada com os atributos solicitados:
  - URL: Link do artigo de checagem
  - Data: Data de publicação (YYYY-MM-DD)
  - Titulo: Título da checagem
  - Author: Veículo/agência responsável (Aos Fatos, Lupa, Boatos.org, etc.)
  - Claim: Texto original do boato/alegação (Texto_Falso_Original / claimReviewed)
  - reviewBody: Texto da checagem/desmentido (Desmentido / reviewBody)
  - is_fake: Booleano declarando se o claim é falso (True) ou verdadeiro (False)
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path


def map_author(author_str: str) -> str:
    """Padroniza domínios e URLs para os nomes canônicos das agências."""
    if not author_str:
        return "Desconhecido"
    s = str(author_str).lower()
    if "boatos" in s:
        return "Boatos.org"
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
    if "oglobo" in s or "globo" in s:
        return "O Globo"
    if "tatu" in s:
        return "Agência Tatu"
    if "nexo" in s:
        return "Nexo Jornal"
    return str(author_str).strip()


def classify_veracity(alt_name: str) -> bool:
    """
    Retorna True se a afirmação for falsa/enganosa (is_fake = True)
    e False se a afirmação for verdadeira/correta (is_fake = False).
    """
    s = str(alt_name).lower().strip()
    if re.search(
        r"\b(fals[oa]|enganos[oa]|enganador|errad[oa]|distorcido|insustent[aá]vel|sem contexto|fora de contexto|fake|s[aá]tira|imposs[ií]vel provar|impreciso|não é bem assim)\b",
        s,
    ):
        return True
    if re.search(r"\b(verdadeir[oa]|certo|fato|procede|aut[eê]ntico)\b", s):
        return False
    return True


def consolidate(
    boatos_file: Path,
    factckbr_saude_file: Path,
    output_file: Path,
):
    """Executa a consolidação dos dois datasets de saúde."""
    print("=" * 75)
    print("CONSOLIDAÇÃO DE DATASETS DE SAÚDE - FACTCK.BR")
    print("=" * 75)

    records = []

    # 1. Processar Boatos.org Saúde
    if boatos_file.exists():
        print(f"Lendo Boatos.org: {boatos_file.name}...")
        df_b = pd.read_csv(boatos_file, sep="\t", dtype=str, keep_default_na=False)
        count_b = 0
        for _, r in df_b.iterrows():
            claim = r.get("Texto_Falso_Original", "").strip()
            if not claim:
                claim = r.get("Resumo_Boato", "").strip()
            if not claim:
                claim = r.get("Titulo", "").strip()

            records.append({
                "URL": r.get("URL", "").strip(),
                "Data": r.get("Data", "").strip(),
                "Titulo": r.get("Titulo", "").strip(),
                "Author": "Boatos.org",
                "Claim": claim,
                "reviewBody": r.get("Desmentido", "").strip(),
                "is_fake": True,  # Todas as matérias do Boatos.org/saúde verificam boatos falsos
            })
            count_b += 1
        print(f"  -> {count_b} registros processados do Boatos.org.")
    else:
        print(f"Aviso: Arquivo {boatos_file} não encontrado.")

    # 2. Processar FACTCKBR_updated_saude
    if factckbr_saude_file.exists():
        print(f"\nLendo FACTCKBR Saúde: {factckbr_saude_file.name}...")
        df_f = pd.read_csv(factckbr_saude_file, sep="\t", dtype=str, keep_default_na=False)
        count_f = 0
        for _, r in df_f.iterrows():
            claim = r.get("claimReviewed", "").strip()
            if not claim:
                claim = r.get("title", "").strip()

            is_fake = classify_veracity(r.get("alternativeName", ""))

            review = r.get("reviewBody", "").strip()
            if not review:
                review = r.get("title", "").strip()

            records.append({
                "URL": r.get("URL", "").strip(),
                "Data": r.get("datePublished", "").strip(),
                "Titulo": r.get("title", "").strip(),
                "Author": map_author(r.get("Author", "")),
                "Claim": claim,
                "reviewBody": review,
                "is_fake": is_fake,
            })
            count_f += 1
        print(f"  -> {count_f} registros processados do FACTCKBR Saúde.")
    else:
        print(f"Aviso: Arquivo {factckbr_saude_file} não encontrado.")

    if not records:
        print("Nenhum registro encontrado para consolidação.")
        return

    # 3. Criar DataFrame consolidado
    df_consolidated = pd.DataFrame(records)

    # Ordenar por data decrescente (mais recentes primeiro)
    df_consolidated = df_consolidated.sort_values("Data", ascending=False)

    # Deduplicar por URL e Claim se houver redundâncias residuais
    orig_total = len(df_consolidated)
    df_consolidated = df_consolidated.drop_duplicates(subset=["URL", "Claim"], keep="first")
    final_total = len(df_consolidated)

    # 4. Salvar arquivo TSV consolidado
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_consolidated.to_csv(output_file, sep="\t", index=False)

    print("\n" + "=" * 75)
    print("ESTATÍSTICAS DA BASE CONSOLIDADA:")
    print(f"  Total de registros: {final_total}")
    print(f"  Notícias falsas / boatos  (is_fake=True):  {(df_consolidated['is_fake'] == True).sum()}")
    print(f"  Notícias verdadeiras      (is_fake=False): {(df_consolidated['is_fake'] == False).sum()}")
    print("\nDistribuição por Veículo / Autor:")
    for author, count in df_consolidated["Author"].value_counts().items():
        print(f"  - {author:<20}: {count}")

    print("\n[OK] Arquivo consolidado salvo em:")
    print(f"     {output_file}")
    print("=" * 75)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_boatos = repo_root / "datasets" / "boatos_saude.tsv"
    default_factckbr_saude = repo_root / "datasets" / "FACTCKBR_updated_saude.tsv"
    default_output = repo_root / "datasets" / "factckbr_boatos_saude_consolidado.tsv"

    parser = argparse.ArgumentParser(
        description="Consolida os datasets de saúde boatos_saude e FACTCKBR_updated_saude."
    )
    parser.add_argument(
        "--boatos",
        type=str,
        default=str(default_boatos),
        help=f"Arquivo TSV do Boatos.org (padrão: {default_boatos.name}).",
    )
    parser.add_argument(
        "--factckbr",
        type=str,
        default=str(default_factckbr_saude),
        help=f"Arquivo TSV do FACTCKBR Saúde (padrão: {default_factckbr_saude.name}).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(default_output),
        help=f"Caminho do arquivo TSV consolidado (padrão: {default_output.name}).",
    )

    args = parser.parse_args()

    consolidate(
        boatos_file=Path(args.boatos),
        factckbr_saude_file=Path(args.factckbr),
        output_file=Path(args.output),
    )


if __name__ == "__main__":
    main()

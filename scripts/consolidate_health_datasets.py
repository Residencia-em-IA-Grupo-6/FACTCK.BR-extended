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
import html
import argparse
import pandas as pd
from pathlib import Path


def is_invalid_claim_text(s: str) -> bool:
    """Detecta se o texto extraído é na verdade um marcador/embed web em vez do boato."""
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


def clean_claim(text: str) -> str:
    """
    Higieniza o texto da alegação para remover vieses que distorcem o treinamento de ML:
      1. Vazamento direto de rótulo: 'Boato -', 'Boato –', 'Boato:', '#boato', etc.
      2. Prefixos editoriais de agências: 'Conteúdo verificado:', 'Versão 1:', 'Texto:'
      3. Aspas envolventes e tipográficas (“...”, "...", '...', «...»)
      4. Emojis e pictogramas de sensacionalismo (🚨, ⚠️, 💉, 💀, etc.)
      5. Links externos e menções de rede social (https://, @user, #tags)
      6. Resíduos de tags HTML e entidades (&amp;, <span...>)
      7. Pontuação excessiva e marcadores editoriais ([…], !!!, ???)
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Decodificar entidades HTML (&amp;, &quot;, &#8220;, etc.)
    t = html.unescape(text)

    # 2. Remover tags HTML residuais (<span...>, <br>, etc.)
    t = re.sub(r"<[^>]+>", " ", t)

    # 3. Remover URLs, encurtadores e links de mídia
    t = re.sub(r"https?://\S+|pic\.twitter\.com/\S+|t\.co/\S+", "", t)

    # 4. Remover vazamentos de rótulo / veredito inicial
    t = re.sub(r"^[Bb]oato\s*[\-–—:\.]\s*", "", t)
    t = re.sub(r"^[Cc]onte[úu]do\s*verificado\s*[\-–—:\.]\s*", "", t)
    t = re.sub(r"^(?:[Vv]ers[ãa]o|[Tt]exto)\s*\d*\s*[\-–—:\.]\s*", "", t)
    t = re.sub(r"^[Ff]also\s*[\-–—:]\s*", "", t)

    # Tags de veredito no corpo ou no final (#boato, [boato], (boato))
    t = re.sub(r"#boato\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[boato\]|\(boato\)", "", t, flags=re.IGNORECASE)

    # 5. Remover emojis e símbolos pictográficos que introduzem atalhos espúrios
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\u200d\ufe0f]",
        flags=re.UNICODE
    )
    t = emoji_pattern.sub("", t)

    # 6. Remover menções sociais (@usuario) e desmembrar hashtags (#palavra -> palavra)
    t = re.sub(r"#([A-Za-z0-9_À-ÿ]+)", r"\1", t)
    t = re.sub(r"@\w+", "", t)

    # 7. Remover marcadores editoriais de corte/truncamento ([…], [...])
    t = re.sub(r"\[…\]|\[\.\.\.\]", "", t)

    # 8. Normalizar pontuação enfática exagerada (!!!! -> !, ???? -> ?)
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"\?{2,}", "?", t)

    # 9. Remover aspas envolventes (leading e trailing quotation marks)
    t = t.strip()
    quote_chars = "\"\'“”‘’«»"
    while len(t) > 1 and t[0] in quote_chars and t[-1] in quote_chars:
        t = t[1:-1].strip()
    while len(t) > 0 and t[0] in quote_chars:
        t = t[1:].strip()
    while len(t) > 0 and t[-1] in quote_chars:
        t = t[:-1].strip()

    # 10. Normalizar espaços múltiplos e quebras de linha
    t = re.sub(r"\s+", " ", t).strip()

    # 11. Garantir primeira letra maiúscula após remoção de prefixos
    if t and t[0].islower():
        t = t[0].upper() + t[1:]

    return t


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
            if is_invalid_claim_text(claim):
                claim = r.get("Resumo_Boato", "").strip()
            if is_invalid_claim_text(claim):
                claim = r.get("Titulo", "").strip()

            clean_c = clean_claim(claim)
            if not clean_c or len(clean_c) < 15:
                clean_c = clean_claim(r.get("Titulo", "").strip())

            records.append({
                "URL": r.get("URL", "").strip(),
                "Data": r.get("Data", "").strip(),
                "Titulo": r.get("Titulo", "").strip(),
                "Author": "Boatos.org",
                "Claim": clean_c,
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

            clean_c = clean_claim(claim)
            if not clean_c or len(clean_c) < 15:
                clean_c = clean_claim(r.get("title", "").strip())

            is_fake = classify_veracity(r.get("alternativeName", ""))

            review = r.get("reviewBody", "").strip()
            if not review:
                review = r.get("title", "").strip()

            records.append({
                "URL": r.get("URL", "").strip(),
                "Data": r.get("datePublished", "").strip(),
                "Titulo": r.get("title", "").strip(),
                "Author": map_author(r.get("Author", "")),
                "Claim": clean_c,
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

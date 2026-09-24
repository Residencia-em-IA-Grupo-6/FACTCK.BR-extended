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

import argparse
import csv
import html
import os
import re
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


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


def load_tsv_records(file_path: Path):
    """Carrega registros de um TSV com pandas (se disponível) ou csv.DictReader."""
    if not file_path or not file_path.exists():
        return []
    if HAS_PANDAS:
        df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
        return df.to_dict(orient="records")
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def consolidate(
    boatos_file: Path,
    factckbr_saude_file: Path,
    output_file: Path,
    noticias_ms_file: Path = None,
    ebc_saude_file: Path = None,
):
    """Executa a consolidação dos datasets de saúde."""
    """Executa a consolidação dos datasets de saúde."""
    print("=" * 75)
    print("CONSOLIDAÇÃO DE DATASETS DE SAÚDE - FACTCK.BR")
    print("=" * 75)

    records = []

    # 1. Processar Boatos.org Saúde
    if boatos_file.exists():
        print(f"Lendo Boatos.org: {boatos_file.name}...")
        rows_b = load_tsv_records(boatos_file)
        count_b = 0
        for r in rows_b:
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
        rows_f = load_tsv_records(factckbr_saude_file)
        count_f = 0
        for r in rows_f:
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

    # 3. Processar Notícias do Ministério da Saúde se informado e existir
    if noticias_ms_file and noticias_ms_file.exists():
        print(f"\nLendo Notícias do Ministério da Saúde: {noticias_ms_file.name}...")
        rows_ms = load_tsv_records(noticias_ms_file)
        count_ms = 0
        for r in rows_ms:
            claim = r.get("Claim", "").strip() or r.get("Subtitulo", "").strip() or r.get("Titulo", "").strip()
            clean_c = clean_claim(claim)
            if not clean_c or len(clean_c) < 15:
                clean_c = clean_claim(r.get("Titulo", "").strip())

            review = r.get("reviewBody", "").strip() or r.get("Conteudo", "").strip()
            if not review:
                review = r.get("Titulo", "").strip()

            records.append({
                "URL": r.get("URL", "").strip(),
                "Data": r.get("Data", "").strip(),
                "Titulo": r.get("Titulo", "").strip(),
                "Author": "Ministério da Saúde",
                "Claim": clean_c,
                "reviewBody": review,
                "is_fake": False,  # Notícias oficiais da autoridade pública de saúde
            })
            count_ms += 1
        print(f"  -> {count_ms} registros processados do Ministério da Saúde.")

    # 4. Processar Notícias da Agência Brasil (EBC) se informado e existir
    if ebc_saude_file and ebc_saude_file.exists():
        print(f"\nLendo Agência Brasil (EBC): {ebc_saude_file.name}...")
        rows_ebc = load_tsv_records(ebc_saude_file)
        count_ebc = 0
        for r in rows_ebc:
            claim = r.get("Claim", "").strip() or r.get("Subtitulo", "").strip() or r.get("Titulo", "").strip()
            clean_c = clean_claim(claim)
            if not clean_c or len(clean_c) < 15:
                clean_c = clean_claim(r.get("Titulo", "").strip())

            review = r.get("reviewBody", "").strip() or r.get("Conteudo", "").strip()
            if not review:
                review = r.get("Titulo", "").strip()

            records.append({
                "URL": r.get("URL", "").strip(),
                "Data": r.get("Data", "").strip(),
                "Titulo": r.get("Titulo", "").strip(),
                "Author": "Agência Brasil",
                "Claim": clean_c,
                "reviewBody": review,
                "is_fake": False,  # Notícias oficiais da agência pública governamental
            })
            count_ebc += 1
        print(f"  -> {count_ebc} registros processados da Agência Brasil.")

    if not records:
        print("Nenhum registro encontrado para consolidação.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PANDAS:
        df_consolidated = pd.DataFrame(records)
        df_consolidated = df_consolidated.sort_values("Data", ascending=False)
        df_consolidated = df_consolidated.drop_duplicates(subset=["URL", "Claim"], keep="first")
        final_records = df_consolidated.to_dict(orient="records")
    else:
        seen = set()
        deduped = []
        for r in records:
            key = (r.get("URL", "").strip(), r.get("Claim", "").strip())
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        deduped.sort(key=lambda x: x.get("Data", ""), reverse=True)
        final_records = deduped

    # 4. Salvar arquivo TSV consolidado
    fieldnames = ["URL", "Data", "Titulo", "Author", "Claim", "reviewBody", "is_fake"]
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(final_records)

    total_fake = sum(1 for r in final_records if str(r.get("is_fake")).lower() in ["true", "1"])
    total_true = sum(1 for r in final_records if str(r.get("is_fake")).lower() in ["false", "0"])
    from collections import Counter
    author_counts = Counter(r.get("Author", "Desconhecido") for r in final_records)

    print("\n" + "=" * 75)
    print("ESTATÍSTICAS DA BASE CONSOLIDADA:")
    print(f"  Total de registros: {len(final_records)}")
    print(f"  Notícias falsas / boatos  (is_fake=True):  {total_fake}")
    print(f"  Notícias verdadeiras      (is_fake=False): {total_true}")
    print("\nDistribuição por Veículo / Autor:")
    for author, count in author_counts.most_common():
        print(f"  - {author:<20}: {count}")

    print("\n[OK] Arquivo consolidado salvo em:")
    print(f"     {output_file}")
    print("=" * 75)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_boatos = repo_root / "datasets" / "boatos_saude.tsv"
    default_factckbr_saude = repo_root / "datasets" / "FACTCKBR_updated_saude.tsv"
    default_noticias_ms = repo_root / "datasets" / "noticias_ms.tsv"
    default_ebc_saude = repo_root / "datasets" / "ebc_saude.tsv"
    default_output = repo_root / "datasets" / "factckbr_boatos_saude_consolidado.tsv"

    parser = argparse.ArgumentParser(
        description="Consolida os datasets de saúde boatos_saude, FACTCKBR_updated_saude, noticias_ms e ebc_saude."
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
        "--noticias-ms",
        type=str,
        default=str(default_noticias_ms) if default_noticias_ms.exists() else None,
        help="Arquivo TSV com notícias oficiais do Ministério da Saúde (opcional).",
    )
    parser.add_argument(
        "--ebc-saude",
        type=str,
        default=str(default_ebc_saude) if default_ebc_saude.exists() else None,
        help="Arquivo TSV com notícias oficiais da Agência Brasil / EBC (opcional).",
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
        noticias_ms_file=Path(args.noticias_ms) if args.noticias_ms else None,
        ebc_saude_file=Path(args.ebc_saude) if args.ebc_saude else None,
    )


if __name__ == "__main__":
    main()

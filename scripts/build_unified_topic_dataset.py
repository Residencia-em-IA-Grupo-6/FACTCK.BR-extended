#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_unified_topic_dataset.py

Gera o dataset balanceado e higienizado para classificação de tópicos e gatekeeper de saúde:
- 8 classes funcionais equilibradas (3.000 amostras por classe = 24.000 amostras totais)
- Mistura balanceada de Notícias Jornalísticas, Redes Sociais/Tweets, Boatos/Checagens e Perguntas Reais de Usuários
- Higienização estrita contra vazamentos de atalho ("Boato -", "Fato ou Fake", URLs, emojis)
- Partição fixa estratificada: 80% treino (19.200), 10% validação (2.400), 10% teste (2.400)
"""

import argparse
import hashlib
import html
import json
import logging
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TARGET_CLASSES = [
    "saude",
    "conversa_geral",
    "politica_governo",
    "entretenimento_cultura",
    "esportes",
    "economia_negocios",
    "ciencia_tecnologia",
    "cotidiano_seguranca",
]

# Expressões regulares para higienização
RE_URL = re.compile(r"https?://\S+|pic\.twitter\.com/\S+|t\.co/\S+")
RE_MENTION = re.compile(r"@\w+")
RE_RT = re.compile(r"^rt\s+", re.IGNORECASE)
RE_EMOJI = re.compile("[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\u200d\ufe0f]", flags=re.UNICODE)
RE_HTML_TAGS = re.compile(r"<[^>]+>")

# Prefixos de atalho e vazamento de rótulo
RE_SHORTCUTS = [
    re.compile(r"^[Bb]oato\s*[\-–—:\.]\s*"),
    re.compile(r"^[Cc]onte[úu]do\s*verificado\s*[\-–—:\.]\s*"),
    re.compile(r"^[Ff]also\s*[\-–—:]\s*"),
    re.compile(r"^[Ff]ato\s+ou\s+[Ff]ake\s*[\-–—:]\s*"),
    re.compile(r"^[Vv]erdadeiro\s*[\-–—:]\s*"),
    re.compile(r"^(?:[Vv]ers[ãa]o|[Tt]exto)\s*\d*\s*[\-–—:\.]\s*"),
]


def clean_text(text: str) -> str:
    """Higieniza o texto removendo vieses de ML, marcadores editoriais e links."""
    if not text or not isinstance(text, str):
        return ""

    t = html.unescape(text)
    t = RE_HTML_TAGS.sub(" ", t)
    t = RE_URL.sub("", t)
    t = RE_RT.sub("", t)
    t = RE_MENTION.sub("", t)

    # Remover prefixos editoriais de boato/checagem
    for pat in RE_SHORTCUTS:
        t = pat.sub("", t)

    t = re.sub(r"#boato\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[boato\]|\(boato\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\[…\]|\[\.\.\.\]", "", t)

    t = RE_EMOJI.sub("", t)

    # Limpar aspas envolventes
    t = t.strip()
    quotes = "\"\'“”‘’«»"
    while len(t) > 1 and t[0] in quotes and t[-1] in quotes:
        t = t[1:-1].strip()
    while len(t) > 0 and t[0] in quotes:
        t = t[1:].strip()
    while len(t) > 0 and t[-1] in quotes:
        t = t[:-1].strip()

    # Normalizar espaços
    t = re.sub(r"\s+", " ", t).strip()

    if t and t[0].islower():
        t = t[0].upper() + t[1:]

    return t


def is_valid_text(text: str) -> bool:
    """Valida se o texto possui tamanho e qualidade adequados."""
    if not text or len(text) < 25 or len(text) > 1200:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    # Evitar ruídos técnicos de web
    bad_patterns = [
        "confira o desmentido",
        "ver essa foto no instagram",
        "assista ao desmentido",
        "data-mce-type",
        "javascript",
        "subscribe to our channel",
    ]
    low = text.lower()
    return not any(p in low for p in bad_patterns)


def get_first_sentence(text: str, max_chars: int = 200) -> str:
    """Extrai o lead ou primeira frase do texto jornalístico."""
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if sentences:
        s = sentences[0].strip()
        if len(s) < 40 and len(sentences) > 1:
            s += " " + sentences[1].strip()
        return s[:max_chars].strip()
    return text[:max_chars].strip()


class DatasetCollector:
    def __init__(self, target_per_class: int = 3000, seed: int = 42):
        self.target_per_class = target_per_class
        self.seed = seed
        self.pools: Dict[str, List[dict]] = {cls: [] for cls in TARGET_CLASSES}
        self.seen_hashes: Set[str] = set()

    def add(self, text: str, topic: str, text_type: str, source: str, is_fake: Optional[bool] = None) -> bool:
        if topic not in self.pools:
            return False

        cleaned = clean_text(text)
        if not is_valid_text(cleaned):
            return False

        # Deduplicação por hash MD5 normalizado
        norm_key = re.sub(r"\W+", "", cleaned.lower())
        if len(norm_key) < 20:
            return False
        h = hashlib.md5(norm_key.encode("utf-8")).hexdigest()
        if h in self.seen_hashes:
            return False

        self.seen_hashes.add(h)
        self.pools[topic].append({
            "id": h[:12],
            "text": cleaned,
            "topic": topic,
            "text_type": text_type,
            "source": source,
            "is_fake": is_fake,
            "is_health": (topic == "saude"),
        })
        return True

    def count(self, topic: str) -> int:
        return len(self.pools.get(topic, []))


def build_dataset(repo_root: Path, target_per_class: int = 3000, seed: int = 42) -> pd.DataFrame:
    raw_dir = repo_root / "data" / "raw"
    datasets_dir = repo_root / "datasets"
    collector = DatasetCollector(target_per_class=target_per_class, seed=seed)

    logger.info("=" * 70)
    logger.info("INICIANDO CONSTRUÇÃO DO DATASET DE TÓPICOS (8 CLASSES)")
    logger.info(f"Classes: {TARGET_CLASSES}")
    logger.info(f"Meta por classe: {target_per_class} amostras")
    logger.info("=" * 70)

    # -------------------------------------------------------------
    # 1. SAÚDE: Perguntas de pacientes (MedPT) e desabafos (MentalHealth)
    # -------------------------------------------------------------
    medpt_file = raw_dir / "medpt__train.parquet"
    if medpt_file.exists():
        logger.info(f"Carregando perguntas de saúde do MedPT ({medpt_file.name})...")
        df_med = pd.read_parquet(medpt_file, columns=["question"])
        for q in df_med["question"].dropna().sample(frac=1.0, random_state=seed):
            collector.add(str(q), "saude", "pergunta_usuario", "hf:medpt")
            if collector.count("saude") >= int(target_per_class * 0.45):
                break
        logger.info(f"  -> Total de saúde após MedPT: {collector.count('saude')}")

    mh_file = raw_dir / "mental_health_pt__train.parquet"
    if mh_file.exists():
        logger.info(f"Carregando desabafos e consultas de saúde mental ({mh_file.name})...")
        df_mh = pd.read_parquet(mh_file, columns=["input"])
        for inp in df_mh["input"].dropna().sample(frac=1.0, random_state=seed):
            collector.add(str(inp), "saude", "conversa", "hf:mental_health_pt")
            if collector.count("saude") >= int(target_per_class * 0.70):
                break
        logger.info(f"  -> Total de saúde após MentalHealth: {collector.count('saude')}")

    # Checagens e notícias oficiais de saúde locais
    health_local = datasets_dir / "saude_consolidado_unificado.tsv"
    if health_local.exists():
        logger.info(f"Carregando notícias e checagens oficiais de saúde ({health_local.name})...")
        df_hl = pd.read_csv(health_local, sep="\t")
        for _, row in df_hl.sample(frac=1.0, random_state=seed).iterrows():
            claim = str(row.get("Claim", "")).strip() or str(row.get("Titulo", "")).strip()
            is_fk = bool(row.get("is_fake")) if pd.notnull(row.get("is_fake")) else None
            author = str(row.get("Author", "local"))
            collector.add(claim, "saude", "alegacao", f"local:{author}", is_fake=is_fk)
            if collector.count("saude") >= target_per_class:
                break
        logger.info(f"  -> Total de saúde após Base Consolidada: {collector.count('saude')}")

    # -------------------------------------------------------------
    # 2. CONVERSA GERAL: Tweets casuais e chitchat
    # -------------------------------------------------------------
    tweets_file = raw_dir / "pt_tweets__train.parquet"
    if tweets_file.exists():
        logger.info(f"Carregando tweets casuais ({tweets_file.name})...")
        df_tw = pd.read_parquet(tweets_file, columns=["tweet_text"])
        for tw in df_tw["tweet_text"].dropna().sample(frac=1.0, random_state=seed):
            # Filtrar tweets muito curtos ou que sejam links puros
            collector.add(str(tw), "conversa_geral", "mensagem_social", "hf:pt_tweets")
            if collector.count("conversa_geral") >= int(target_per_class * 0.75):
                break
        logger.info(f"  -> Total em conversa_geral após Tweets: {collector.count('conversa_geral')}")

    told_file = raw_dir / "told_br__train.parquet"
    if told_file.exists():
        logger.info(f"Carregando mensagens sociais cotidianas ({told_file.name})...")
        df_told = pd.read_parquet(told_file, columns=["text", "homophobia", "racism", "misogyny"])
        # Filtrar mensagens não tóxicas
        df_told_clean = df_told[(df_told["homophobia"] == 0) & (df_told["racism"] == 0) & (df_told["misogyny"] == 0)]
        for t in df_told_clean["text"].dropna().sample(frac=1.0, random_state=seed):
            collector.add(str(t), "conversa_geral", "mensagem_social", "hf:told_br")
            if collector.count("conversa_geral") >= target_per_class:
                break
        logger.info(f"  -> Total em conversa_geral após ToldBR: {collector.count('conversa_geral')}")

    # Saudações e diálogos de apoio para conversa geral
    chitchat_samples = [
        "Olá, bom dia! Como você está hoje?",
        "Boa tarde! Gostaria de conversar um pouco.",
        "Oi, tudo bem contigo? Como foi seu final de semana?",
        "Muito obrigado pela ajuda, tenha um excelente dia!",
        "Boa noite, você pode me ajudar a tirar uma dúvida geral?",
        "Olá amigo, tudo certo por aí?",
        "Valeu demais pelo apoio, agradeço bastante!",
        "Oi, você sabe que horas são agora em Brasília?",
        "Bom dia a todos, que tenhamos uma semana produtiva e abençoada!",
        "Boa tarde pessoal, alguém aí animado para o feriado?",
    ]
    for s in chitchat_samples:
        collector.add(s, "conversa_geral", "conversa", "manual:chitchat")

    # -------------------------------------------------------------
    # 3. FAKERECOGNA: Boatos e checagens categorizadas
    # -------------------------------------------------------------
    fake_file = raw_dir / "fakerecogna__train.parquet"
    if fake_file.exists():
        logger.info(f"Carregando checagens e boatos do FakeRecogna ({fake_file.name})...")
        df_fk = pd.read_parquet(fake_file, columns=["Titulo", "Noticia", "Categoria"])
        fk_map = {
            "política": "politica_governo",
            "mundo": "politica_governo",
            "entretenimento": "entretenimento_cultura",
            "ciência": "ciencia_tecnologia",
        }
        for _, row in df_fk.sample(frac=1.0, random_state=seed).iterrows():
            cat = str(row.get("Categoria", "")).lower().strip()
            target_topic = fk_map.get(cat)
            if target_topic and collector.count(target_topic) < int(target_per_class * 0.35):
                tit = str(row.get("Titulo", "")).strip()
                lead = get_first_sentence(str(row.get("Noticia", "")))
                comb = f"{tit}. {lead}" if lead and lead != tit else tit
                collector.add(comb, target_topic, "alegacao", "hf:fakerecogna", is_fake=True)

    # -------------------------------------------------------------
    # 4. FOLHA DE S.PAULO: Notícias e manchetes jornalísticas
    # -------------------------------------------------------------
    folha_file = raw_dir / "folha__train.parquet"
    if folha_file.exists():
        logger.info(f"Carregando matérias jornalísticas da Folha ({folha_file.name})...")
        df_folha = pd.read_parquet(folha_file, columns=["title", "text", "category"])
        folha_map = {
            "poder": "politica_governo",
            "mundo": "politica_governo",
            "mercado": "economia_negocios",
            "esporte": "esportes",
            "ilustrada": "entretenimento_cultura",
            "tv": "entretenimento_cultura",
            "ilustrissima": "entretenimento_cultura",
            "cotidiano": "cotidiano_seguranca",
            "saopaulo": "cotidiano_seguranca",
            "tec": "ciencia_tecnologia",
            "ciencia": "ciencia_tecnologia",
        }

        # Iterar com amostragem aleatória
        for _, row in df_folha.sample(frac=1.0, random_state=seed).iterrows():
            f_cat = str(row.get("category", "")).lower().strip()
            target_topic = folha_map.get(f_cat)
            if not target_topic:
                continue

            if collector.count(target_topic) < target_per_class:
                tit = str(row.get("title", "")).strip()
                lead = get_first_sentence(str(row.get("text", "")), max_chars=180)
                comb = f"{tit}. {lead}" if lead and lead != tit else tit
                collector.add(comb, target_topic, "noticia", "hf:folha", is_fake=False)

    # -------------------------------------------------------------
    # 5. DATASET LOCAL DE 7 TEMAS (Injetar checagens e boundary cases)
    # -------------------------------------------------------------
    temas_local = datasets_dir / "dataset_classificacao_temas.tsv"
    if temas_local.exists():
        logger.info(f"Complementando com dataset calibrado local ({temas_local.name})...")
        df_temas = pd.read_csv(temas_local, sep="\t")
        temas_map = {
            "politica": "politica_governo",
            "esporte": "esportes",
            "entretenimento": "entretenimento_cultura",
            "economia": "economia_negocios",
            "tecnologia": "ciencia_tecnologia",
            "seguranca_publica": "cotidiano_seguranca",
        }
        for _, row in df_temas.sample(frac=1.0, random_state=seed).iterrows():
            t_orig = str(row.get("tema", "")).strip()
            target_topic = temas_map.get(t_orig)
            if target_topic and collector.count(target_topic) < target_per_class:
                claim = str(row.get("Claim", "")).strip()
                is_fk = bool(row.get("is_fake")) if pd.notnull(row.get("is_fake")) else None
                collector.add(claim, target_topic, "alegacao", "local:factckbr_temas", is_fake=is_fk)

    # -------------------------------------------------------------
    # 6. Balanceamento e Montagem Final
    # -------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("RESUMO DAS AMOSTRAS COLETADAS POR CLASSE:")
    final_rows = []
    for cls in TARGET_CLASSES:
        pool = collector.pools[cls]
        actual_count = len(pool)
        logger.info(f"  - {cls:<25}: {actual_count} amostras disponíveis")
        # Subamostrar exatamente para target_per_class
        if actual_count > target_per_class:
            selected = pd.DataFrame(pool).sample(n=target_per_class, random_state=seed).to_dict("records")
        else:
            selected = pool
        final_rows.extend(selected)

    df_final = pd.DataFrame(final_rows)
    logger.info(f"\nTotal final de amostras no dataset: {len(df_final)}")
    logger.info("Distribuição de tópicos final:")
    print(df_final["topic"].value_counts())

    # -------------------------------------------------------------
    # 7. Partição Estratificada (80% train, 10% val, 10% test)
    # -------------------------------------------------------------
    logger.info("Criando partição estratificada (80% train, 10% val, 10% test)...")
    train_idx, temp_idx = train_test_split(
        df_final.index,
        test_size=0.20,
        random_state=seed,
        stratify=df_final["topic"],
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=seed,
        stratify=df_final.loc[temp_idx, "topic"],
    )

    df_final["split"] = "train"
    df_final.loc[val_idx, "split"] = "val"
    df_final.loc[test_idx, "split"] = "test"

    logger.info("Distribuição por split:")
    print(df_final["split"].value_counts())

    return df_final


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Constrói o dataset unificado balanceado de tópicos (8 classes)")
    parser.add_argument("--target-per-class", type=int, default=3000, help="Número de amostras por classe (padrão: 3000)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (padrão: 42)")
    args = parser.parse_args()

    df = build_dataset(repo_root=repo_root, target_per_class=args.target_per_class, seed=args.seed)

    out_dir = repo_root / "datasets" / "topic_classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "topic_dataset_8classes.parquet"
    tsv_path = out_dir / "topic_dataset_8classes.tsv"

    logger.info(f"Salvando dataset em {parquet_path}...")
    df.to_parquet(parquet_path, index=False)

    logger.info(f"Salvando dataset em {tsv_path}...")
    df.to_tsv_path = tsv_path
    df.to_csv(tsv_path, sep="\t", index=False)

    # Também atualizamos topic_dataset.parquet e topic_dataset.tsv como versão principal
    main_parquet = out_dir / "topic_dataset.parquet"
    main_tsv = out_dir / "topic_dataset.tsv"
    df.to_parquet(main_parquet, index=False)
    df.to_csv(main_tsv, sep="\t", index=False)
    logger.info(f"Atualizado {main_parquet} e {main_tsv} com sucesso!")

    logger.info("=" * 70)
    logger.info("DATASET DE 8 CLASSES CONSTRUÍDO COM SUCESSO!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

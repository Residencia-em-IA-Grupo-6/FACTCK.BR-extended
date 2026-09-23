#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_community_notes.py

Pipeline de extração e rotulagem de postagens do Twitter/X associadas a
checagens de fatos comunitárias (Twitter Community Notes / Birdwatch),
sem custo de API oficial, utilizando o endpoint público oEmbed.

Separa automaticamente as checagens em dois arquivos por idioma:
  - community_notes_pt.tsv (Português)
  - community_notes_en.tsv (Inglês)

Campos extraídos:
  - tweet_id: ID original do Tweet
  - note_id: ID da checagem do Community Notes
  - claim: Texto original da postagem (obtido via oEmbed)
  - review: Texto da checagem de fatos (summary do Community Notes)
  - classification: Classificação original (MISINFORMED_OR_POTENTIALLY_MISLEADING / NOT_MISLEADING)
  - is_fake: Classificação binária (1 para boato/falso, 0 para verdadeiro/não enganoso)
  - language: pt ou en
  - status: Status da nota (CURRENTLY_RATED_HELPFUL, NEEDS_MORE_RATINGS, etc.)
  - tweet_author: Nome do autor do tweet
  - tweet_url: Link da postagem
  - misleading_reasons: Motivos/tags da checagem
  - sources: Fontes citadas
  - created_at: Data da checagem
"""

import os
import sys
import csv
import json
import time
import html
import random
import logging
import argparse
import datetime
import re
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, Any, List

import requests
from bs4 import BeautifulSoup

try:
    from langdetect import detect as lang_detect
except ImportError:
    lang_detect = None

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CommunityNotesScraper")

# URLs padrão de espelhos públicos do Community Notes
DEFAULT_NOTES_URL = "https://shiruken.github.io/birdwatch-data/notes-00000.tsv"
DEFAULT_STATUS_URL = "https://shiruken.github.io/birdwatch-data/noteStatusHistory-00000.tsv"

OEMBED_ENDPOINT = "https://publish.twitter.com/oembed"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

OUTPUT_COLUMNS = [
    "tweet_id",
    "note_id",
    "claim",
    "review",
    "classification",
    "is_fake",
    "language",
    "status",
    "tweet_author",
    "tweet_url",
    "misleading_reasons",
    "sources",
    "created_at",
]

DISTINCTIVE_PT_WORDS = [
    " não ", " para ", " uma ", " este ", " esta ", " notícia ", " governo ",
    " brasil ", " brasileiro ", " está ", " são ", " pelo ", " pela ", " também ",
    " foram ", " trata-se ", " boato ", " verdade ", " afirmação ", " sobre "
]


class CommunityNotesScraper:
    def __init__(
        self,
        output_dir: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        pt_filename: str = "community_notes_pt.tsv",
        en_filename: str = "community_notes_en.tsv",
        delay: float = 0.3,
        status_filter: str = "all",
        max_retries: int = 3,
    ):
        repo_root = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else repo_root / "datasets"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if checkpoint_file:
            self.checkpoint_path = Path(checkpoint_file)
        else:
            self.checkpoint_path = repo_root / ".community_notes_checkpoint.json"

        self.pt_filepath = self.output_dir / pt_filename
        self.en_filepath = self.output_dir / en_filename

        self.delay = delay
        self.status_filter = status_filter.lower()  # 'helpful', 'balanced', 'all'
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json, text/html, */*",
        })

        # Dicionário de status da nota (note_id -> currentStatus)
        self.note_statuses: Dict[str, str] = {}

        # Carregar checkpoints
        self.checkpoint_data = self._load_checkpoint()
        self.processed_notes: Set[str] = set(self.checkpoint_data.get("processed_notes", []))
        self.processed_tweets: Set[str] = set(self.checkpoint_data.get("processed_tweets", []))
        self.failed_tweets: Set[str] = set(self.checkpoint_data.get("failed_tweets", []))
        self.stats = self.checkpoint_data.get("stats", {"pt": 0, "en": 0, "failed": 0, "skipped": 0})

        # Inicializar cabeçalhos se arquivos não existirem
        self._init_output_files()

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(
                        f"Checkpoint carregado: {len(data.get('processed_notes', []))} notas processadas, "
                        f"{len(data.get('failed_tweets', []))} tweets indisponíveis."
                    )
                    return data
            except Exception as e:
                logger.warning(f"Erro ao ler checkpoint: {e}. Iniciando novo.")
        return {
            "processed_notes": [],
            "processed_tweets": [],
            "failed_tweets": [],
            "stats": {"pt": 0, "en": 0, "failed": 0, "skipped": 0},
        }

    def _save_checkpoint(self):
        data = {
            "processed_notes": list(self.processed_notes),
            "processed_tweets": list(self.processed_tweets),
            "failed_tweets": list(self.failed_tweets),
            "stats": self.stats,
            "last_updated": datetime.datetime.now().isoformat(),
        }
        temp_file = self.checkpoint_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.checkpoint_path)

    def _init_output_files(self):
        for filepath in [self.pt_filepath, self.en_filepath]:
            if not filepath.exists() or filepath.stat().st_size == 0:
                with open(filepath, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(OUTPUT_COLUMNS)
                logger.info(f"Arquivo inicializado com cabeçalhos: {filepath.name}")

    def download_file_if_missing(self, target_path: Path, url: str) -> Path:
        """Baixa o dump de dados caso não esteja presente no disco."""
        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info(f"Arquivo local encontrado: {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
            return target_path

        logger.info(f"Baixando dados de {url} para {target_path}...")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        resp = self.session.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        total_bytes = 0
        with open(target_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)
                    if total_bytes % (5 * 1024 * 1024) == 0:
                        logger.info(f"Baixados {total_bytes / (1024*1024):.1f} MB...")

        logger.info(f"Download concluído: {target_path} ({total_bytes / (1024*1024):.1f} MB)")
        return target_path

    def load_all_note_statuses(self, status_file: Path):
        """Lê o arquivo noteStatusHistory e mapeia noteId para o seu status atual."""
        if not status_file.exists():
            logger.warning(f"Arquivo de status não encontrado: {status_file}.")
            return

        logger.info(f"Carregando histórico de status das notas de {status_file.name}...")
        count = 0
        with open(status_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                note_id = row.get("noteId")
                status = row.get("currentStatus")
                if note_id and status:
                    self.note_statuses[note_id] = status
                count += 1
                if count % 250000 == 0:
                    logger.info(f"Carregados status de {count} notas...")

        logger.info(f"Mapeamento concluído: {len(self.note_statuses)} notas indexadas com status.")

    def fetch_tweet_oembed(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Extrai os dados do tweet através do endpoint oEmbed oficial sem necessidade de chave de API."""
        if tweet_id in self.failed_tweets:
            return None

        oembed_url = f"{OEMBED_ENDPOINT}?url=https://twitter.com/x/status/{tweet_id}"
        
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(self.delay)
                resp = self.session.get(oembed_url, timeout=10)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code in (404, 403, 400):
                    # Tweet deletado, conta suspensa ou privada
                    self.failed_tweets.add(tweet_id)
                    self.stats["failed"] += 1
                    return None
                elif resp.status_code == 429:
                    # Rate limit atingido - aguardar com backoff
                    wait_time = attempt * 5 + random.uniform(1.0, 3.0)
                    logger.warning(f"HTTP 429 (Rate Limit) para tweet {tweet_id}. Aguardando {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.debug(f"HTTP {resp.status_code} para tweet {tweet_id} (tentativa {attempt})")
            except Exception as e:
                logger.debug(f"Erro na requisição oEmbed para {tweet_id}: {e} (tentativa {attempt})")
                time.sleep(1.0)

        # Se falhou após todas as tentativas, marcar para não tentar repetidamente
        self.failed_tweets.add(tweet_id)
        self.stats["failed"] += 1
        return None

    def clean_tweet_text(self, oembed_html: str) -> Tuple[str, Optional[str]]:
        """Extrai texto limpo e atributo de idioma da tag <p> do oEmbed HTML."""
        if not oembed_html:
            return "", None

        soup = BeautifulSoup(oembed_html, "html.parser")
        p_tag = soup.find("p")
        if not p_tag:
            return "", None

        p_lang = p_tag.get("lang")
        
        # Converter quebras de linha <br> para espaços / \n
        for br in p_tag.find_all("br"):
            br.replace_with("\n")

        raw_text = p_tag.get_text()
        clean_text = html.unescape(raw_text).strip()
        # Normalizar espaços múltiplos mantendo quebras de linha limpas
        clean_text = " ".join(clean_text.split())

        return clean_text, p_lang

    def is_candidate_pt(self, summary: str) -> bool:
        """Checa se o resumo possui características de texto em português."""
        if not summary:
            return False
        clean = re.sub(r"https?://\S+", "", summary).strip()
        if len(clean) < 15:
            return False
        lower = f" {clean.lower()} "
        if any(w in lower for w in DISTINCTIVE_PT_WORDS):
            if lang_detect:
                try:
                    return lang_detect(clean) == "pt"
                except Exception:
                    pass
            return True
        return False

    def detect_language(self, tweet_text: str, review_text: str, oembed_lang: Optional[str]) -> Optional[str]:
        """
        Determina o idioma com precisão estrita ('pt' ou 'en').
        """
        clean_rev = re.sub(r"https?://\S+", "", review_text).strip()
        clean_tw = re.sub(r"https?://\S+", "", tweet_text).strip()

        # 1. Avaliar texto da checagem (review) via langdetect
        if lang_detect and len(clean_rev) >= 20:
            try:
                detected = lang_detect(clean_rev)
                if detected in ("pt", "en"):
                    return detected
            except Exception:
                pass

        # 2. Avaliar texto do tweet (claim) via langdetect
        if lang_detect and len(clean_tw) >= 20:
            try:
                detected = lang_detect(clean_tw)
                if detected in ("pt", "en"):
                    return detected
            except Exception:
                pass

        # 3. Fallback no atributo oficial retornado pelo oEmbed
        if oembed_lang in ("pt", "en"):
            return oembed_lang

        return None

    def extract_misleading_tags(self, row: dict) -> str:
        """Mapeia os campos booleanos de tags de desinformação para uma lista legível."""
        tags = []
        mapping = {
            "misleadingFactualError": "factual_error",
            "misleadingManipulatedMedia": "manipulated_media",
            "misleadingOutdatedInformation": "outdated_info",
            "misleadingMissingImportantContext": "missing_context",
            "misleadingUnverifiedClaimAsFact": "unverified_claim",
            "misleadingSatire": "satire",
            "notMisleadingFactuallyCorrect": "factually_correct",
            "notMisleadingClearlySatire": "clearly_satire",
            "notMisleadingPersonalOpinion": "personal_opinion",
        }
        for col, tag_name in mapping.items():
            if row.get(col) == "1":
                tags.append(tag_name)
        return ",".join(tags)

    def format_timestamp(self, ts_millis: str) -> str:
        """Converte millisegundos para formato de data ISO legível."""
        try:
            ts = int(ts_millis) / 1000.0
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ""

    def append_row(self, filepath: Path, row_data: dict):
        """Escreve uma linha diretamente no arquivo TSV correspondente com flush imediato."""
        with open(filepath, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow([row_data.get(col, "") for col in OUTPUT_COLUMNS])
            f.flush()

    def process(
        self,
        notes_file: Path,
        status_file: Optional[Path] = None,
        limit: Optional[int] = None,
        pt_only: bool = False,
        en_only: bool = False,
    ):
        """Executa o pipeline principal de leitura, filtragem e raspagem."""
        if status_file and status_file.exists():
            self.load_all_note_statuses(status_file)

        logger.info(f"Iniciando processamento das notas de {notes_file.name} (Modo: {self.status_filter})...")
        total_scraped = 0
        processed_count = 0

        # Carregar linhas do arquivo TSV
        with open(notes_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        logger.info(f"Total de registros no arquivo de notas: {len(rows)}")

        # Se for pt_only, ordenar priorizando notas que tenham indícios de português
        if pt_only:
            logger.info("Priorizando notas com indicadores de língua portuguesa...")
            rows.sort(key=lambda r: 0 if self.is_candidate_pt(r.get("summary", "")) else 1)

        for row in rows:
            note_id = row.get("noteId")
            tweet_id = row.get("tweetId")
            classification = row.get("classification")
            summary = row.get("summary", "").strip()

            if not note_id or not tweet_id:
                continue

            processed_count += 1

            # Verificar se já processamos esta nota
            if note_id in self.processed_notes:
                continue

            # Obter status atual da nota
            note_status = self.note_statuses.get(note_id, "NEEDS_MORE_RATINGS")

            # Filtrar por status conforme configuração
            if self.status_filter == "helpful":
                if note_status != "CURRENTLY_RATED_HELPFUL":
                    self.processed_notes.add(note_id)
                    continue
            elif self.status_filter == "balanced":
                if classification == "MISINFORMED_OR_POTENTIALLY_MISLEADING" and note_status != "CURRENTLY_RATED_HELPFUL":
                    self.processed_notes.add(note_id)
                    continue
            # No modo 'all', não filtra por status (processa todos)

            # Validar classificação esperada
            if classification not in ("MISINFORMED_OR_POTENTIALLY_MISLEADING", "NOT_MISLEADING"):
                self.processed_notes.add(note_id)
                continue

            # Classificação binária:
            # 1 = fake news / misleading
            # 0 = not misleading / verdadeiro
            is_fake = 1 if classification == "MISINFORMED_OR_POTENTIALLY_MISLEADING" else 0

            # Pré-filtragem de idioma pelo resumo antes de fazer requisição oEmbed
            is_pt = self.is_candidate_pt(summary)
            if pt_only and not is_pt:
                self.processed_notes.add(note_id)
                continue
            if en_only and is_pt:
                self.processed_notes.add(note_id)
                continue

            # Extrair dados do tweet via oEmbed
            oembed_data = self.fetch_tweet_oembed(tweet_id)
            self.processed_notes.add(note_id)

            if not oembed_data:
                # Tweet não disponível / 404
                if processed_count % 25 == 0:
                    self._save_checkpoint()
                continue

            self.processed_tweets.add(tweet_id)
            tweet_text, oembed_lang = self.clean_tweet_text(oembed_data.get("html", ""))
            if not tweet_text:
                continue

            # Detecção e restrição estrita de idioma
            lang = self.detect_language(tweet_text, summary, oembed_lang)
            if not lang:
                self.stats["skipped"] += 1
                continue

            if pt_only and lang != "pt":
                continue
            if en_only and lang != "en":
                continue

            # Preparar registro
            author_name = oembed_data.get("author_name", "")
            tweet_url = f"https://twitter.com/x/status/{tweet_id}"
            reasons = self.extract_misleading_tags(row)
            created_at = self.format_timestamp(row.get("createdAtMillis", ""))

            row_data = {
                "tweet_id": tweet_id,
                "note_id": note_id,
                "claim": tweet_text,
                "review": summary,
                "classification": classification,
                "is_fake": is_fake,
                "language": lang,
                "status": note_status,
                "tweet_author": author_name,
                "tweet_url": tweet_url,
                "misleading_reasons": reasons,
                "sources": row.get("trustworthySources", ""),
                "created_at": created_at,
            }

            # Salvar no arquivo correto
            target_file = self.pt_filepath if lang == "pt" else self.en_filepath
            self.append_row(target_file, row_data)

            self.stats[lang] += 1
            total_scraped += 1

            logger.info(
                f"[{lang.upper()}] Tweet {tweet_id} | is_fake={is_fake} | "
                f"Status: {note_status} | Claim: '{tweet_text[:60]}...' | Review: '{summary[:60]}...'"
            )

            if total_scraped % 10 == 0:
                self._save_checkpoint()
                logger.info(
                    f"Progresso: {self.stats['pt']} PT, {self.stats['en']} EN | "
                    f"{self.stats['failed']} falhas (404s) | {total_scraped} novos extraídos."
                )

            if limit and total_scraped >= limit:
                logger.info(f"Limite solicitado atingido ({limit} checagens extraídas).")
                break

        # Salvar estado final
        self._save_checkpoint()
        logger.info(
            f"Processamento concluído. Estatísticas finais: "
            f"{self.stats['pt']} em Português ({self.pt_filepath.name}), "
            f"{self.stats['en']} em Inglês ({self.en_filepath.name})."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de extração de checagens comunitárias do Twitter (Community Notes) em PT e EN via oEmbed."
    )
    parser.add_argument(
        "--notes-file",
        type=str,
        default=None,
        help="Caminho local para o arquivo notes-*.tsv. Se não fornecido, será baixado automaticamente.",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        default=None,
        help="Caminho local para o arquivo noteStatusHistory-*.tsv. Se não fornecido, será baixado automaticamente.",
    )
    repo_root = Path(__file__).resolve().parent.parent
    default_data_dir = str(repo_root / "data" / "community_notes")
    default_output_dir = str(repo_root / "datasets")

    parser.add_argument(
        "--data-dir",
        type=str,
        default=default_data_dir,
        help="Diretório para armazenar os dumps baixados (padrão: data/community_notes).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Diretório de saída para os arquivos TSV gerados (padrão: datasets/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite máximo de novos tweets/checagens a extrair (útil para testes ou execuções em lotes).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Atalho para --limit N (executar uma amostragem).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Intervalo em segundos entre chamadas oEmbed (padrão: 0.3s).",
    )
    parser.add_argument(
        "--status-filter",
        type=str,
        choices=["helpful", "balanced", "all"],
        default="all",
        help="Filtro de status das notas: 'helpful' (apenas consenso útil), 'balanced' (helpful para fake + qualquer status para true), ou 'all' (todas as notas, recomendado para PT e amostras binárias).",
    )
    parser.add_argument(
        "--pt-only",
        action="store_true",
        help="Extrair apenas checagens em português.",
    )
    parser.add_argument(
        "--en-only",
        action="store_true",
        help="Extrair apenas checagens em inglês.",
    )

    args = parser.parse_args()

    limit = args.sample if args.sample is not None else args.limit

    scraper = CommunityNotesScraper(
        output_dir=args.output_dir,
        delay=args.delay,
        status_filter=args.status_filter,
    )

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Obter ou baixar arquivos necessários
    if args.notes_file:
        notes_path = Path(args.notes_file)
    else:
        notes_path = scraper.download_file_if_missing(
            data_dir / "notes-00000.tsv",
            DEFAULT_NOTES_URL
        )

    if args.status_file:
        status_path = Path(args.status_file)
    else:
        status_path = scraper.download_file_if_missing(
            data_dir / "noteStatusHistory-00000.tsv",
            DEFAULT_STATUS_URL
        )

    scraper.process(
        notes_file=notes_path,
        status_file=status_path,
        limit=limit,
        pt_only=args.pt_only,
        en_only=args.en_only,
    )


if __name__ == "__main__":
    main()

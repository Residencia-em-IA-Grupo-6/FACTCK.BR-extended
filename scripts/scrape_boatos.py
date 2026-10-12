#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_boatos.py

Scraper concorrente unificado para o portal Boatos.org.
Permite coletar matérias e alegações desmentidas por categoria individual ou em lote.

Uso:
  # Coletar categoria específica (ex: saude, esporte, entretenimento, tecnologia, politica)
  python scripts/scrape_boatos.py --category saude --limit 250
  python scripts/scrape_boatos.py --category esporte --limit 250

  # Coletar todas as categorias principais de uma vez
  python scripts/scrape_boatos.py --all --limit 250
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import List, Optional, Set

import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
}

TSV_COLUMNS = [
    'URL',
    'Data',
    'Titulo',
    'Texto_Falso_Original',
    'Resumo_Boato',
    'Desmentido',
    'Tags',
    'Categoria',
]

DEFAULT_CATEGORIES = [
    'saude',
    'politica',
    'esporte',
    'entretenimento',
    'tecnologia',
]

INTERRUPTED = False


def signal_handler(sig, frame):
    global INTERRUPTED
    INTERRUPTED = True
    print("\n[Aviso] Interrupção detectada (Ctrl+C). Finalizando requisições pendentes e salvando...")


signal.signal(signal.SIGINT, signal_handler)


def sanitize_text(text: str) -> str:
    if not text:
        return ''
    cleaned = re.sub(r'[\r\n\t]+', ' ', str(text))
    return cleaned.strip()


def fetch_url(url: str, session: Optional[requests.Session] = None, max_retries: int = 3, timeout: int = 15):
    requester = session if session is not None else requests
    for attempt in range(1, max_retries + 1):
        if INTERRUPTED:
            return None
        try:
            resp = requester.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait_s = attempt * 2.0
                time.sleep(wait_s)
            elif resp.status_code >= 500:
                time.sleep(attempt * 1.5)
            else:
                return resp
        except Exception:
            time.sleep(attempt * 1.0)
    return None


def extract_article(url: str, session: requests.Session, category: str) -> Optional[dict]:
    """Extrai campos estruturados de uma matéria do Boatos.org."""
    resp = fetch_url(url, session=session)
    if not resp or resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.content, 'html.parser')

    # 1. Título
    h1 = soup.find('h1')
    title = sanitize_text(h1.get_text(strip=True)) if h1 else ''

    # 2. Data
    time_tag = soup.find('time')
    date = ''
    if time_tag:
        dt = time_tag.get('datetime', '')
        date = dt[:10] if dt else sanitize_text(time_tag.get_text(strip=True))

    # 3. Tags
    tags = [sanitize_text(t.get_text(strip=True)) for t in soup.find_all('a', rel='tag')]
    tags_str = ', '.join([t for t in tags if t])

    # 4. Texto falso em blockquote
    bqs = soup.find_all('blockquote')
    fake_texts = [sanitize_text(b.get_text(separator=' ', strip=True)) for b in bqs]
    fake_text = ' '.join([t for t in fake_texts if t])

    # 5. Resumo e desmentido
    content_div = soup.find('div', class_='entry-content') or soup.find('article')
    summary = ''
    debunk_paragraphs = []

    if content_div:
        for p in content_div.find_all('p'):
            t = sanitize_text(p.get_text(separator=' ', strip=True))
            if not t:
                continue
            t_low = t.lower()
            if t_low.startswith('boato –') or t_low.startswith('boato -') or t_low.startswith('boato:'):
                summary = t
            else:
                debunk_paragraphs.append(t)

    debunk_text = ' '.join(debunk_paragraphs)
    if not fake_text:
        fake_text = summary

    return {
        'URL': url,
        'Data': date,
        'Titulo': title,
        'Texto_Falso_Original': fake_text,
        'Resumo_Boato': summary,
        'Desmentido': debunk_text,
        'Tags': tags_str,
        'Categoria': category,
    }


def get_urls_from_page(category: str, page_num: int, session: requests.Session) -> List[str]:
    """Retorna links de matérias a partir da listagem da categoria."""
    url = f"https://www.boatos.org/{category}/page/{page_num}/" if page_num > 1 else f"https://www.boatos.org/{category}/"
    resp = fetch_url(url, session=session)
    if not resp or resp.status_code != 200:
        url_query = f"https://www.boatos.org/{category}/?paged={page_num}"
        resp = fetch_url(url_query, session=session)
        if not resp or resp.status_code != 200:
            return []

    soup = BeautifulSoup(resp.content, 'html.parser')
    links = []
    for art in soup.find_all('article'):
        a = art.find('a')
        if a and a.get('href') and a.get('href').startswith('http'):
            links.append(a.get('href'))
    return list(dict.fromkeys(links))


def scrape_category(category: str, output_file: Path, limit: int = 300, workers: int = 8) -> int:
    """Executa a coleta concorrente de uma categoria específica."""
    cat = category.lower().strip()
    print("=" * 70)
    print(f"SCRAPING BOATOS.ORG — CATEGORIA: {cat.upper()}")
    print(f"Destino: {output_file} | Limite: {limit} | Workers: {workers}")
    print("=" * 70)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing_urls: Set[str] = set()

    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                u = row.get('URL')
                if u:
                    existing_urls.add(u.strip())
        print(f"Encontradas {len(existing_urls)} matérias já salvas em cache.")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers * 2, pool_maxsize=workers * 2, max_retries=1)
    session.mount('https://', adapter)

    page = 1
    total_saved = len(existing_urls)
    file_mode = 'a' if existing_urls else 'w'

    with open(output_file, file_mode, encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=TSV_COLUMNS, delimiter='\t')
        if not existing_urls:
            writer.writeheader()
            f_out.flush()

        while total_saved < limit and not INTERRUPTED:
            urls = get_urls_from_page(cat, page, session)
            if not urls:
                print(f"Fim das páginas disponíveis na página {page}.")
                break

            new_urls = [u for u in urls if u not in existing_urls]
            print(f"[Página {page}] {len(urls)} links encontrados ({len(new_urls)} novos).")

            if not new_urls:
                page += 1
                continue

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(extract_article, u, session, cat): u for u in new_urls}
                for fut in as_completed(futures):
                    if INTERRUPTED:
                        break
                    res = fut.result()
                    if res and res.get('Titulo'):
                        writer.writerow(res)
                        f_out.flush()
                        existing_urls.add(res['URL'])
                        total_saved += 1
                        if total_saved >= limit:
                            break

            print(f"  -> Progresso: {total_saved}/{limit}")
            page += 1
            time.sleep(0.3)

    print(f"[OK] Categoria '{cat}' finalizada com {total_saved} registros em {output_file}\n")
    return total_saved


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Scraper concorrente unificado do Boatos.org")
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Categoria específica (ex: saude, esporte, politica, tecnologia, entretenimento)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Coleta sequencial de todas as categorias padrão",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=300,
        help="Limite máximo de matérias por categoria (padrão: 300)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Número de threads simultâneas para requisições HTTP (padrão: 8)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Caminho personalizado para o arquivo TSV de saída",
    )

    args = parser.parse_args()

    if args.all:
        for cat in DEFAULT_CATEGORIES:
            out = Path(args.output) if args.output else repo_root / "datasets" / f"boatos_{cat}.tsv"
            scrape_category(cat, out, limit=args.limit, workers=args.workers)
            if INTERRUPTED:
                break
    elif args.category:
        cat = args.category.lower().strip()
        out = Path(args.output) if args.output else repo_root / "datasets" / f"boatos_{cat}.tsv"
        scrape_category(cat, out, limit=args.limit, workers=args.workers)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_ebc.py

Scraper concorrente unificado para notícias autênticas da Agência Brasil (EBC).
Suporta coleta por categoria individual (saude, esportes, cultura, economia, politica, geral, direitos-humanos)
ou em lote para balanceamento de datasets.

Uso:
  # Coletar categoria específica
  python scripts/scrape_ebc.py --category saude --limit 250
  python scripts/scrape_ebc.py --category esportes --limit 250
  python scripts/scrape_ebc.py --category politica --limit 250

  # Coletar todas as categorias padrão de uma vez
  python scripts/scrape_ebc.py --all --limit 250
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import html
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import List, Optional, Set
from urllib.parse import urljoin

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
    'Connection': 'keep-alive',
}

TSV_COLUMNS = [
    'URL',
    'Data',
    'Titulo',
    'Subtitulo',
    'Conteudo',
    'Author',
    'Categoria',
    'is_fake',
]

DEFAULT_CATEGORIES = [
    'saude',
    'politica',
    'esportes',
    'cultura',
    'economia',
    'geral',
    'direitos-humanos',
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
    cleaned = html.unescape(str(text))
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
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
    """Extrai campos estruturados de uma matéria jornalística da Agência Brasil."""
    resp = fetch_url(url, session=session)
    if not resp or resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.content, 'html.parser')

    # 1. Título
    h1 = soup.find('h1')
    title = sanitize_text(h1.get_text(strip=True)) if h1 else ''
    if not title:
        return None

    # 2. Subtítulo (Linha fina)
    lead = (
        soup.find('p', class_='lead')
        or soup.find('div', class_='field--name-field-lead')
        or soup.find('div', class_='article-lead')
        or soup.find('h2', class_='lead')
    )
    subtitle = sanitize_text(lead.get_text(strip=True)) if lead else ''

    # 3. Data de publicação
    date = ''
    time_tag = soup.find('time')
    if time_tag:
        dt = time_tag.get('datetime', '')
        if dt:
            date = dt[:10]
        else:
            date = sanitize_text(time_tag.get_text(strip=True))[:10]

    if not date:
        date_span = soup.find('span', class_='date') or soup.find('span', class_='published-date')
        if date_span:
            raw_d = sanitize_text(date_span.get_text(strip=True))
            m = re.search(r'(\d{2})/(\d{2})/(\d{4})', raw_d)
            if m:
                date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    # 4. Conteúdo íntegro do corpo jornalístico
    body_container = (
        soup.find('div', class_='field--name-body')
        or soup.find('div', class_='article-content')
        or soup.find('div', class_='node__content')
    )
    paragraphs = []
    if body_container:
        for p in body_container.find_all('p'):
            t = sanitize_text(p.get_text(strip=True))
            if t and not t.startswith('Edição:') and not t.startswith('Publicado em'):
                paragraphs.append(t)
    conteudo = ' '.join(paragraphs)

    # 5. Autor
    author = 'Agência Brasil'
    author_elem = soup.find('span', class_='author') or soup.find('span', class_='field--name-uid')
    if author_elem:
        a_txt = sanitize_text(author_elem.get_text(strip=True))
        if a_txt:
            author = f"Agência Brasil ({a_txt})"

    return {
        'URL': url,
        'Data': date,
        'Titulo': title,
        'Subtitulo': subtitle,
        'Conteudo': conteudo,
        'Author': author,
        'Categoria': category,
        'is_fake': False,
    }


def get_urls_from_page(category: str, page_num: int, session: requests.Session) -> List[str]:
    """Retorna URLs da página de listagem da categoria na EBC (paginação Drupal: page=0, 1, 2...)."""
    base_cat_url = f"https://agenciabrasil.ebc.com.br/{category}"
    url = f"{base_cat_url}?page={page_num}" if page_num > 0 else base_cat_url

    resp = fetch_url(url, session=session)
    if not resp or resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.content, 'html.parser')
    links = []

    for art in soup.find_all(['article', 'div'], class_=re.compile(r'(views-row|node--type-noticia|teaser)')):
        a = art.find('a')
        if a and a.get('href'):
            full_url = urljoin('https://agenciabrasil.ebc.com.br', a.get('href'))
            if f'/{category}/' in full_url or full_url.startswith('https://agenciabrasil.ebc.com.br/'):
                links.append(full_url)

    if not links:
        for a in soup.find_all('a'):
            href = a.get('href', '')
            if href and f'/{category}/' in href and not href.endswith(f'/{category}'):
                full_url = urljoin('https://agenciabrasil.ebc.com.br', href)
                links.append(full_url)

    return list(dict.fromkeys(links))


def scrape_category(category: str, output_file: Path, limit: int = 300, workers: int = 8) -> int:
    """Executa a coleta concorrente de uma editoria específica da Agência Brasil."""
    cat = category.lower().strip()
    print("=" * 70)
    print(f"SCRAPING EBC / AGÊNCIA BRASIL — CATEGORIA: {cat.upper()}")
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

    page = 0
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
                print(f"Fim das matérias disponíveis na página {page}.")
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
    parser = argparse.ArgumentParser(description="Scraper concorrente unificado da Agência Brasil / EBC")
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Categoria específica (ex: saude, esportes, cultura, economia, politica, geral, direitos-humanos)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Coleta sequencial de todas as editorias padrão da EBC",
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
            out = Path(args.output) if args.output else repo_root / "datasets" / f"ebc_{cat}.tsv"
            scrape_category(cat, out, limit=args.limit, workers=args.workers)
            if INTERRUPTED:
                break
    elif args.category:
        cat = args.category.lower().strip()
        out = Path(args.output) if args.output else repo_root / "datasets" / f"ebc_{cat}.tsv"
        scrape_category(cat, out, limit=args.limit, workers=args.workers)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

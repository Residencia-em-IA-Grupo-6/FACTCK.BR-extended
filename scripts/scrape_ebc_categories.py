#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_ebc_categories.py

Scraper paralelo de alta performance para coletar notícias autênticas da Agência Brasil
(EBC) por categorias editoriais (esportes, cultura, economia, politica, geral).
Utilizado para balancear as classes temáticas e prover exemplos com is_fake=False.

Uso:
  python scripts/scrape_ebc_categories.py --category esportes --limit 250
  python scripts/scrape_ebc_categories.py --category cultura --limit 250
  python scripts/scrape_ebc_categories.py --category economia --limit 250
  python scripts/scrape_ebc_categories.py --category geral --limit 250
"""

import os
import re
import csv
import time
import html
import signal
import argparse
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
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
    'is_fake'
]

INTERRUPTED = False


def signal_handler(sig, frame):
    global INTERRUPTED
    INTERRUPTED = True
    print("\n[Aviso] Interrupção detectada (Ctrl+C). Finalizando requisições pendentes...")


signal.signal(signal.SIGINT, signal_handler)


def sanitize_text(text: str) -> str:
    if not text:
        return ''
    cleaned = re.sub(r'[\r\n\t]+', ' ', str(text))
    return cleaned.strip()


def parse_ebc_date(raw_date: str) -> str:
    if not raw_date:
        return ''
    m = re.search(r'(\d{2})/(\d{2})/(\d{4})', raw_date)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        return f"{year}-{month}-{day}"
    return ''


def fetch_url(url: str, session: requests.Session = None, max_retries: int = 3, timeout: int = 12):
    requester = session if session is not None else requests
    for attempt in range(1, max_retries + 1):
        if INTERRUPTED:
            return None
        try:
            resp = requester.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                time.sleep(attempt * 2.0)
            elif resp.status_code >= 500:
                time.sleep(attempt * 1.5)
            else:
                return resp
        except Exception:
            time.sleep(attempt * 1.0)
    return None


def extract_page_links(html_content: str, category: str, base_url: str):
    """Extrai os links das notícias a partir da página de listagem."""
    seen = set()
    links = []
    pattern = re.compile(rf'href=[\"\'](/{category}/noticia/\d{{4}}-\d{{2}}/[^\"]+)[\"\']', re.IGNORECASE)
    for match in pattern.finditer(html_content):
        rel_url = match.group(1).strip()
        full_url = urljoin(base_url, rel_url)
        if full_url not in seen:
            seen.add(full_url)
            links.append(full_url)
    return links


def extract_article(url: str, session: requests.Session, category: str):
    """Extrai os metadados e o conteúdo de uma matéria da EBC."""
    resp = fetch_url(url, session=session)
    if not resp or resp.status_code != 200:
        return None

    html_content = resp.text

    # 1. Título
    m_t = re.search(r'<h1[^>]*class=[\"\']titulo-materia[\"\'][^>]*>([\s\S]*?)</h1>', html_content, re.IGNORECASE)
    if not m_t:
        m_t = re.search(r'<h1[^>]*>([\s\S]*?)</h1>', html_content, re.IGNORECASE)
    title = html.unescape(re.sub(r'<[^>]+>', ' ', m_t.group(1))).strip() if m_t else ''

    # 2. Subtítulo / Linha fina
    m_sub = re.search(r'<div[^>]*class=[\"\']linha-fina-noticia[\"\'][^>]*>([\s\S]*?)</div>', html_content, re.IGNORECASE)
    subtitle = html.unescape(re.sub(r'<[^>]+>', ' ', m_sub.group(1))).strip() if m_sub else ''

    # 3. Data
    m_date = re.search(r'Publicado\s+em\s+(\d{2}/\d{2}/\d{4}(?:\s*-\s*\d{2}:\d{2})?)', html_content, re.IGNORECASE)
    raw_date = m_date.group(1) if m_date else ''
    date = parse_ebc_date(raw_date)

    # 4. Conteúdo textual
    soup = BeautifulSoup(html_content, 'html.parser')
    body_div = soup.find('div', class_='conteudo-materia') or soup.find('div', class_='field--name-body')
    content_text = ''
    if body_div:
        # Remover blocos indesejados
        for tag in body_div.find_all(['script', 'style', 'figure', 'aside']):
            tag.decompose()
        paragraphs = [sanitize_text(p.get_text(separator=' ', strip=True)) for p in body_div.find_all('p')]
        content_text = ' '.join([p for p in paragraphs if p])
    else:
        paragraphs = [sanitize_text(p.get_text(separator=' ', strip=True)) for p in soup.find_all('p')]
        content_text = ' '.join([p for p in paragraphs if p])

    if not title:
        return None

    return {
        'URL': url,
        'Data': date,
        'Titulo': sanitize_text(title),
        'Subtitulo': sanitize_text(subtitle),
        'Conteudo': sanitize_text(content_text),
        'Author': 'Agência Brasil',
        'Categoria': category,
        'is_fake': 'False'
    }


def scrape_category(category: str, output_file: Path, limit: int = 300, workers: int = 8):
    base_url = f'https://agenciabrasil.ebc.com.br/{category}'
    print("=" * 70)
    print(f"SCRAPING AGÊNCIA BRASIL (EBC) — CATEGORIA: {category.upper()}")
    print(f"URL: {base_url} | Destino: {output_file} | Limite: {limit} | Workers: {workers}")
    print("=" * 70)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing_urls = set()
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row.get('URL'):
                    existing_urls.add(row['URL'])
        print(f"Encontradas {len(existing_urls)} matérias já salvas.")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=1)
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
            page_url = f"{base_url}?page={page}" if page > 0 else base_url
            resp = fetch_url(page_url, session=session)
            if not resp or resp.status_code != 200:
                print(f"Fim das páginas ou erro ao acessar {page_url}.")
                break

            urls = extract_page_links(resp.text, category, base_url)
            if not urls:
                print(f"Nenhum link encontrado na página {page}. Encerrando.")
                break

            new_urls = [u for u in urls if u not in existing_urls]
            print(f"[Página {page}] {len(urls)} links encontrados ({len(new_urls)} novos).")

            if not new_urls:
                page += 1
                continue

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(extract_article, u, session, category): u for u in new_urls}
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

            print(f"  -> Total coletado até agora: {total_saved}/{limit}")
            page += 1
            time.sleep(0.3)

    print(f"\n[OK] Coleta de {category} concluída! Total salvo: {total_saved} em {output_file}\n")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Scraper multi-categoria da Agência Brasil (EBC)")
    parser.add_argument("--category", "-c", required=True, help="Categoria (ex: esportes, cultura, economia, politica, geral)")
    parser.add_argument("--limit", "-l", type=int, default=300, help="Limite de matérias a coletar")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Número de threads simultâneas")
    parser.add_argument("--output", "-o", type=str, default=None, help="Caminho do arquivo TSV de saída")
    args = parser.parse_args()

    cat = args.category.lower().strip()
    out = Path(args.output) if args.output else repo_root / "datasets" / f"ebc_{cat}.tsv"
    scrape_category(category=cat, output_file=out, limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_noticias_ms.py

Scraper oficial de notícias de saúde do Ministério da Saúde (gov.br/saude).
Portal: https://www.gov.br/saude/pt-br/assuntos/noticias-ms

Recursos:
- Paginação automática do Plone CMS (b_start:int=0, 15, 30, ...).
- Paralelismo multi-thread de alta velocidade via ThreadPoolExecutor (--workers, -w).
- Pool de conexões HTTP Keep-Alive reutilizáveis para máxima vazão de requisições.
- Extração de metadados ricos: JSON-LD (NewsArticle), título, linha fina,
  data de publicação/modificação, editoria temática e corpo integral limpo.
- Salvamento atômico incremental thread-safe em TSV/CSV.
- Checkpoint e retomada (resume) transparente para evitar duplicações e requisições repetidas.
- Tolerância a falhas com retries e backoff exponencial.
- Controle gracioso de interrupção via Ctrl+C (SIGINT).
- Suporte opcional à subseção regional: noticias-para-os-estados.
- Formatos de saída: 'raw' (completo) e 'factckbr' (compatível com a base consolidada de checagem).

LIMITAÇÃO TÉCNICA DO PORTAL GOV.BR/SAUDE:
O feed oficial 'noticias-ms' possui uma janela de retenção recente (rolling collection)
no CMS Plone, disponibilizando apenas cerca de 18 páginas (~264 a 270 matérias mais recentes,
cobrindo aproximadamente os últimos 2 a 3 meses, retroagindo apenas até julho).
Não há suporte a navegação histórica de anos anteriores nesta coleção específica.
Para volumes históricos maiores de comunicação pública governamental em saúde,
recomenda-se o uso de fontes como Agência Brasil (EBC) e Agência Fiocruz.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import html
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
from urllib.parse import urljoin

import requests

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


BASE_URL = 'https://www.gov.br/saude/pt-br/assuntos/noticias-ms'
ESTADOS_URL = 'https://www.gov.br/saude/pt-br/assuntos/noticias-ms/noticias-para-os-estados'

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

RAW_COLUMNS = [
    'URL',
    'Data',
    'Data_Hora',
    'Data_Modificacao',
    'Titulo',
    'Subtitulo',
    'Categoria',
    'Conteudo',
    'Author',
    'is_fake'
]

FACTCKBR_COLUMNS = [
    'URL',
    'Data',
    'Titulo',
    'Author',
    'Claim',
    'reviewBody',
    'is_fake'
]

INTERRUPTED = False
FILE_LOCK = threading.Lock()


def signal_handler(sig, frame):
    """Trata interrupções do usuário (Ctrl+C) de forma graciosa."""
    global INTERRUPTED
    INTERRUPTED = True
    print('\n[AVISO] Interrupção detectada (Ctrl+C). Finalizando tarefas ativas e salvando o estado...')


signal.signal(signal.SIGINT, signal_handler)


def sanitize_text(text: str) -> str:
    """Remove quebras de linha e tabulações para preservar a integridade do formato TSV/CSV."""
    if not text:
        return ''
    cleaned = re.sub(r'[\r\n\t]+', ' ', str(text))
    return cleaned.strip()


def parse_date_to_iso(date_str: str) -> str:
    """Converte strings de datas do padrão brasileiro (DD/MM/YYYY) para ISO (YYYY-MM-DD)."""
    if not date_str:
        return ''
    m_iso = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"
    m_br = re.search(r'(\d{2})/(\d{2})/(\d{4})', date_str)
    if m_br:
        return f"{m_br.group(3)}-{m_br.group(2)}-{m_br.group(1)}"
    return date_str.strip()


def create_session(workers: int = 6) -> requests.Session:
    """Cria sessão HTTP com pooling de conexões Keep-Alive para requisições concorrentes."""
    session = requests.Session()
    pool_size = max(12, workers * 2)
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=1
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update(HEADERS)
    return session


def fetch_url(url: str, session: requests.Session = None, max_retries: int = 3, backoff_factor: float = 1.5, timeout: int = 15):
    """Executa requisição HTTP com tentativas e backoff exponencial."""
    requester = session if session is not None else requests
    for attempt in range(1, max_retries + 1):
        if INTERRUPTED:
            return None
        try:
            resp = requester.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait_time = (backoff_factor ** attempt) * 2.0
                print(f"    [Rate limit 429] Aguardando {wait_time:.1f}s antes de retentar...")
                time.sleep(wait_time)
            elif resp.status_code >= 500:
                wait_time = attempt * 2.0
                print(f"    [Erro servidor {resp.status_code}] Tentativa {attempt}/{max_retries}. Aguardando {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                return resp
        except (requests.RequestException, Exception) as e:
            if attempt == max_retries:
                print(f"    [Falha de conexão] Não foi possível acessar {url}: {e}")
                return None
            time.sleep(attempt * 1.0)
    return None


def extract_listing_cards(html_content: str, base_url: str):
    """
    Extrai as notícias da listagem da página no padrão Plone CMS do Portal gov.br.
    Retorna uma lista de dicionários com os metadados preliminares e informações da paginação.
    """
    cards = []
    seen_urls = set()

    item_pattern = re.compile(
        r'<article[^>]*class=[\"\'][^\"\']*entry[^\"\']*[\"\'][^>]*>([\s\S]*?)</article>',
        re.IGNORECASE
    )
    items = item_pattern.findall(html_content)

    if not items:
        item_pattern = re.compile(
            r'<li[^>]*class=[\"\'][^\"\']*tileItem[^\"\']*[\"\'][^>]*>([\s\S]*?)</li>',
            re.IGNORECASE
        )
        items = item_pattern.findall(html_content)

    for item_html in items:
        link_m = re.search(r'<a[^>]*href=[\"\']([^\"]+)[\"\'][^>]*>([\s\S]*?)</a>', item_html, re.IGNORECASE)
        if not link_m:
            continue

        raw_url = link_m.group(1).strip()
        full_url = urljoin(base_url, raw_url)

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        title = html.unescape(re.sub(r'<[^>]+>', ' ', link_m.group(2))).strip()

        desc_m = re.search(r'<span[^>]*class=[\"\'][^\"\']*description[^\"\']*[\"\'][^>]*>([\s\S]*?)</span>', item_html, re.IGNORECASE)
        subtitle = html.unescape(re.sub(r'<[^>]+>', ' ', desc_m.group(1))).strip() if desc_m else ''

        date_m = re.search(r'(\d{2}/\d{2}/\d{4})', item_html)
        date_str = date_m.group(1) if date_m else ''

        cat_m = re.search(r'<span[^>]*class=[\"\'][^\"\']*subtitle[^\"\']*[\"\'][^>]*>([\s\S]*?)</span>', item_html, re.IGNORECASE)
        category = html.unescape(re.sub(r'<[^>]+>', ' ', cat_m.group(1))).strip() if cat_m else ''

        cards.append({
            'url': full_url,
            'title': title,
            'subtitle': subtitle,
            'date': date_str,
            'category': category
        })

    next_pattern = re.compile(r'href=[\"\'][^\"\']*b_start:int=(\d+)[\"\'][^>]*>(?:Próximo|Next|>|»)', re.IGNORECASE)
    next_m = next_pattern.search(html_content)
    next_offset = int(next_m.group(1)) if next_m else None
    has_next = next_offset is not None

    return cards, next_offset, has_next


def extract_article_page(html_content: str, url: str, fallback_card: dict = None) -> dict:
    """
    Extrai o conteúdo integral e metadados detalhados de uma matéria no portal gov.br/saude.
    """
    card = fallback_card or {}
    title = card.get('title', '')
    subtitle = card.get('subtitle', '')
    date_iso = parse_date_to_iso(card.get('date', ''))
    datetime_iso = ''
    date_modified = ''
    category = card.get('category', '')
    author = 'Ministério da Saúde'
    body_text = ''

    # Tentativa 1: Metadados estruturados JSON-LD (NewsArticle / WebPage)
    json_ld_matches = re.findall(r'<script[^>]*type=[\"\']application/ld\+json[\"\'][^>]*>([\s\S]*?)</script>', html_content, re.IGNORECASE)
    for j_text in json_ld_matches:
        try:
            data = json.loads(j_text.strip())
            items = data if isinstance(data, list) else [data]
            for item in items:
                schema_type = str(item.get('@type', ''))
                if 'NewsArticle' in schema_type or 'Article' in schema_type:
                    title = item.get('headline') or title
                    subtitle = item.get('description') or subtitle
                    dp = item.get('datePublished', '')
                    dm = item.get('dateModified', '')
                    if dp:
                        datetime_iso = dp
                        date_iso = dp[:10]
                    if dm:
                        date_modified = dm
                    aut = item.get('author')
                    if isinstance(aut, dict):
                        author = aut.get('name') or author
                    elif isinstance(aut, list) and aut:
                        author = aut[0].get('name') if isinstance(aut[0], dict) else str(aut[0])
                    elif isinstance(aut, str) and aut:
                        author = aut
                    article_body = item.get('articleBody', '')
                    if article_body and len(article_body) > len(body_text):
                        body_text = article_body
                    break
        except Exception:
            pass

    # Tentativa 2: Extração visual no HTML
    if not title:
        m_t = re.search(r'<h1[^>]*class=[\"\'][^\"\']*documentFirstHeading[^\"\']*[\"\'][^>]*>([\s\S]*?)</h1>', html_content, re.IGNORECASE)
        if not m_t:
            m_t = re.search(r'<h1[^>]*>([\s\S]*?)</h1>', html_content, re.IGNORECASE)
        title = html.unescape(re.sub(r'<[^>]+>', ' ', m_t.group(1))).strip() if m_t else ''

    if not subtitle:
        m_s = re.search(r'<div[^>]*class=[\"\'][^\"\']*documentDescription[^\"\']*[\"\'][^>]*>([\s\S]*?)</div>', html_content, re.IGNORECASE)
        subtitle = html.unescape(re.sub(r'<[^>]+>', ' ', m_s.group(1))).strip() if m_s else ''

    if not date_iso:
        m_d = re.search(r'(\d{2}/\d{2}/\d{4})(?:\s*às\s*(\d{2}h\d{2}))?', html_content)
        if m_d:
            date_iso = parse_date_to_iso(m_d.group(1))
            if m_d.group(2):
                h, m = m_d.group(2).replace('h', ':'), '00'
                datetime_iso = f"{date_iso}T{h}:{m}"

    if not body_text:
        body_m = re.search(r'<div[^>]*id=[\"\']parent-fieldname-text[\"\'][^>]*>([\s\S]*?)</div>\s*<!--', html_content, re.IGNORECASE)
        if not body_m:
            body_m = re.search(r'<div[^>]*property=[\"\']rnews:articleBody[\"\'][^>]*>([\s\S]*?)</div>', html_content, re.IGNORECASE)
        if not body_m:
            body_m = re.search(r'<div[^>]*class=[\"\'][^\"\']*newsImageContainer[\"\'][^>]*>[\s\S]*?</div>([\s\S]*?)<div[^>]*class=[\"\'][^\"\']*visualClear', html_content, re.IGNORECASE)

        raw_body_html = body_m.group(1) if body_m else html_content

        paragraphs = re.findall(r'<p[^>]*>([\s\S]*?)</p>', raw_body_html, re.IGNORECASE)
        cleaned_paras = []
        for p in paragraphs:
            cp = html.unescape(re.sub(r'<[^>]+>', ' ', p)).strip()
            cp = re.sub(r'\s+', ' ', cp)
            if len(cp) > 20 and not cp.startswith('Foto:') and not cp.startswith('Por '):
                cleaned_paras.append(cp)

        body_text = ' '.join(cleaned_paras)

    return {
        'URL': url,
        'Data': sanitize_text(date_iso),
        'Data_Hora': sanitize_text(datetime_iso),
        'Data_Modificacao': sanitize_text(date_modified),
        'Titulo': sanitize_text(title),
        'Subtitulo': sanitize_text(subtitle),
        'Categoria': sanitize_text(category),
        'Conteudo': sanitize_text(body_text),
        'Author': sanitize_text(author),
        'is_fake': False
    }


def fetch_and_parse_article_ms(card: dict, session: requests.Session, format_mode: str = 'raw'):
    """Worker paralelo para carregar e extrair uma matéria do Ministério da Saúde."""
    if INTERRUPTED:
        return None

    url = card['url']
    art_resp = fetch_url(url, session=session)
    if not art_resp or art_resp.status_code != 200:
        return None

    art_data = extract_article_page(art_resp.text, url, card)

    if format_mode == 'factckbr':
        claim_text = art_data['Subtitulo'] if art_data['Subtitulo'] else art_data['Titulo']
        record = {
            'URL': art_data['URL'],
            'Data': art_data['Data'],
            'Titulo': art_data['Titulo'],
            'Author': art_data['Author'],
            'Claim': claim_text,
            'reviewBody': art_data['Conteudo'],
            'is_fake': False
        }
    else:
        record = art_data

    return record


def load_existing_urls(file_path: str) -> set:
    """Carrega as URLs já extraídas do arquivo TSV/CSV para evitar duplicatas."""
    if not os.path.exists(file_path):
        return set()

    urls = set()
    try:
        if HAS_PANDAS:
            sep = '\t' if file_path.endswith('.tsv') else ','
            df = pd.read_csv(file_path, sep=sep, dtype=str, keep_default_na=False)
            if 'URL' in df.columns:
                return set(df['URL'].dropna().str.strip().tolist())
        else:
            sep = '\t' if file_path.endswith('.tsv') else ','
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=sep)
                for row in reader:
                    url = row.get('URL', '').strip()
                    if url:
                        urls.add(url)
    except Exception as e:
        print(f"[Aviso] Falha ao ler URLs existentes de {file_path}: {e}")
    return urls


def load_checkpoint(checkpoint_file: str) -> dict:
    """Lê o estado do checkpoint JSON."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_checkpoint(checkpoint_file: str, data: dict):
    """Grava o progresso atual no arquivo de checkpoint com proteção por lock."""
    with FILE_LOCK:
        try:
            Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            data['timestamp'] = datetime.now().isoformat()
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Aviso] Falha ao salvar checkpoint em {checkpoint_file}: {e}")


def append_rows(file_path: str, records: list, columns: list):
    """Acrescenta registros atomicamente ao arquivo de saída com flush forçado."""
    if not records:
        return
    with FILE_LOCK:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0

        sep = '\t' if file_path.endswith('.tsv') else ','
        with open(file_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=sep, extrasaction='ignore')
            if is_new:
                writer.writeheader()
            for record in records:
                writer.writerow(record)
            f.flush()


def append_row(file_path: str, record: dict, columns: list):
    """Acrescenta um único registro ao arquivo (wrapper para append_rows)."""
    append_rows(file_path, [record], columns)


def scrape_section(section_name: str, base_url: str, output_path: str, format_mode: str,
                   checkpoint_file: str, max_pages: int, delay: float, resume: bool,
                   existing_urls: set, checkpoint_data: dict, workers: int = 6,
                   session: requests.Session = None) -> int:
    """
    Executa a raspagem paralela de uma seção de notícias (nacional ou estados).
    Retorna a quantidade de novas matérias salvas.
    """
    columns = FACTCKBR_COLUMNS if format_mode == 'factckbr' else RAW_COLUMNS
    section_cp = checkpoint_data.get(section_name, {})

    start_offset = 0
    if resume and section_cp.get('last_offset') is not None:
        start_offset = section_cp.get('last_offset')
        print(f"[{section_name}] Retomando a partir do offset b_start:int={start_offset}")

    current_offset = start_offset
    page_count = 0
    new_saved = 0

    print(f"\n" + "=" * 70)
    print(f"INICIANDO RASPAGEM: {section_name} [PARALELO: {workers} workers]")
    print(f"URL Base: {base_url}")
    print(f"Arquivo de Saída: {output_path}")
    print("=" * 70)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            if INTERRUPTED:
                print("\n[Parada solicitada] Interrompendo coleta da seção...")
                break

            if max_pages > 0 and page_count >= max_pages:
                print(f"[{section_name}] Limite máximo de {max_pages} página(s) atingido.")
                break

            page_count += 1
            t_start = time.time()
            page_url = f"{base_url}?b_start:int={current_offset}" if current_offset > 0 else base_url
            print(f"\n-> Página {page_count} (offset={current_offset}): {page_url}")

            resp = fetch_url(page_url, session=session)
            if not resp or resp.status_code != 200:
                print(f"   [Falha] Código HTTP {getattr(resp, 'status_code', 'Sem resposta')} ao obter listagem. Finalizando seção.")
                break

            cards, next_offset, has_next = extract_listing_cards(resp.text, base_url)
            if not cards:
                print(f"   [Fim] Nenhuma notícia encontrada na página. Encerrando paginação.")
                break

            new_cards = [c for c in cards if c['url'] not in existing_urls]
            print(f"   Encontradas {len(cards)} matéria(s) na página ({len(new_cards)} novas, {len(cards) - len(new_cards)} já salvas).")

            if new_cards:
                futures = {
                    executor.submit(fetch_and_parse_article_ms, card, session, format_mode): (idx, card)
                    for idx, card in enumerate(new_cards, start=1)
                }

                results = {}
                for future in as_completed(futures):
                    if INTERRUPTED:
                        break
                    idx, card = futures[future]
                    try:
                        record = future.result()
                        if record:
                            results[idx] = record
                    except Exception as e:
                        print(f"   [Erro] Falha em {card['url']}: {e}")

                sorted_records = [results[i] for i in sorted(results.keys())]

                if sorted_records:
                    append_rows(output_path, sorted_records, columns)
                    for r in sorted_records:
                        existing_urls.add(r['URL'])
                    new_saved += len(sorted_records)
                    for r in sorted_records:
                        print(f"   + [{r['Data']}] {r['Titulo'][:65]}...")

            dur = time.time() - t_start
            print(f"   ✓ Página concluída em {dur:.2f}s (Total acumulado: {len(existing_urls)})")

            checkpoint_data[section_name] = {
                'last_offset': current_offset,
                'page_count': page_count,
                'total_saved': new_saved
            }
            save_checkpoint(checkpoint_file, checkpoint_data)

            if not has_next or next_offset is None or next_offset <= current_offset:
                print(f"[{section_name}] Fim do catálogo atingido (sem próxima página).")
                break

            current_offset = next_offset

            if delay > 0:
                time.sleep(delay)

    return new_saved


def main():
    parser = argparse.ArgumentParser(
        description="Scraper oficial do portal Notícias da Saúde (Ministério da Saúde - gov.br) [PARALELO]"
    )
    repo_root = Path(__file__).resolve().parent.parent
    default_output = repo_root / "datasets" / "noticias_ms.tsv"
    default_checkpoint = repo_root / "datasets" / ".checkpoint_noticias_ms.json"

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(default_output),
        help=f"Caminho do arquivo TSV/CSV de saída (Padrão: {default_output})"
    )
    parser.add_argument(
        "--format",
        choices=["raw", "factckbr"],
        default="raw",
        help="Formato dos dados: 'raw' (metadados completos) ou 'factckbr' (compatível com a base consolidada)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=6,
        help="Quantidade de threads concorrentes para raspagem das matérias (Padrão: 6)"
    )
    parser.add_argument(
        "--max-pages", "-p",
        type=int,
        default=0,
        help="Quantidade máxima de páginas por seção (0 = raspar todas as páginas disponíveis)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.1,
        help="Intervalo em segundos entre páginas para rate limiting educado (Padrão: 0.1s)"
    )
    parser.add_argument(
        "--include-estados",
        action="store_true",
        help="Inclui também a subseção 'noticias-para-os-estados' além do feed principal"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_checkpoint),
        help=f"Arquivo de checkpoint JSON (Padrão: {default_checkpoint})"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora o checkpoint existente e inicia do zero"
    )

    args = parser.parse_args()

    print("=" * 75)
    print("SCRAPER DE NOTÍCIAS DA SAÚDE — MINISTÉRIO DA SAÚDE (GOV.BR) [PARALELO]")
    print(f"  Modo de Formato    : {args.format}")
    print(f"  Arquivo de Saída   : {args.output}")
    print(f"  Workers Paralelos  : {args.workers}")
    print(f"  Delay entre Páginas: {args.delay}s")
    print(f"  Limite de Páginas  : {'Todas disponíveis' if args.max_pages == 0 else args.max_pages}")
    print(f"  Incluir Estados    : {'Sim' if args.include_estados else 'Não'}")
    print(f"  Checkpoint         : {args.checkpoint}")
    print("  [Nota de Limitação]: O portal gov.br/saude mantém janela de retenção de")
    print("                       ~2 a 3 meses (até julho). Não possui arquivo histórico extenso.")
    print("=" * 75)

    existing_urls = set()
    if not args.no_resume:
        existing_urls = load_existing_urls(args.output)
        if existing_urls:
            print(f"[Retomada] {len(existing_urls)} matéria(s) já presentes no arquivo de saída.")

    checkpoint_data = {}
    if not args.no_resume:
        checkpoint_data = load_checkpoint(args.checkpoint)

    session = create_session(workers=args.workers)
    total_new = 0

    # 1. Seção Principal: Notícias Nacionais
    new_ms = scrape_section(
        section_name="noticias_nacionais",
        base_url=BASE_URL,
        output_path=args.output,
        format_mode=args.format,
        checkpoint_file=args.checkpoint,
        max_pages=args.max_pages,
        delay=args.delay,
        resume=(not args.no_resume),
        existing_urls=existing_urls,
        checkpoint_data=checkpoint_data,
        workers=args.workers,
        session=session
    )
    total_new += new_ms

    # 2. Seção Opcional: Notícias para os Estados
    if args.include_estados and not INTERRUPTED:
        new_est = scrape_section(
            section_name="noticias_estados",
            base_url=ESTADOS_URL,
            output_path=args.output,
            format_mode=args.format,
            checkpoint_file=args.checkpoint,
            max_pages=args.max_pages,
            delay=args.delay,
            resume=(not args.no_resume),
            existing_urls=existing_urls,
            checkpoint_data=checkpoint_data,
            workers=args.workers,
            session=session
        )
        total_new += new_est

    print("\n" + "=" * 75)
    print("RELATÓRIO FINAL DE EXTRAÇÃO:")
    print(f"  Novas matérias salvas nesta execução: {total_new}")
    print(f"  Total acumulado no arquivo          : {len(existing_urls)}")
    print(f"  Destino                             : {args.output}")
    print("=" * 75)


if __name__ == '__main__':
    main()

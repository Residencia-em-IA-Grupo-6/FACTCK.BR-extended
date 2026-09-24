#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_ebc_saude.py

Scraper oficial de notícias de saúde da Agência Brasil (EBC - Empresa Brasil de Comunicação).
Portal: https://agenciabrasil.ebc.com.br/saude

Recursos:
- Paginação sequencial robusta do Drupal (page=0, 1, 2, ...).
- Concorrência paralela via ThreadPoolExecutor (--workers, -w) acelerando a coleta em mais de 10x a 20x.
- Reutilização de conexões HTTP Keep-Alive com pooling sob medida para alta performance.
- Acervo histórico aprofundado com anos de matérias jornalísticas públicas de saúde (>12.000 matérias).
- Extração de metadados ricos: título, linha fina, local, autor, data/hora e corpo íntegro.
- Salvamento incremental thread-safe: grava registros em lote atômico no TSV com flush de disco.
- Checkpoint e retomada (resume) transparente via JSON para evitar requisições repetidas.
- Tolerância a falhas com retries e backoff exponencial.
- Controle gracioso de interrupção via Ctrl+C (SIGINT).
- Filtro opcional por data mínima (--min-date YYYY-MM-DD).
- Formatos de saída: 'raw' (metadados completos) e 'factckbr' (compatível com a base consolidada).
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


BASE_URL = 'https://agenciabrasil.ebc.com.br/saude'

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
    'Titulo',
    'Subtitulo',
    'Local',
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
    print('\n[AVISO] Interrupção detectada (Ctrl+C). Finalizando tarefas ativas e salvando o checkpoint...')


signal.signal(signal.SIGINT, signal_handler)


def sanitize_text(text: str) -> str:
    """Remove quebras de linha e tabulações para preservar a integridade do formato TSV/CSV."""
    if not text:
        return ''
    cleaned = re.sub(r'[\r\n\t]+', ' ', str(text))
    return cleaned.strip()


def parse_ebc_date(raw_date: str):
    """
    Converte strings de datas do formato da EBC ('Publicado em 23/09/2026 - 18:49')
    para data ISO (YYYY-MM-DD) e datetime ISO (YYYY-MM-DDTHH:MM:00).
    """
    if not raw_date:
        return '', ''
    m = re.search(r'(\d{2})/(\d{2})/(\d{4})(?:\s*-\s*(\d{2}):(\d{2}))?', raw_date)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        iso_date = f"{year}-{month}-{day}"
        hour = m.group(4) or '00'
        minute = m.group(5) or '00'
        iso_datetime = f"{iso_date}T{hour}:{minute}:00"
        return iso_date, iso_datetime
    return '', ''


def create_session(workers: int = 8) -> requests.Session:
    """Cria uma sessão HTTP reutilizável com pool de conexões Keep-Alive para alta performance concorrente."""
    session = requests.Session()
    pool_size = max(16, workers * 2)
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


def extract_page_links(html_content: str, base_url: str):
    """
    Extrai todos os links únicos de matérias de saúde da página de listagem da EBC.
    Preserva a ordem de publicação (mais recentes primeiro).
    """
    seen = set()
    links = []
    # Links da Agência Brasil seguem o padrão /saude/noticia/YYYY-MM/slug
    pattern = re.compile(r'href=[\"\'](/saude/noticia/\d{4}-\d{2}/[^\"]+)[\"\']', re.IGNORECASE)
    for match in pattern.finditer(html_content):
        rel_url = match.group(1).strip()
        full_url = urljoin(base_url, rel_url)
        if full_url not in seen:
            seen.add(full_url)
            links.append(full_url)
    return links


def extract_article_page(html_content: str, url: str):
    """
    Extrai o conteúdo completo e os metadados de uma matéria da Agência Brasil.
    """
    # 1. Título
    m_t = re.search(r'<h1[^>]*class=[\"\']titulo-materia[\"\'][^>]*>([\s\S]*?)</h1>', html_content, re.IGNORECASE)
    if not m_t:
        m_t = re.search(r'<h1[^>]*>([\s\S]*?)</h1>', html_content, re.IGNORECASE)
    title = html.unescape(re.sub(r'<[^>]+>', ' ', m_t.group(1))).strip() if m_t else ''

    # 2. Linha Fina / Subtítulo
    m_sub = re.search(r'<div[^>]*class=[\"\']linha-fina-noticia[\"\'][^>]*>([\s\S]*?)</div>', html_content, re.IGNORECASE)
    if not m_sub:
        m_sub = re.search(r'<meta property=\"og:description\" content=\"([^\"]+)\"', html_content, re.IGNORECASE)
    subtitle = html.unescape(re.sub(r'<[^>]+>', ' ', m_sub.group(1))).strip() if m_sub else ''

    # 3. Data de Publicação
    m_d = re.search(r'<div[^>]*class=[\"\']data[\"\'][^>]*>([\s\S]*?)</div>', html_content, re.IGNORECASE)
    date_text = html.unescape(re.sub(r'<[^>]+>', ' ', m_d.group(1))).strip() if m_d else ''
    iso_date, iso_datetime = parse_ebc_date(date_text)

    # 4. Autor
    m_aut = re.search(r'<div[^>]*class=[\"\']autor-noticia[\"\'][^>]*>([\s\S]*?)</div>', html_content, re.IGNORECASE)
    author = html.unescape(re.sub(r'<[^>]+>', ' ', m_aut.group(1))).strip() if m_aut else 'Agência Brasil'
    if not author:
        author = 'Agência Brasil'

    # 5. Local
    m_loc = re.search(r'<div[^>]*class=[\"\']local[\"\'][^>]*>([\s\S]*?)</div>', html_content, re.IGNORECASE)
    local = html.unescape(re.sub(r'<[^>]+>', ' ', m_loc.group(1))).strip() if m_loc else ''

    # 6. Corpo do Artigo
    # Restringe a busca ao nó do artigo para não pegar texto de rodapés ou notícias relacionadas
    m_node = re.search(r'<div[^>]*class=[\"\']node node-conteudo[^\"\']*[\"\'][^>]*>([\s\S]*?)</div>\s*<!--\s*node', html_content, re.IGNORECASE)
    body_scope = m_node.group(1) if m_node else html_content

    paras = re.findall(r'<p[^>]*>([\s\S]*?)</p>', body_scope, re.IGNORECASE)
    clean_paras = []
    for p in paras:
        cp = html.unescape(re.sub(r'<[^>]+>', ' ', p)).strip()
        cp = re.sub(r'\s+', ' ', cp)
        # Filtra botões de acessibilidade, crédito de imagem e linha editorial residual
        if len(cp) > 25 and not cp.startswith('A+') and not cp.startswith('Foto:') and not cp.startswith('Edição:'):
            clean_paras.append(cp)

    body_text = ' '.join(clean_paras)

    return {
        'URL': url,
        'Data': sanitize_text(iso_date),
        'Data_Hora': sanitize_text(iso_datetime),
        'Titulo': sanitize_text(title),
        'Subtitulo': sanitize_text(subtitle),
        'Local': sanitize_text(local),
        'Conteudo': sanitize_text(body_text),
        'Author': sanitize_text(author),
        'is_fake': False
    }


def fetch_and_parse_article(url: str, session: requests.Session, output_format: str = 'raw'):
    """
    Worker paralelo para requisição HTTP e extração estruturada de uma matéria.
    """
    if INTERRUPTED:
        return None

    art_resp = fetch_url(url, session=session)
    if not art_resp or art_resp.status_code != 200:
        return None

    art_data = extract_article_page(art_resp.text, url)

    if output_format == 'factckbr':
        claim_text = art_data['Subtitulo'] if art_data['Subtitulo'] else art_data['Titulo']
        record = {
            'URL': art_data['URL'],
            'Data': art_data['Data'],
            'Titulo': art_data['Titulo'],
            'Author': 'Agência Brasil',
            'Claim': claim_text,
            'reviewBody': art_data['Conteudo'],
            'is_fake': False
        }
    else:
        record = art_data

    return record


def load_existing_urls(file_path: str) -> set:
    """Carrega as URLs já salvas no arquivo para garantir idempotência."""
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
        print(f"[Aviso] Erro ao ler URLs existentes de {file_path}: {e}")
    return urls


def load_checkpoint(checkpoint_file: str) -> dict:
    """Carrega estado anterior do checkpoint JSON."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_checkpoint(checkpoint_file: str, data: dict):
    """Salva estado atual do progresso no arquivo de checkpoint de forma segura."""
    with FILE_LOCK:
        try:
            Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
            data['timestamp'] = datetime.now().isoformat()
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Aviso] Falha ao salvar checkpoint em {checkpoint_file}: {e}")


def append_rows(file_path: str, records: list, columns: list):
    """Grava uma lista de registros atomicamente no arquivo TSV com flush."""
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
    """Grava um registro individualmente (wrapper de conveniência para append_rows)."""
    append_rows(file_path, [record], columns)


def main():
    parser = argparse.ArgumentParser(
        description="Scraper oficial de notícias de saúde da Agência Brasil (EBC) com paralelismo de alta velocidade"
    )
    repo_root = Path(__file__).resolve().parent.parent
    default_output = repo_root / "datasets" / "ebc_saude.tsv"
    default_checkpoint = repo_root / "datasets" / ".checkpoint_ebc_saude.json"

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(default_output),
        help=f"Caminho do arquivo TSV de saída (Padrão: {default_output})"
    )
    parser.add_argument(
        "--format",
        choices=["raw", "factckbr"],
        default="raw",
        help="Formato dos dados: 'raw' (completo) ou 'factckbr' (compatível com a base consolidada)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Quantidade de threads concorrentes para raspagem paralela das matérias (Padrão: 8. Use 1 para sequencial)"
    )
    parser.add_argument(
        "--max-pages", "-p",
        type=int,
        default=10,
        help="Quantidade máxima de páginas a raspar (0 = raspar todas as páginas disponíveis, ~900+ páginas). Padrão: 10 (~140 matérias)"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="Página inicial para raspagem (Padrão: 0)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.1,
        help="Intervalo em segundos entre páginas para rate limiting educado (Padrão: 0.1s)"
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default=None,
        help="Data mínima no formato YYYY-MM-DD. O script encerra se encontrar matérias anteriores a essa data."
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
    print("SCRAPER DE NOTÍCIAS DE SAÚDE — AGÊNCIA BRASIL (EBC) [PARALELO]")
    print(f"  Modo de Formato    : {args.format}")
    print(f"  Arquivo de Saída   : {args.output}")
    print(f"  Workers Paralelos  : {args.workers}")
    print(f"  Delay entre Páginas: {args.delay}s")
    print(f"  Página Inicial     : {args.start_page}")
    print(f"  Limite de Páginas  : {'Todas disponíveis (~900+)' if args.max_pages == 0 else args.max_pages}")
    print(f"  Data Mínima        : {args.min_date or 'Sem limite'}")
    print(f"  Checkpoint         : {args.checkpoint}")
    print("=" * 75)

    existing_urls = set()
    if not args.no_resume:
        existing_urls = load_existing_urls(args.output)
        if existing_urls:
            print(f"[Retomada] {len(existing_urls)} matéria(s) já presentes no arquivo de saída.")

    checkpoint_data = {}
    if not args.no_resume:
        checkpoint_data = load_checkpoint(args.checkpoint)

    start_page = args.start_page
    if not args.no_resume and checkpoint_data.get('last_page') is not None:
        start_page = checkpoint_data['last_page'] + 1
        print(f"[Retomada] Continuando a partir da página {start_page}...")

    columns = FACTCKBR_COLUMNS if args.format == 'factckbr' else RAW_COLUMNS
    current_page = start_page
    pages_processed = 0
    total_saved = 0

    session = create_session(workers=args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while True:
            if INTERRUPTED:
                print("\n[Parada solicitada] Encerrando ciclo de coleta...")
                break

            if args.max_pages > 0 and pages_processed >= args.max_pages:
                print(f"\n[Limite de Páginas] Total de {args.max_pages} página(s) processada(s). Finalizando.")
                break

            page_start_time = time.time()
            page_url = f"{BASE_URL}?page={current_page}" if current_page > 0 else BASE_URL
            print(f"\n-> Página {current_page} (Processadas: {pages_processed}/{args.max_pages if args.max_pages > 0 else '∞'}): {page_url}")

            resp = fetch_url(page_url, session=session)
            if not resp or resp.status_code != 200:
                print(f"   [Falha] Código HTTP {getattr(resp, 'status_code', 'Sem resposta')} ao obter listagem. Encerrando.")
                break

            links = extract_page_links(resp.text, BASE_URL)
            if not links:
                print("   [Fim de Catálogo] Nenhuma matéria encontrada na página. Encerrando raspagem.")
                break

            new_links = [u for u in links if u not in existing_urls]
            already_count = len(links) - len(new_links)
            print(f"   Identificadas {len(links)} matéria(s) na página ({len(new_links)} novas, {already_count} já salvas).")

            if not new_links:
                print(f"   Todas as matérias da página {current_page} já haviam sido processadas.")
                checkpoint_data['last_page'] = current_page
                checkpoint_data['total_saved'] = len(existing_urls)
                save_checkpoint(args.checkpoint, checkpoint_data)
                pages_processed += 1
                current_page += 1
                continue

            # Raspagem paralela das matérias da página atual
            futures = {
                executor.submit(fetch_and_parse_article, art_url, session, args.format): (idx, art_url)
                for idx, art_url in enumerate(new_links, start=1)
            }

            results = {}
            for future in as_completed(futures):
                if INTERRUPTED:
                    break
                idx, art_url = futures[future]
                try:
                    record = future.result()
                    if record:
                        results[idx] = record
                    else:
                        print(f"   [{idx}/{len(new_links)}] Falha ao processar matéria: {art_url}")
                except Exception as e:
                    print(f"   [{idx}/{len(new_links)}] Erro inesperado em {art_url}: {e}")

            # Reordena mantendo a ordem original da página
            sorted_records = [results[i] for i in sorted(results.keys())]

            valid_records = []
            reached_min_date = False
            for rec in sorted_records:
                if args.min_date and rec.get('Data') and rec['Data'] < args.min_date:
                    print(f"\n[Filtro de Data] Data da matéria ({rec['Data']}) atingiu o limite mínimo ({args.min_date}).")
                    reached_min_date = True
                    break
                valid_records.append(rec)
                print(f"   + [{rec['Data']}] {rec['Titulo'][:65]}...")

            if valid_records:
                append_rows(args.output, valid_records, columns)
                for r in valid_records:
                    existing_urls.add(r['URL'])
                total_saved += len(valid_records)

            dur = time.time() - page_start_time
            print(f"   ✓ Página {current_page} processada em {dur:.2f}s (+{len(valid_records)} novas salvas, total acumulado: {len(existing_urls)}).")

            # Atualiza checkpoint ao fim da página
            checkpoint_data['last_page'] = current_page
            checkpoint_data['total_saved'] = len(existing_urls)
            save_checkpoint(args.checkpoint, checkpoint_data)

            pages_processed += 1
            current_page += 1

            if reached_min_date:
                break

            if args.delay > 0:
                time.sleep(args.delay)

    print("\n" + "=" * 75)
    print("RELATÓRIO FINAL DE EXTRAÇÃO — EBC (AGÊNCIA BRASIL):")
    print(f"  Novas matérias salvas nesta execução: {total_saved}")
    print(f"  Total acumulado no arquivo          : {len(existing_urls)}")
    print(f"  Destino                             : {args.output}")
    print("=" * 75)


if __name__ == '__main__':
    main()

import argparse
import csv
import json
import os
import re
import signal
import sys
import time
from datetime import datetime

from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import requests

# Scraper de Notícias Falsas de Saúde do Boatos.org
# Suporta retomada automática (resume), salvamento incremental por matéria e paginação completa (~53 páginas).
# URL Base: https://www.boatos.org/saude

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

BASE_URL = 'https://www.boatos.org/saude'
FEED_URL = 'https://www.boatos.org/saude/feed'

TSV_COLUMNS = [
    'URL',
    'Data',
    'Titulo',
    'Texto_Falso_Original',
    'Resumo_Boato',
    'Desmentido',
    'Tags'
]

FACTCKBR_COLUMNS = [
    'URL',
    'Author',
    'datePublished',
    'claimReviewed',
    'reviewBody',
    'title',
    'ratingValue',
    'bestRating',
    'alternativeName',
    'fakeTextOriginal'
]

# Controle de interrupção graciosa (Ctrl+C)
INTERRUPTED = False


def signal_handler(sig, frame):
    global INTERRUPTED
    INTERRUPTED = True
    print("\n\n[AVISO] Interrupção detectada (Ctrl+C). Finalizando a matéria em andamento e salvando o estado...")


signal.signal(signal.SIGINT, signal_handler)


def sanitize_text(text):
    """Remove quebras de linha e tabulações para preservar a estrutura TSV/CSV."""
    if not text:
        return ''
    cleaned = re.sub(r'[\r\n\t]+', ' ', str(text))
    return cleaned.strip()


def fetch_url(url, max_retries=3, backoff_factor=1.5):
    """Executa requisição HTTP com tentativas e backoff exponencial em caso de falha transitória."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:  # Rate limit
                wait_time = (backoff_factor ** attempt) * 2
                print(f"    [Rate limit 429] Aguardando {wait_time:.1f}s antes de tentar novamente...")
                time.sleep(wait_time)
            elif resp.status_code >= 500:
                time.sleep(attempt * 2)
            else:
                return resp
        except (requests.RequestException, Exception) as e:
            if attempt == max_retries:
                print(f"    [Erro de conexão] Falha definitiva ao acessar {url}: {e}")
                return None
            time.sleep(attempt * 2)
    return None


def detect_max_pages():
    """Detecta dinamicamente a quantidade total de páginas da categoria de saúde."""
    try:
        resp = fetch_url(BASE_URL)
        if resp and resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            max_p = 1
            for a in soup.find_all('a'):
                href = a.get('href', '')
                m = re.search(r'paged=(\d+)', href)
                if m:
                    max_p = max(max_p, int(m.group(1)))
            return max_p
    except Exception:
        pass
    return 53  # Valor padrão de fallback baseado no catálogo atual


def load_existing_urls(file_path):
    """Carrega conjunto de URLs já salvas para evitar duplicações e requisições repetidas."""
    if not os.path.exists(file_path):
        return set()
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True, dtype=False)
        else:
            df = pd.read_csv(file_path, sep='\t', dtype=str, keep_default_na=False)

        if 'URL' in df.columns:
            return set(df['URL'].dropna().str.strip().tolist())
    except Exception as e:
        print(f"Aviso ao ler URLs existentes de {file_path}: {e}")
    return set()


def load_checkpoint(checkpoint_file):
    """Carrega dados do arquivo de checkpoint."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_checkpoint(checkpoint_file, last_page, total_saved):
    """Salva estado atual no arquivo de checkpoint."""
    try:
        data = {
            'last_page': last_page,
            'total_saved': total_saved,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Aviso ao salvar checkpoint: {e}")


def append_article_to_file(file_path, data_row):
    """Salva imediatamente a matéria no disco (append incremental)."""
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0

    if file_path.endswith('.jsonl'):
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_row, ensure_ascii=False) + '\n')
            f.flush()
    elif file_path.endswith('.csv'):
        with open(file_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_row)
            f.flush()
    else:  # TSV padrão
        with open(file_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter='\t')
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_row)
            f.flush()


def append_factckbr_to_file(file_path, data_row):
    """Salva imediatamente uma linha compatível com o FACTCK.BR."""
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    factck_row = {
        'URL': data_row.get('URL', ''),
        'Author': 'https://www.boatos.org',
        'datePublished': data_row.get('Data', ''),
        'claimReviewed': data_row.get('Resumo_Boato') or data_row.get('Titulo', ''),
        'reviewBody': data_row.get('Desmentido', ''),
        'title': data_row.get('Titulo', ''),
        'ratingValue': '1',
        'bestRating': '5',
        'alternativeName': 'falso',
        'fakeTextOriginal': data_row.get('Texto_Falso_Original', '')
    }
    with open(file_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FACTCKBR_COLUMNS, delimiter='\t')
        if not file_exists:
            writer.writeheader()
        writer.writerow(factck_row)
        f.flush()


def scrape_article(url):
    """Extrai os dados de uma checagem do Boatos.org, capturando o texto falso literal."""
    resp = fetch_url(url)
    if not resp or resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.content, 'html.parser')

    # 1. Título
    h1 = soup.find('h1')
    title = h1.get_text(strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else '')
    title = sanitize_text(title)

    # 2. Data de publicação
    time_tag = soup.find('time')
    date = time_tag.get('datetime', '')[:10] if time_tag else ''

    # 3. Tags / Categorias
    tags = [t.get_text(strip=True) for t in soup.find_all('a', rel='tag')]
    tags_str = ', '.join(tags)

    # 4. TEXTO ORIGINAL DA NOTÍCIA FALSA (dentro de <blockquote>)
    bqs = soup.find_all('blockquote')
    fake_texts = [sanitize_text(b.get_text(separator=' ', strip=True)) for b in bqs]
    fake_text = ' '.join([t for t in fake_texts if t])

    # 5. Resumo e texto explicativo do desmentido
    content_div = soup.find('div', class_='entry-content') or soup.find('article')
    summary = ''
    debunk_paragraphs = []

    if content_div:
        for p in content_div.find_all('p'):
            text = sanitize_text(p.get_text(separator=' ', strip=True))
            if not text:
                continue
            if text.lower().startswith('boato –') or text.lower().startswith('boato -'):
                summary = text
            else:
                debunk_paragraphs.append(text)

    debunk_text = ' '.join(debunk_paragraphs)

    # Fallback caso não haja blockquote
    if not fake_text:
        fake_text = summary

    return {
        'URL': url,
        'Data': date,
        'Titulo': title,
        'Texto_Falso_Original': fake_text,
        'Resumo_Boato': summary,
        'Desmentido': debunk_text,
        'Tags': tags_str
    }


def get_urls_from_page(page_num):
    """Obtém os links das matérias de uma página da seção de saúde."""
    # Método 1: via feed RSS
    feed_url = f"{FEED_URL}/?paged={page_num}"
    try:
        feed = feedparser.parse(feed_url)
        if feed.entries:
            return [e.link for e in feed.entries if hasattr(e, 'link') and e.link]
    except Exception:
        pass

    # Método 2: via HTML direto da página
    page_url = f"{BASE_URL}?paged={page_num}" if page_num > 1 else BASE_URL
    try:
        resp = fetch_url(page_url)
        if resp and resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            links = []
            for art in soup.find_all('article'):
                a = art.find('a')
                if a and a.get('href') and a.get('href').startswith('http'):
                    links.append(a.get('href'))
            return list(dict.fromkeys(links))
    except Exception as e:
        print(f"Erro ao acessar {page_url}: {e}")

    return []


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_output = os.path.join(repo_root, 'datasets', 'boatos_saude.tsv')
    default_checkpoint = os.path.join(repo_root, '.boatos_saude_checkpoint.json')

    parser = argparse.ArgumentParser(
        description="Scraper resiliente de Notícias Falsas de Saúde do Boatos.org com suporte a retomada (resume)."
    )
    parser.add_argument('--pages', '-p', type=int, default=0,
                        help="Quantidade de páginas a coletar (25 notícias/página; 0 para coletar TODAS as páginas do site).")
    parser.add_argument('--start-page', type=int, default=0,
                        help="Página inicial (0 = retoma automaticamente do último checkpoint salvo).")
    parser.add_argument('--limit', '-l', type=int, default=0,
                        help="Limite máximo de matérias a coletar (0 para sem limite).")
    parser.add_argument('--output', '-o', default=default_output,
                        help="Arquivo de saída (.tsv, .csv ou .jsonl; padrão: datasets/boatos_saude.tsv).")
    parser.add_argument('--delay', type=float, default=0.5,
                        help="Intervalo em segundos entre cada requisição para não sobrecarregar o site (padrão: 0.5s).")
    parser.add_argument('--factckbr-format', action='store_true', default=True,
                        help="Salva também versão no formato padrão do FACTCK.BR (boatos_saude_factckbr.tsv).")
    parser.add_argument('--no-resume', action='store_true',
                        help="Ignora o checkpoint anterior e reinicia a partir da página 1 (mantendo deduplicação).")
    parser.add_argument('--checkpoint-file', default=default_checkpoint,
                        help="Arquivo para armazenamento do estado da raspagem.")

    args = parser.parse_args()

    total_site_pages = detect_max_pages()
    target_pages = args.pages if args.pages > 0 else total_site_pages

    # 1. Gerenciar Checkpoint e Retomada
    checkpoint = {}
    if not args.no_resume:
        checkpoint = load_checkpoint(args.checkpoint_file)

    if args.start_page > 0:
        start_page = args.start_page
    elif checkpoint.get('last_page'):
        start_page = checkpoint['last_page']
        print(f"[RETOMADA] Checkpoint encontrado: retomando a partir da página {start_page}.")
    else:
        start_page = 1

    # 2. Carregar URLs já existentes no arquivo de saída
    seen_urls = load_existing_urls(args.output)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    factckbr_output = os.path.join(output_dir, 'boatos_saude_factckbr.tsv')

    print("=" * 75)
    print("SCRAPER DE NOTÍCIAS FALSAS DE SAÚDE - BOATOS.ORG (COM RETOMADA)")
    print(f"URL de Origem           : {BASE_URL}")
    print(f"Total de páginas no site: ~{total_site_pages} (até {total_site_pages * 25} matérias)")
    print(f"Página inicial          : {start_page} de {target_pages}")
    print(f"Matérias já no disco    : {len(seen_urls)}")
    print(f"Arquivo de saída        : {args.output}")
    if args.factckbr_format:
        print(f"Saída FACTCK.BR         : {factckbr_output}")
    print("=" * 75)

    total_new_saved = 0

    try:
        for page in range(start_page, target_pages + 1):
            if INTERRUPTED:
                break
            if args.limit and total_new_saved >= args.limit:
                break

            print(f"\n>>> [Página {page}/{target_pages}] Buscando lista de matérias...")
            urls = get_urls_from_page(page)

            if not urls:
                print(f"Nenhum link encontrado na página {page}. Finalizando.")
                break

            new_in_page = [u for u in urls if u not in seen_urls]
            print(f"    Links na página: {len(urls)} ({len(new_in_page)} novos, {len(urls) - len(new_in_page)} já salvos)")

            for url in urls:
                if INTERRUPTED:
                    break
                if args.limit and total_new_saved >= args.limit:
                    print(f"\n[Limite atingido] Meta de {args.limit} matérias alcançada.")
                    break

                if url in seen_urls:
                    continue

                seen_urls.add(url)
                print(f"  [{len(seen_urls)}] Extraindo: {url}")
                data = scrape_article(url)

                if data and data.get('Texto_Falso_Original'):
                    # Salva imediatamente no disco (persistência incremental)
                    append_article_to_file(args.output, data)
                    if args.factckbr_format:
                        append_factckbr_to_file(factckbr_output, data)

                    total_new_saved += 1
                    sample = data['Texto_Falso_Original'][:75]
                    print(f"      [SALVO] Boato: \"{sample}...\"")
                else:
                    print(f"      (Ignorado: texto original não identificado)")

                time.sleep(args.delay)

            # Atualiza checkpoint ao concluir cada página
            if not INTERRUPTED:
                save_checkpoint(args.checkpoint_file, last_page=page + 1, total_saved=len(seen_urls))

    except KeyboardInterrupt:
        print("\n\n[AVISO] Interrupção pelo usuário via teclado.")
    finally:
        print("\n" + "=" * 75)
        print("SITUAÇÃO DO SCRAPING:")
        print(f"Novas matérias salvas nesta execução : {total_new_saved}")
        print(f"Total acumulado salvo no disco       : {len(seen_urls)}")
        print(f"Arquivo principal                    : {args.output}")
        if args.factckbr_format:
            print(f"Arquivo padrão FACTCK.BR             : {factckbr_output}")

        if INTERRUPTED:
            print(f"\n[DICA] O processo foi pausado. Para continuar de onde parou, basta rodar:")
            print(f"       python scrape_boatos_saude.py")
        else:
            print("\n[CONCLUÍDO] Raspagem das páginas solicitadas finalizada com sucesso.")
        print("=" * 75)


if __name__ == '__main__':
    main()

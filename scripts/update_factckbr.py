import argparse
import json
import os
import re
import sys
import xml.sax.saxutils as saxutils

from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import requests

# FACTCK.BR - Extrator de Checagens Focado em Saude
# Suporta Google Fact Check Tools API e Extracao Direta via Sitemaps/Feeds

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; FactckBR-Saude/2.0; +https://github.com/jghm-f/FACTCK.BR)'
}

DEFAULT_PUBLISHERS = [
    'aosfatos.org',
    'agencialupa.org',
    'lupa.uol.com.br',
    'apublica.org',
    'estadao.com.br',
    'uol.com.br',
    'g1.globo.com'
]

HEALTH_QUERIES = [
    'vacina',
    'covid',
    'saúde',
    'medicamento',
    'dengue',
    'câncer',
    'remédio',
    'anvisa',
    'sus',
    'hospital',
    'tratamento',
    'doença',
    'vírus',
    'cloroquina',
    'ivermectina',
    'pandemia'
]

HEALTH_KEYWORDS = [
    # Pandemias, virus e infeccoes
    r'covid', r'coronav[ií]rus', r'sars-cov', r'pandemia', r'epidemia', r'surto',
    r'dengue', r'zika', r'chikungunya', r'chicungunha', r'mpox', r'var[ií]ola',
    r'gripe', r'influenza', r'h1n1', r'h5n1', r'v[ií]rus', r'bact[eé]ria',
    r'infec[cç][aã]o', r'cont[aá]gio', r'contamina[cç][aã]o',
    # Vacinas e imunizacao
    r'vacin[a-z]*', r'imuniza[cç][aã]o', r'imunizante', r'coronavac', r'pfizer',
    r'astrazeneca', r'janssen', r'butantan', r'fiocruz', r'gotinha', r'antivacin[a-z]*',
    # Doencas e condicoes clinicas
    r'c[aâ]ncer', r'tumor', r'leucemia', r'diabetes', r'autismo', r'aids', r'hiv',
    r'tuberculose', r'pneumonia', r'infarto', r'avc', r'parada card[ií]aca',
    r'miocardite', r'trombose', r'hipertens[aã]o', r'press[aã]o alta', r'febre amarela',
    r'doen[cç]a[s]?', r's[ií]ndrome', r'sequela[s]?',
    # Medicamentos e tratamentos
    r'rem[eé]dio[s]?', r'medicamento[s]?', r'f[aá]rmaco[s]?', r'farm[aá]cia',
    r'cloroquina', r'hidroxicloroquina', r'ivermectina', r'ozonioterapia',
    r'antibi[oó]tico[s]?', r'insulina', r'di[oó]xido de cloro', r'quimioterapia',
    r'tratamento[s]?', r'tratamento precoce', r'efeito colateral', r'efeitos colaterais',
    r'placebo', r'cura', r'rem[eé]dio caseiro', r'vitamina[s]?',
    # Sistema de saude, instituicoes e profissionais
    r'sa[uú]de', r'\bsus\b', r'anvisa', r'\boms\b', r'hospital', r'hospitais',
    r'm[eé]dic[oa][s]?', r'enfermagem', r'enfermeir[oa][s]?', r'\buti\b', r'leito[s]?',
    r'posto de sa[uú]de', r'cl[ií]nica[s]?', r'minist[eé]rio da sa[uú]de',
    r'secretaria de sa[uú]de', r'morte s[uú]bita', r'[oó]bito[s]?', r'mortalidade'
]

HEALTH_REGEX = re.compile(r'\b(' + '|'.join(HEALTH_KEYWORDS) + r')\b', re.IGNORECASE)

RATING_MAP = {
    'falso': ('1', '5'),
    'falsa': ('1', '5'),
    'mentira': ('1', '5'),
    'fake': ('1', '5'),
    'insustentável': ('2', '5'),
    'distorcido': ('3', '5'),
    'impreciso': ('3', '5'),
    'exagerado': ('3', '5'),
    'sem contexto': ('3', '5'),
    'subestimado': ('4', '5'),
    'contraditório': ('4', '5'),
    'verdadeiro': ('5', '5'),
    'fato': ('5', '5'),
    'verdade': ('5', '5'),
    'não é bem assim': ('5', '6'),
    'de olho': ('3', '6'),
}


def is_health_related(text):
    """Verifica se o texto pertence ao dominio de saude."""
    if not text:
        return False
    return bool(HEALTH_REGEX.search(str(text)))


def sanitize_text(s):
    """Remove quebras de linha e tabulacoes para preservar a estrutura TSV."""
    if not s:
        return ''
    cleaned = re.sub(r'[\r\n\t]+', ' ', str(s))
    return cleaned.strip()


def re_char(s):
    """Filtra caracteres especiais mantendo acentuacao e pontuacao comum."""
    if not s:
        return ''
    return re.sub(r'[^A-Za-z0-9 \!\@\#\$\%\&\*\:\,\.\;\:\-\_\"\'\]\[\}\{\+\á\à\é\è\í\ì\ó\ò\ú\ù\ã\õ\â\ê\ô\ç\|]+', '', str(s))


def map_rating(textual_rating):
    """Mapeia o veredito textual para a escala numerica original."""
    if not textual_rating:
        return ('', '')
    clean = textual_rating.lower().strip()
    for key, val in RATING_MAP.items():
        if key in clean:
            return val
    return ('', '5')


def load_tsv_pandas(file_name):
    """Carrega o dataset preservando tipos string exatos."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = file_name if file_name.endswith('.tsv') else file_name + '.tsv'
    if not os.path.exists(path):
        candidates = [
            path,
            os.path.join(repo_root, 'datasets', os.path.basename(path)),
            os.path.join(repo_root, 'datasets', 'FACTCKBR.tsv'),
            'datasets/FACTCKBR.tsv',
            '../datasets/FACTCKBR.tsv',
            'FACTCKBR.tsv',
            'FACTCKBR_old.tsv',
            'factCkBr.tsv',
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                path = candidate
                break
    if os.path.exists(path):
        return pd.read_csv(path, sep='\t', index_col=0, dtype=str, keep_default_na=False)
    return None


def save_tsv_pandas(data, file_name):
    """Salva dataset em arquivo TSV."""
    path = file_name if file_name.endswith('.tsv') else file_name + '.tsv'
    data.to_csv(path, sep='\t', index=True)


def update_dataset(dataset, new_entries):
    """Mescla o dataset anterior com as novas entradas removendo duplicatas de URL e checagem."""
    temp_df = pd.concat([dataset, new_entries])
    temp_df = temp_df.reset_index().drop_duplicates(subset=['URL', 'claimReviewed']).set_index('URL')
    return temp_df


def extract_claim_reviews(data):
    """Extrai recursivamente objetos ClaimReview do JSON-LD."""
    results = []
    if isinstance(data, dict):
        obj_type = data.get('@type', [])
        if isinstance(obj_type, str):
            obj_type = [obj_type]
        if 'ClaimReview' in obj_type:
            results.append(data)
        if '@graph' in data and isinstance(data['@graph'], list):
            for item in data['@graph']:
                results.extend(extract_claim_reviews(item))
    elif isinstance(data, list):
        for item in data:
            results.extend(extract_claim_reviews(item))
    return results


def get_author(item, url):
    """Extrai com seguranca a URL ou nome da agencia verificadora."""
    pub = item.get('publisher')
    if isinstance(pub, dict):
        if pub.get('url'):
            return pub['url']
        if pub.get('name'):
            return pub['name']

    author = item.get('author')
    if isinstance(author, dict):
        if author.get('url'):
            return author['url']
        if author.get('name'):
            return author['name']
    elif isinstance(author, list) and len(author) > 0 and isinstance(author[0], dict):
        if author[0].get('url'):
            return author[0]['url']
        if author[0].get('name'):
            return author[0]['name']
    elif isinstance(author, str) and author:
        return author

    if 'aosfatos.org' in url:
        return 'https://www.aosfatos.org'
    elif 'apublica.org' in url:
        return 'https://apublica.org'
    elif 'lupa' in url:
        return 'https://piaui.folha.uol.com.br/lupa'
    return 'Unknown'


def parse_claim_review_from_url(url, must_be_health=True):
    """Busca o artigo e extrai registros ClaimReview validando o filtro de saude."""
    claims = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return claims
        soup = BeautifulSoup(r.content, 'html.parser')
        page_title = soup.title.get_text(strip=True) if soup.title else ''
        page_title = sanitize_text(re_char(page_title))

        for script in soup.find_all('script', attrs={'type': 'application/ld+json'}):
            script_text = script.get_text(strip=True)
            if not script_text:
                continue
            try:
                data = json.loads(script_text)
            except Exception:
                try:
                    data = json.loads(saxutils.unescape(script_text))
                except Exception:
                    continue

            reviews = extract_claim_reviews(data)
            for rev in reviews:
                author = get_author(rev, url)
                date_published = sanitize_text(rev.get('datePublished', ''))
                claim_reviewed = sanitize_text(re_char(rev.get('claimReviewed', '')))
                review_body = rev.get('reviewBody') or rev.get('description') or 'Empty'
                review_body = sanitize_text(re_char(review_body))

                review_rating = rev.get('reviewRating', {})
                if not isinstance(review_rating, dict):
                    review_rating = {}

                rating_value = review_rating.get('ratingValue', '')
                if isinstance(rating_value, float) and rating_value.is_integer():
                    rating_value = int(rating_value)
                best_rating = review_rating.get('bestRating', '')
                if isinstance(best_rating, float) and best_rating.is_integer():
                    best_rating = int(best_rating)
                alt_name = sanitize_text(review_rating.get('alternateName', ''))

                # Validacao de saude
                combined_text = f"{url} {claim_reviewed} {page_title} {review_body}"
                if must_be_health and not is_health_related(combined_text):
                    continue

                row = [
                    url,
                    author,
                    date_published,
                    claim_reviewed,
                    review_body,
                    page_title,
                    str(rating_value) if rating_value != '' else '',
                    str(best_rating) if best_rating != '' else '',
                    alt_name
                ]
                claims.append(row)
    except Exception:
        pass
    return claims


def extract_from_sitemaps(existing_urls, limit=50, health_only=True):
    """Extrai checagens de saude diretamente dos sitemaps oficiais das agencias."""
    print("\n--- Modo Direto: Coletando noticias de saude via Sitemap (Aos Fatos) ---")
    sitemap_url = 'https://www.aosfatos.org/sitemap-noticias.xml'
    try:
        r = requests.get(sitemap_url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            print(f"Erro ao acessar sitemap: {r.status_code}")
            return []
        soup = BeautifulSoup(r.content, 'xml')
        all_urls = [l.text.strip() for l in soup.find_all('loc')]
        print(f"Total de materias encontradas no sitemap: {len(all_urls)}")
    except Exception as e:
        print(f"Falha ao carregar sitemap: {e}")
        return []

    # Filtrar candidatos pelo slug/URL primeiro
    candidates = []
    for u in all_urls:
        if u in existing_urls:
            continue
        if not health_only or is_health_related(u):
            candidates.append(u)

    print(f"Materias candidatas de saude (novas): {len(candidates)}")
    extracted = []
    count = 0

    for u in candidates:
        if limit and len(extracted) >= limit:
            break
        count += 1
        print(f"[{count}/{len(candidates)}] Verificando: {u}")
        claims = parse_claim_review_from_url(u, must_be_health=health_only)
        for c in claims:
            extracted.append(c)
            print(f"  -> [SAUDE] Checagem: {c[3][:70]} | Veredito: {c[8]}")
            if limit and len(extracted) >= limit:
                break

    return extracted


def extract_from_google_api(api_key, existing_urls, limit=50, max_pages=5, health_only=True):
    """Extrai checagens de saude utilizando a Google Fact Check Tools API."""
    print("\n--- Modo API: Coletando noticias de saude via Google Fact Check Tools API ---")
    endpoint = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'
    extracted = []

    for query in HEALTH_QUERIES:
        if limit and len(extracted) >= limit:
            break
        print(f"\nConsultando termo de saude: '{query}'...")
        page_token = None

        for page in range(max_pages):
            if limit and len(extracted) >= limit:
                break
            params = {
                'key': api_key,
                'query': query,
                'languageCode': 'pt-BR',
                'pageSize': 50
            }
            if page_token:
                params['pageToken'] = page_token

            try:
                resp = requests.get(endpoint, params=params, headers=HEADERS, timeout=15)
                if resp.status_code == 403:
                    print("Erro 403: Chave de API invalida ou sem permissao no Google Cloud.")
                    return extracted
                elif resp.status_code != 200:
                    print(f"Erro {resp.status_code} na API: {resp.text}")
                    break

                data = resp.json()
                claims = data.get('claims', [])
                if not claims:
                    break

                for claim in claims:
                    claim_text = sanitize_text(re_char(claim.get('text', '')))
                    for cr in claim.get('claimReview', []):
                        url = cr.get('url', '').strip()
                        if not url or url in existing_urls:
                            continue

                        title = sanitize_text(re_char(cr.get('title', '')))
                        textual_rating = sanitize_text(cr.get('textualRating', ''))

                        # Filtro de saude
                        if health_only and not is_health_related(f"{url} {claim_text} {title}"):
                            continue

                        pub = cr.get('publisher', {})
                        site = pub.get('site') or pub.get('name') or 'Unknown'
                        author = f"https://{site}" if not site.startswith('http') else site
                        date_pub = (cr.get('reviewDate') or claim.get('claimDate') or '')[:10]
                        rating_val, best_rating = map_rating(textual_rating)
                        review_body = title or claim_text or 'Empty'

                        row = [
                            url,
                            author,
                            date_pub,
                            claim_text,
                            review_body,
                            title,
                            rating_val,
                            best_rating,
                            textual_rating
                        ]
                        extracted.append(row)
                        existing_urls.add(url)
                        print(f"  -> [SAUDE] {claim_text[:65]} | {textual_rating}")

                        if limit and len(extracted) >= limit:
                            break

                page_token = data.get('nextPageToken')
                if not page_token:
                    break

            except Exception as e:
                print(f"Erro na requisicao: {e}")
                break

    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extrai apenas checagens de fatos relacionadas a SAUDE para o FACTCK.BR.")
    parser.add_argument('--source', choices=['auto', 'google-api', 'sitemap'], default='auto',
                        help="Fonte: 'auto' (usa API se houver chave, sitemap caso contrario), 'google-api' ou 'sitemap'.")
    parser.add_argument('--key', '-k', default=os.getenv('GOOGLE_API_KEY') or os.getenv('FACTCHECK_API_KEY'),
                        help="Chave de API do Google Cloud (para Google Fact Check Tools API).")
    parser.add_argument('--limit', type=int, default=50,
                        help="Quantidade maxima de novas checagens de saude a extrair (padrao: 50; use 0 para sem limite).")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dataset = os.path.join(repo_root, 'datasets', 'FACTCKBR.tsv')

    parser.add_argument('--dataset', default=default_dataset,
                        help="Caminho do arquivo TSV do dataset (padrao: datasets/FACTCKBR.tsv).")
    parser.add_argument('--export-health-only', action='store_true', default=True,
                        help="Gera tambem um arquivo exclusivo com apenas as checagens de saude (FACTCKBR_saude.tsv).")

    args = parser.parse_args()

    toprow = ['URL', 'Author', 'datePublished', 'claimReviewed', 'reviewBody', 'title', 'ratingValue', 'bestRating', 'alternativeName']

    print("=" * 70)
    print("FACTCK.BR - EXTRATOR DE CHECAGENS DE SAUDE")
    print("=" * 70)

    # 1. Carregar dataset existente
    dataset_file = args.dataset
    dataset = load_tsv_pandas(dataset_file)
    if dataset is not None:
        old_count = len(dataset)
        existing_urls = set(dataset.index.dropna().tolist())
        print(f"Dataset carregado: {old_count} checagens existentes encontradas.")
    else:
        dataset = pd.DataFrame(columns=toprow).set_index('URL')
        old_count = 0
        existing_urls = set()
        print("Dataset anterior nao encontrado. Iniciando nova base.")

    # 2. Determinar fonte de extracao
    use_google_api = False
    if args.source == 'google-api':
        if not args.key:
            print("Erro: Para usar --source google-api e obrigatorio fornecer --key ou a variavel GOOGLE_API_KEY.")
            sys.exit(1)
        use_google_api = True
    elif args.source == 'auto':
        if args.key:
            print("Chave de API detectada. Utilizando Google Fact Check Tools API.")
            use_google_api = True
        else:
            print("Nenhuma chave de API do Google informada. Utilizando modo direto via Sitemap oficial (Aos Fatos).")
            use_google_api = False
    else:
        use_google_api = False

    # 3. Executar extracao
    if use_google_api:
        new_claims = extract_from_google_api(
            api_key=args.key,
            existing_urls=existing_urls,
            limit=args.limit,
            health_only=True
        )
    else:
        new_claims = extract_from_sitemaps(
            existing_urls=existing_urls,
            limit=args.limit,
            health_only=True
        )

    print(f"\nTotal de novas checagens de saude extraidas: {len(new_claims)}")

    if not new_claims:
        print("Nenhuma nova checagem de saude foi adicionada.")
        return

    # 4. Criar DataFrame com as novas entradas
    new_entries = pd.DataFrame(new_claims, columns=toprow)
    new_entries = new_entries.set_index('URL')

    # 5. Atualizar o dataset principal
    updated_dataset = update_dataset(dataset, new_entries)
    new_count = len(updated_dataset)
    added_count = new_count - old_count

    save_tsv_pandas(updated_dataset, dataset_file)
    print(f"\n[OK] Dataset principal atualizado: {dataset_file}")
    print(f"     Anteriores: {old_count} | Adicionados: {added_count} | Total: {new_count}")

    # 6. Exportar subconjunto exclusivo de saude se solicitado
    if args.export_health_only:
        def filter_health_rows(df):
            mask = df.apply(lambda r: is_health_related(f"{r.name} {r.get('claimReviewed', '')} {r.get('title', '')} {r.get('reviewBody', '')}"), axis=1)
            return df[mask]

        health_subset = filter_health_rows(updated_dataset)
        health_file = os.path.join(os.path.dirname(dataset_file), 'FACTCKBR_saude.tsv')
        save_tsv_pandas(health_subset, health_file)
        print(f"[OK] Base exclusiva de saude gerada: {health_file} ({len(health_subset)} checagens totais de saude)")

    print("=" * 70)


if __name__ == '__main__':
    main()

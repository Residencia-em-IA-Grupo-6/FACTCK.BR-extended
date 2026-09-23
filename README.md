# FACTCK.BR: Plataforma de Datasets e Ferramentas para Detecção de Desinformação

O **FACTCK.BR** é um projeto e ecossistema de dados voltado ao estudo, pesquisa e desenvolvimento de modelos de Inteligência Artificial e Processamento de Linguagem Natural (NLP) para **detecção de desinformação, notícias falsas (fake news) e notícias verdadeiras**, com ênfase primordial em língua portuguesa e expansões para o inglês.

Originalmente concebido como uma base de dados de checagens baseadas no esquema [ClaimReview](https://schema.org/ClaimReview) (publicado no simpósio WebMedia '19), o repositório foi modernizado para incluir:
1. **Pipelines de atualização contínua** com a API oficial do Google Fact Check Tools e sitemaps de agências brasileiras (Aos Fatos, Agência Lupa, Boatos.org, etc.).
2. **Filtros temáticos especializados** com foco em desinformação sobre **saúde** (vacinas, tratamentos, epidemias e saúde pública).
3. **Scrapers especializados** capazes de extrair o texto original completo de boatos e suas respectivas refutações.
4. **Extração de checagens de redes sociais** via **Twitter Community Notes** (Birdwatch) e oEmbed gratuito (sem custo de API), fornecendo amostras de postagens com textos reais de tweets (`claim`), avaliações comunitárias (`review`) e **classificação binária de veracidade** (`is_fake`).

---

## 📁 Estrutura do Repositório

```
FACTCK.BR/
├── datasets/                      # Bases de dados em formato TSV (UTF-8)
│   ├── FACTCKBR.tsv               # Dataset principal atualizado (multi-agências)
│   ├── FACTCKBR_saude.tsv         # Subconjunto filtrado para desinformação em saúde
│   ├── FACTCKBR_old.tsv           # Dataset histórico original (WebMedia 2019)
│   ├── boatos_saude.tsv           # Notícias de saúde do Boatos.org (texto do boato + checagem)
│   ├── boatos_saude_factckbr.tsv  # Boatos.org formatado no schema padrão FACTCK.BR
│   ├── community_notes_pt.tsv     # Checagens do Twitter Community Notes em Português
│   └── community_notes_en.tsv     # Checagens do Twitter Community Notes em Inglês
├── scripts/                       # Scripts e pipelines executáveis
│   ├── update_factckbr.py         # Atualizador multi-fonte com suporte a Google API e Sitemaps
│   ├── scrape_boatos_saude.py     # Scraper com suporte a retomada (resume) para o Boatos.org
│   └── scrape_community_notes.py  # Pipeline do Twitter Community Notes via oEmbed sem custos
├── requirements.txt               # Dependências do Python
├── LICENSE                        # Licença MIT
└── README.md                      # Documentação geral do repositório
```

---

## 📊 Dicionário dos Datasets

### 1. `datasets/FACTCKBR.tsv` e `datasets/FACTCKBR_saude.tsv`
Bases tabulares contendo checagens estruturadas de agências de checagem profissionais (Aos Fatos, Lupa, Boatos.org, etc.):

| Coluna | Descrição |
| :--- | :--- |
| `URL` | Endereço web do artigo de checagem |
| `Author` | Agência responsável pela verificação |
| `datePublished` | Data de publicação da checagem |
| `claimReviewed` | Frase ou alegação analisada |
| `reviewBody` | Resumo ou conclusão da checagem |
| `title` | Título da matéria publicada |
| `ratingValue` | Nota numérica atribuída pela agência |
| `bestRating` | Escala máxima da agência |
| `alternativeName` | Veredito textual (ex.: Falso, Verdadeiro, Exagerado) |

### 2. `datasets/boatos_saude.tsv`
Matérias extraídas diretamente da seção de saúde do portal Boatos.org, contendo o texto completo do boato:

| Coluna | Descrição |
| :--- | :--- |
| `url` | Link original no Boatos.org |
| `title` | Título do artigo |
| `date` | Data de publicação |
| `author` | Nome do autor da checagem |
| `claim` | Frase principal ou resumo do boato |
| `rumors_text` | **Texto original transcrito do boato** (mensagens de WhatsApp, redes sociais, etc.) |
| `fact_check` | Explicação da desmistificação pelo jornalista |
| `verdict` | Classificação do Boatos.org (ex.: Boato, Golpe, Falso) |
| `sources` | Fontes e referências consultadas |

### 3. `datasets/community_notes_pt.tsv` e `datasets/community_notes_en.tsv`
Postagens do Twitter/X associadas a checagens colaborativas comunitárias (**Community Notes** / Birdwatch), separadas em arquivos distintos por idioma:

| Coluna | Descrição |
| :--- | :--- |
| `tweet_id` | Identificador único do Tweet |
| `note_id` | Identificador único da Nota da Comunidade |
| `claim` | **Texto original do tweet** recuperado via oEmbed oficial |
| `review` | **Resumo da checagem** redigido pelos contribuidores comunitários |
| `classification` | Classificação original (`MISINFORMED_OR_POTENTIALLY_MISLEADING` ou `NOT_MISLEADING`) |
| `is_fake` | **Classificação binária**: `1` (boato/fake news) ou `0` (notícia/postagem verdadeira) |
| `language` | Idioma validado (`pt` ou `en`) |
| `status` | Status de consenso da nota (`CURRENTLY_RATED_HELPFUL`, `NEEDS_MORE_RATINGS`, etc.) |
| `tweet_author` | Nome do autor da postagem |
| `tweet_url` | Link permanente da postagem no X |
| `misleading_reasons` | Tags detalhadas de motivo de erro (ex.: `factual_error`, `missing_context`) |
| `sources` | Fontes confiáveis citadas no review |
| `created_at` | Data e hora em formato UTC (`YYYY-MM-DD HH:MM:SS`) |

---

## 🚀 Como Usar os Scripts

### Instalação das Dependências

Recomenda-se o uso de um ambiente virtual Python (3.8+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Script 1: Atualização do FACTCK.BR (`scripts/update_factckbr.py`)

Atualiza `datasets/FACTCKBR.tsv` coletando checagens recentes via API oficial do Google Fact Check Tools (com chave) ou via sitemaps XML das agências (sem chave necessária). Exporta automaticamente o subconjunto exclusivo de saúde para `datasets/FACTCKBR_saude.tsv`.

```bash
# Modo automático (extrai até 50 novas checagens de saúde)
python scripts/update_factckbr.py --limit 50

# Utilizando sitemaps diretamente (sem necessidade de chave de API)
python scripts/update_factckbr.py --source sitemap --limit 100

# Utilizando Google Fact Check Tools API
python scripts/update_factckbr.py --source google-api --key "SUA_CHAVE_GOOGLE" --limit 200
```

---

### Script 2: Scraper de Notícias Falsas de Saúde (`scripts/scrape_boatos_saude.py`)

Extrai matérias do portal Boatos.org com suporte completo a **retomada automática (resume)** via checkpoint `.boatos_saude_checkpoint.json`.

```bash
# Coletar uma amostra de 5 páginas (~125 matérias)
python scripts/scrape_boatos_saude.py --pages 5

# Coletar todas as matérias disponíveis no site (modo contínuo)
python scripts/scrape_boatos_saude.py --pages 0

# Ajustar intervalo entre requisições (evitar bloqueios)
python scripts/scrape_boatos_saude.py --delay 0.8
```

---

### Script 3: Scraper do Twitter Community Notes (`scripts/scrape_community_notes.py`)

Extrai postagens do Twitter, obtém o texto completo do tweet via `oEmbed` (zero custo), detecta o idioma e salva separadamente em `datasets/community_notes_pt.tsv` e `datasets/community_notes_en.tsv` com rótulos binários de veracidade (`is_fake`).

```bash
# Extrair amostra de checagens em português
python scripts/scrape_community_notes.py --pt-only --limit 50

# Extrair amostra de checagens em inglês
python scripts/scrape_community_notes.py --en-only --limit 100

# Executar balanceando checagens falsas e verdadeiras (is_fake=1 e is_fake=0)
python scripts/scrape_community_notes.py --status-filter all --limit 500

# Usar apenas checagens com consenso comunitário aprovado (CURRENTLY_RATED_HELPFUL)
python scripts/scrape_community_notes.py --status-filter helpful --limit 200
```

---

## 📖 Referência Científica

Se você utilizar o FACTCK.BR ou suas derivações em sua pesquisa, por favor cite o artigo original:

```bibtex
@inproceedings{factckbr2019,
  author    = {Silva, J. and others},
  title     = {FACTCK.BR: A Dataset for Fake News Detection in Portuguese},
  booktitle = {Proceedings of the 25th Brazilian Symposium on Multimedia and the Web (WebMedia '19)},
  year      = {2019},
  pages     = {1--8},
  publisher = {ACM},
  address   = {Rio de Janeiro, Brazil},
  doi       = {10.1145/3323503.3361698}
}
```

---

## 📄 Licença

- O código-fonte e os scripts são disponibilizados sob a licença **MIT** (consulte o arquivo `LICENSE`).
- Os dados originais de checagem pertencem e são creditados às respectivas agências de verificação e contribuidores de cada plataforma.

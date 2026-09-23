# FACTCK.BR: Plataforma de Datasets e Ferramentas para Detecção de Desinformação em Saúde

O **FACTCK.BR** é um projeto e ecossistema de dados voltado ao estudo, pesquisa e desenvolvimento de modelos de Inteligência Artificial e Processamento de Linguagem Natural (NLP) para **detecção de desinformação, boatos (fake news) e notícias verdadeiras**, com ênfase primordial em língua portuguesa e no domínio de **saúde pública** (vacinas, tratamentos, epidemias, medicamentos e terapias alternativas).

Originalmente concebido como uma base de dados de checagens baseadas no esquema [ClaimReview](https://schema.org/ClaimReview) (publicado no simpósio WebMedia '19), o repositório foi modernizado e estendido para incluir:
1. **Base consolidada unificada de saúde** (`factckbr_boatos_saude_consolidado.tsv`) combinando transcrições originais de boatos e checagens profissionais com rotulagem booleana de veracidade (`is_fake`).
2. **Pipelines de atualização contínua** com a API oficial do Google Fact Check Tools e sitemaps de agências brasileiras (Aos Fatos, Agência Lupa, Boatos.org, Estadão Verifica, UOL Confere, etc.).
3. **Filtros temáticos especializados** com taxonomia médica e semântica refinada para saúde.
4. **Scrapers especializados** capazes de extrair o texto original completo de boatos de redes sociais/mensageiros e suas respectivas refutações jornalísticas.

---

## 📁 Estrutura do Repositório

```
FACTCK.BR/
├── datasets/                                 # Bases de dados em formato TSV (UTF-8) limpas e deduplicadas
│   ├── factckbr_boatos_saude_consolidado.tsv # Base UNIFICADA de saúde (2.928 checagens padronizadas com is_fake)
│   ├── FACTCKBR_updated.tsv                  # Dataset principal atualizado (multi-agências: 2.888 checagens)
│   ├── FACTCKBR_updated_saude.tsv            # Recorte especializado em saúde do FACTCK.BR (1.641 checagens)
│   ├── FACTCKBR_old.tsv                      # Dataset histórico original de referência (WebMedia 2019: 1.334 checagens)
│   └── boatos_saude.tsv                      # Matérias de saúde do Boatos.org (1.287 matérias com boato íntegro)
├── scripts/                                  # Scripts e pipelines executáveis
│   ├── consolidate_health_datasets.py        # Consolidador unificado de saúde (Boatos.org + FACTCKBR Saúde)
│   ├── update_factckbr.py                    # Atualizador multi-fonte com suporte a Google API e Sitemaps
│   ├── classify_health.py                    # Classificador e extrator temático de checagens de saúde
│   └── scrape_boatos_saude.py                # Scraper com suporte a retomada (resume) para o Boatos.org
├── requirements.txt                          # Dependências do Python
├── LICENSE                                   # Licença MIT
└── README.md                                 # Documentação geral do repositório
```

---

## 📊 Dicionário dos Datasets

### 1. `datasets/factckbr_boatos_saude_consolidado.tsv` (Base Unificada de Saúde)
Consolidação completa e padronizada das checagens de saúde do Boatos.org e do FACTCK.BR (2.928 checagens únicas, ordenadas por data decrescente):

| Coluna | Descrição |
| :--- | :--- |
| `URL` | Link do artigo de checagem |
| `Data` | Data de publicação da verificação (`YYYY-MM-DD`) |
| `Titulo` | Título da checagem jornalística |
| `Author` | Agência / veículo responsável (Boatos.org, Aos Fatos, Observador, UOL Confere, AFP Checamos, Estadão Verifica, Projeto Comprova, etc.) |
| `Claim` | Texto integral do boato / alegação analisada (`Texto_Falso_Original` ou `claimReviewed`) |
| `reviewBody` | Texto completo do desmentido / checagem explicativa |
| `is_fake` | Booleano declarando se a alegação é falsa (`True`) ou verdadeira (`False`) |

**Distribuição da Base Consolidada:**
- **Total de registros**: 2.928
- **Boatos / Fake news (`is_fake=True`)**: 2.904 (99.18%)
- **Fatos / Notícias verdadeiras (`is_fake=False`)**: 24 (0.82%)

---

### 2. `datasets/FACTCKBR_updated.tsv` e `datasets/FACTCKBR_updated_saude.tsv`
Bases tabulares contendo checagens estruturadas de agências de checagem profissionais (Aos Fatos, Lupa, UOL Confere, Estadão Verifica, AFP, Observador, etc.):

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
| `alternativeName` | Veredito textual (ex.: Falso, Verdadeiro, Exagerado, Enganoso) |

---

### 3. `datasets/boatos_saude.tsv`
Matérias extraídas diretamente da seção de saúde do portal Boatos.org, contendo a transcrição literal do boato:

| Coluna | Descrição |
| :--- | :--- |
| `URL` | Link original no Boatos.org |
| `Data` | Data de publicação |
| `Titulo` | Título do artigo |
| `Texto_Falso_Original` | **Texto original transcrito do boato** (mensagens de WhatsApp, posts de redes sociais, correntes) |
| `Resumo_Boato` | Frase principal ou síntese do boato |
| `Desmentido` | Explicação e desmistificação detalhada pelo jornalista |
| `Tags` | Categoria temática |

---

### 4. `datasets/FACTCKBR_old.tsv`
Dataset histórico de referência do artigo acadêmico original (WebMedia 2019), preservado como benchmark com 1.334 checagens.

---

## 🚀 Como Executar os Scripts

### Instalação das Dependências

Recomenda-se o uso de um ambiente virtual Python (3.8+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Script 1: Consolidação da Base de Saúde (`scripts/consolidate_health_datasets.py`)

Gera a base unificada padronizada a partir de `boatos_saude.tsv` e `FACTCKBR_updated_saude.tsv`:

```bash
python scripts/consolidate_health_datasets.py
```

Argumentos opcionais:
- `--boatos`: Caminho para o arquivo do Boatos.org (padrão: `datasets/boatos_saude.tsv`).
- `--factckbr`: Caminho para o arquivo FACTCKBR saúde (padrão: `datasets/FACTCKBR_updated_saude.tsv`).
- `--output`, `-o`: Caminho para o arquivo consolidado de saída (padrão: `datasets/factckbr_boatos_saude_consolidado.tsv`).

---

### Script 2: Atualização do FACTCK.BR (`scripts/update_factckbr.py`)

Atualiza `datasets/FACTCKBR_updated.tsv` coletando checagens recentes via API oficial do Google Fact Check Tools (com chave) ou via sitemaps XML das agências (sem necessidade de chave de API). Exporta automaticamente o subconjunto de saúde para `datasets/FACTCKBR_updated_saude.tsv`.

```bash
# Modo automático (extrai até 50 novas checagens de saúde via sitemaps)
python scripts/update_factckbr.py --limit 50

# Utilizando sitemaps diretamente
python scripts/update_factckbr.py --source sitemap --limit 100

# Utilizando Google Fact Check Tools API
python scripts/update_factckbr.py --source google-api --key "SUA_CHAVE_GOOGLE" --limit 200
```

---

### Script 3: Classificador e Extrator de Notícias de Saúde (`scripts/classify_health.py`)

Analisa qualquer base de checagens (ex.: `FACTCKBR_old.tsv`), aplicando taxonomia médica refinada para identificar desinformação em saúde, vacinas, medicamentos e saúde pública, desconsiderando metáforas políticas e termos policiais.

```bash
# Classificar FACTCKBR_old.tsv e extrair a base exclusiva de saúde
python scripts/classify_health.py --input datasets/FACTCKBR_old.tsv --output-health datasets/FACTCKBR_old_saude.tsv

# Classificar e gerar uma cópia anotada com is_health (1/0) e health_categories
python scripts/classify_health.py -i datasets/FACTCKBR_old.tsv -o datasets/FACTCKBR_old_saude.tsv -a datasets/FACTCKBR_old_annotated.tsv
```

---

### Script 4: Scraper de Notícias Falsas de Saúde (`scripts/scrape_boatos_saude.py`)

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
- Os dados originais de checagem pertencem e são creditados às respectivas agências de verificação e portais de informação.

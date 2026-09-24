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
│   ├── factckbr_boatos_saude_consolidado.tsv # Base UNIFICADA de saúde (5.359 checagens/notícias com balanceamento is_fake)
│   ├── FACTCKBR_updated.tsv                  # Dataset principal atualizado (multi-agências: 2.888 checagens)
│   ├── FACTCKBR_updated_saude.tsv            # Recorte especializado em saúde do FACTCK.BR (1.641 checagens)
│   ├── FACTCKBR_old.tsv                      # Dataset histórico original de referência (WebMedia 2019: 1.334 checagens)
│   ├── boatos_saude.tsv                      # Matérias de saúde do Boatos.org (1.287 matérias com boato íntegro)
│   ├── noticias_ms.tsv                       # Notícias do Ministério da Saúde (264 matérias com texto íntegro)
│   └── ebc_saude.tsv                         # Notícias de saúde da Agência Brasil / EBC (2.167 matérias íntegras com histórico)
├── scripts/                                  # Scripts e pipelines executáveis
│   ├── consolidate_health_datasets.py        # Consolidador unificado de saúde (Boatos.org + FACTCKBR + MS + EBC)
│   ├── update_factckbr.py                    # Atualizador multi-fonte com suporte a Google API e Sitemaps
│   ├── classify_health.py                    # Classificador e extrator temático de checagens de saúde
│   ├── scrape_boatos_saude.py                # Scraper com suporte a retomada (resume) para o Boatos.org
│   ├── scrape_noticias_ms.py                 # Scraper oficial de notícias do Ministério da Saúde (gov.br)
│   └── scrape_ebc_saude.py                   # Scraper oficial de notícias de saúde da Agência Brasil (EBC)
├── requirements.txt                          # Dependências do Python
├── LICENSE                                   # Licença MIT
└── README.md                                 # Documentação geral do repositório
```

---

## 📊 Dicionário dos Datasets

### 1. `datasets/factckbr_boatos_saude_consolidado.tsv` (Base Unificada de Saúde)
Consolidação completa e padronizada das checagens de saúde do Boatos.org, do FACTCK.BR, do Ministério da Saúde e da Agência Brasil/EBC (5.359 registros únicos, ordenados por data decrescente de 2013 a 2026):

| Coluna | Descrição |
| :--- | :--- |
| `URL` | Link do artigo de checagem ou notícia oficial |
| `Data` | Data de publicação da verificação (`YYYY-MM-DD`) |
| `Titulo` | Título da checagem jornalística ou notícia oficial |
| `Author` | Agência / veículo responsável (Agência Brasil, Boatos.org, Ministério da Saúde, Aos Fatos, Observador, UOL Confere, AFP Checamos, Estadão Verifica, Projeto Comprova, etc.) |
| `Claim` | Texto integral do boato / alegação analisada (`Texto_Falso_Original`, `claimReviewed` ou linha fina oficial) |
| `reviewBody` | Texto completo do desmentido / checagem explicativa ou matéria oficial |
| `is_fake` | Booleano declarando se a alegação é falsa (`True`) ou verdadeira (`False`) |

**Distribuição da Base Consolidada:**
- **Total de registros**: 5.359
- **Boatos / Fake news (`is_fake=True`)**: 2.904 (54,19%)
- **Fatos / Notícias verdadeiras (`is_fake=False`)**: 2.455 (45,81%)

**Composição por Veículo / Autor:**
| Veículo / Autor | Quantidade | % do Total | Natureza dos Registros |
| :--- | :---: | :---: | :--- |
| **Agência Brasil (EBC)** | 2.167 | 40,44% | Notícias oficiais verificadas (`is_fake=False`) |
| **Boatos.org** | 1.287 | 24,02% | Boatos de redes sociais / WhatsApp (`is_fake=True`) |
| **Aos Fatos** | 330 | 6,16% | Checagens jornalísticas |
| **Observador** | 310 | 5,78% | Checagens jornalísticas |
| **UOL Confere** | 265 | 4,94% | Checagens jornalísticas |
| **Ministério da Saúde** | 264 | 4,93% | Comunicados oficiais gov.br (`is_fake=False`) |
| **AFP Checamos** | 249 | 4,65% | Checagens jornalísticas |
| **Estadão Verifica** | 247 | 4,61% | Checagens jornalísticas |
| **Projeto Comprova** | 146 | 2,72% | Checagens colaborativas |
| **Agência Pública** | 54 | 1,01% | Checagens investigativas |
| **Agência Lupa** | 28 | 0,52% | Checagens jornalísticas |
| **Outros (O Globo, Agência Tatu, Nexo)** | 12 | 0,22% | Checagens jornalísticas |

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

### 4. `datasets/noticias_ms.tsv` (Notícias do Ministério da Saúde)
Matérias jornalísticas oficiais extraídas diretamente do portal [Notícias da Saúde](https://www.gov.br/saude/pt-br/assuntos/noticias-ms) do Ministério da Saúde (264 matérias íntegras):

| Coluna | Descrição |
| :--- | :--- |
| `URL` | Endereço oficial da matéria no portal gov.br |
| `Data` | Data de publicação (`YYYY-MM-DD`) |
| `Data_Hora` | Data e hora ISO 8601 (`YYYY-MM-DDTHH:MM:SS-03:00`) |
| `Data_Modificacao` | Data e hora de modificação ISO |
| `Titulo` | Título da notícia / comunicado |
| `Subtitulo` | Linha fina / subtítulo / resumo oficial |
| `Categoria` | Editoria temática oficial (ex.: *Saúde Global*, *Atenção Primária*, *Vigilância em Saúde*) |
| `Conteudo` | Texto integral limpo da matéria |
| `Author` | `Ministério da Saúde` |
| `is_fake` | `False` (comunicação oficial verificada da autoridade sanitária) |

> [!WARNING]
> **Limitação Técnica de Retenção do Portal:**
> A coleção de notícias do Ministério da Saúde (`gov.br/saude/pt-br/assuntos/noticias-ms`) opera sob uma política de retenção recente (*rolling window*) no CMS Plone do portal gov.br. O feed disponibiliza aproximadamente 18 páginas (~264 notícias), cobrindo apenas os últimos 2 a 3 meses (retroagindo apenas até o início de julho de 2026). A paginação é encerrada pelo servidor em `b_start:int=255`, impossibilitando a extração histórica contínua de anos anteriores por esse endpoint.

---

### 5. `datasets/ebc_saude.tsv` (Notícias de Saúde da Agência Brasil / EBC)
Base jornalística pública governamental com matérias completas extraídas do portal [Agência Brasil — Saúde](https://agenciabrasil.ebc.com.br/saude) da Empresa Brasil de Comunicação (EBC). Apresenta cobertura histórica de longo prazo cobrindo anos de políticas públicas, pesquisas, vacinas e ações sanitárias (atualmente com 2.167 matérias íntegras cobrindo de março de 2024 a setembro de 2026, coletadas ao longo de 156 páginas):

| Coluna | Descrição |
| :--- | :--- |
| `URL` | Link canônico da matéria no portal da Agência Brasil |
| `Data` | Data de publicação (`YYYY-MM-DD`) |
| `Data_Hora` | Data e hora de publicação ISO 8601 (`YYYY-MM-DDTHH:MM:00`) |
| `Titulo` | Manchete / título da matéria |
| `Subtitulo` | Linha fina oficial explicativa |
| `Local` | Cidade ou praça de apuração da reportagem (ex.: *Brasília*, *Rio de Janeiro*, *São Paulo*) |
| `Conteudo` | Texto integral limpo dos parágrafos da matéria |
| `Author` | Repórter responsável / `Agência Brasil` |
| `is_fake` | `False` (matéria pública oficial verificada) |

---

### 6. `datasets/FACTCKBR_old.tsv`
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

Gera a base unificada padronizada a partir de `boatos_saude.tsv`, `FACTCKBR_updated_saude.tsv`, `noticias_ms.tsv` e `ebc_saude.tsv`:

```bash
python scripts/consolidate_health_datasets.py
```

Argumentos opcionais:
- `--boatos`: Caminho para o arquivo do Boatos.org (padrão: `datasets/boatos_saude.tsv`).
- `--factckbr`: Caminho para o arquivo FACTCKBR saúde (padrão: `datasets/FACTCKBR_updated_saude.tsv`).
- `--noticias-ms`: Caminho para notícias do Ministério da Saúde (detectado automaticamente se `datasets/noticias_ms.tsv` existir).
- `--ebc-saude`: Caminho para notícias da Agência Brasil / EBC (detectado automaticamente se `datasets/ebc_saude.tsv` existir).
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

### Script 5: Scraper Oficial do Ministério da Saúde (`scripts/scrape_noticias_ms.py`)

Extrai matérias jornalísticas e comunicados oficiais íntegros do portal [Notícias da Saúde — Ministério da Saúde](https://www.gov.br/saude/pt-br/assuntos/noticias-ms). Suporta paginação Plone (`b_start:int`), **paralelismo multi-thread (`--workers`, `-w`)** com pool de conexões HTTP Keep-Alive, salvamento atômico thread-safe em TSV/CSV com flush, retomada automática (`.checkpoint_noticias_ms.json`), metadados estruturados (JSON-LD `NewsArticle`), tolerância a falhas (retries/backoff) e controle gracioso de interrupção (Ctrl+C).

```bash
# Coletar todas as matérias disponíveis em modo paralelo (padrão: 6 workers)
python scripts/scrape_noticias_ms.py --workers 6

# Coletar amostra inicial (ex.: 2 páginas, ~30 matérias)
python scripts/scrape_noticias_ms.py --max-pages 2

# Incluir também a subseção regional 'noticias-para-os-estados'
python scripts/scrape_noticias_ms.py --include-estados

# Exportar diretamente no formato compatível com o FACTCK.BR (Claim e reviewBody)
python scripts/scrape_noticias_ms.py --format factckbr -o datasets/noticias_ms_factckbr.tsv
```

---

### Script 6: Scraper de Notícias da Agência Brasil / EBC (`scripts/scrape_ebc_saude.py`)

Extrai matérias jornalísticas públicas com profundidade histórica do portal [Agência Brasil — Saúde](https://agenciabrasil.ebc.com.br/saude). Suporta paginação contínua (`?page=N`), acervo histórico de anos (>12.000 matérias), **paralelismo multi-thread de alta velocidade (`--workers`, `-w`)** com pool de conexões HTTP Keep-Alive, salvamento atômico thread-safe em TSV com flush, retomada automática (`.checkpoint_ebc_saude.json`), extração de manchete, linha fina, praça local, repórter e corpo íntegro, além de filtro por data mínima (`--min-date YYYY-MM-DD`).

```bash
# Coleta paralela de alta velocidade (padrão: 8 workers, ~3s por página)
python scripts/scrape_ebc_saude.py --max-pages 20 --workers 8

# Coleta com maior concorrência (ex.: 12 workers)
python scripts/scrape_ebc_saude.py --max-pages 50 --workers 12

# Coleta sequencial tradicional (1 worker)
python scripts/scrape_ebc_saude.py --max-pages 10 --workers 1 --delay 0.8

# Coletar matérias até uma data mínima histórica (ex.: desde 2020)
python scripts/scrape_ebc_saude.py --max-pages 0 --min-date 2020-01-01 --workers 8

# Exportar diretamente no formato compatível com o FACTCK.BR
python scripts/scrape_ebc_saude.py --format factckbr -o datasets/ebc_saude_factckbr.tsv
```

---

## 🏛️ Alternativas Governamentais para Notícias de Saúde

Conforme documentado, o feed de notícias do Ministério da Saúde (`gov.br/saude/pt-br/assuntos/noticias-ms`) possui uma **limitação arquitetural**: ele é configurado como um catálogo de notícias recentes (*rolling window* de ~2 a 3 meses, retroagindo apenas até o início de julho de 2026 com ~264 matérias). A paginação é encerrada pelo servidor em `b_start:int=255`, sem indexação histórica de anos anteriores nesse endpoint. Essa característica sugere que a view `/noticias-ms` foi reestruturada ou implementada recentemente como uma vitrine de notícias recentes, em vez de um repositório histórico contínuo.

Para projetos e pesquisas que necessitem de uma base histórica mais profunda de notícias oficiais e comunicações governamentais de saúde pública em língua portuguesa, recomendam-se as seguintes alternativas:

### 1. Agência Brasil — Editoria de Saúde (EBC)
- **Portal**: [Agência Brasil — Saúde](https://agenciabrasil.ebc.com.br/saude)
- **Perfil**: Agência pública oficial de notícias do Governo Federal (Empresa Brasil de Comunicação - EBC).
- **Cobertura**: Cobre todas as decisões, campanhas de vacinação, portarias do Ministério da Saúde, decisões da Anvisa, epidemiologia e ações do SUS com texto jornalístico completo.
- **Acervo**: Possui acervo histórico de mais de **15 anos**, com paginação contínua (`?page=N`).
- **Consideração Técnica**: Utiliza proteção Cloudflare/WAF, requerendo headers de navegador completos ou bibliotecas de automação para extração.

### 2. Agência Fiocruz de Notícias (Fundação Oswaldo Cruz)
- **Portal**: [Agência Fiocruz de Notícias](https://agencia.fiocruz.br) / [Portal Fiocruz](https://portal.fiocruz.br/noticias)
- **Perfil**: Principal instituição pública federal de ciência, tecnologia e saúde da América Latina, vinculada ao Ministério da Saúde.
- **Cobertura**: Artigos científicos, vigilância epidemiológica (Dengue, Covid-19, Mpox), desenvolvimento e produção de vacinas, e notas técnicas.
- **Acervo**: Arquivo contínuo e histórico indexado com anos de publicações categorizadas.

### 3. Portal Anvisa (Agência Nacional de Vigilância Sanitária)
- **Portal**: [Notícias da Anvisa](https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa)
- **Perfil**: Agência reguladora federal responsável pelo controle sanitário, farmacovigilância e aprovação de vacinas e medicamentos.
- **Cobertura**: Alertas de produtos e medicamentos irregulares/falsificados, autorizações de uso emergencial e resoluções sanitárias.

### 4. OpenDataSUS / DataSUS (Dados Abertos em Saúde)
- **Portal**: [OpenDataSUS](https://opendatasus.saude.gov.br)
- **Perfil**: Plataforma governamental de dados abertos para epidemiologia e saúde pública.
- **Cobertura**: Microdados estruturados de vacinação (SI-PNI), notificações de Síndrome Respiratória Aguda Grave (SRAG), Dengue e internações hospitalares (SIH/SUS).

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

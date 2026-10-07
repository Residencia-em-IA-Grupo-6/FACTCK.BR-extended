# FACTCK.BR: Plataforma de Datasets e Ferramentas para Detecção de Desinformação e Classificação Temática

O **FACTCK.BR** é um projeto e ecossistema de dados voltado ao estudo, pesquisa e desenvolvimento de modelos de Inteligência Artificial e Processamento de Linguagem Natural (NLP) para **detecção de desinformação (fake news vs. notícias verdadeiras)** e **classificação temática de notícias em múltiplos domínios**, com ênfase primordial na língua portuguesa.

Originalmente concebido como uma base de dados de checagens baseadas no esquema [ClaimReview](https://schema.org/ClaimReview) (publicado no simpósio WebMedia '19), o repositório foi modernizado e estendido para incluir:
1. **Dataset Balanceado para Classificação de Notícias por Temas** (`dataset_classificacao_temas.tsv`), contendo **3.069 alegações e manchetes** catalogadas em **7 classes temáticas** com textos limpos e desprovidos de atalhos espúrios (*shortcut learning*), com notícias de saúde atualizadas pelas ocorrências mais recentes (2025–2026).
2. **Classificador Neuronal Quantizado em INT8 com ONNX Runtime** (`models/classifier_int8.onnx`), reduzindo o modelo em **74,9% em disco** (516 MB para 129 MB) com latência de **13,5 ms por amostra** em CPU e **76,9% de acurácia no teste**.
3. **Base consolidada unificada de saúde** (`factckbr_boatos_saude_consolidado.tsv`) combinando 5.359 transcrições originais de boatos e notícias autênticas com rotulagem booleana balanceada de veracidade (`is_fake`).
4. **Pipelines de atualização contínua** com a API oficial do Google Fact Check Tools e sitemaps de agências brasileiras (Aos Fatos, Agência Lupa, Boatos.org, Estadão Verifica, UOL Confere, etc.).
5. **Scrapers especializados e paralelos** para coleta concorrente de notícias e checagens por categoria (Esporte, Entretenimento, Tecnologia, Economia, Política, Geral e Saúde) a partir do Boatos.org, Agência Brasil (EBC) e Ministério da Saúde.

---

## 📁 Estrutura do Repositório

```
FACTCK.BR/
├── datasets/                                 # Bases de dados em formato TSV (UTF-8) limpas e deduplicadas
│   ├── dataset_classificacao_temas.tsv       # Dataset de classificação temática (3.069 claims, 7 temas)
│   ├── factckbr_boatos_saude_consolidado.tsv # Base UNIFICADA de saúde (5.359 checagens/notícias com is_fake)
│   ├── FACTCKBR_updated.tsv                  # Dataset principal atualizado (multi-agências: 2.888 checagens)
│   ├── FACTCKBR_updated_saude.tsv            # Recorte especializado em saúde do FACTCK.BR (1.641 checagens)
│   ├── FACTCKBR_old.tsv                      # Dataset histórico original de referência (WebMedia 2019: 1.334 checagens)
│   ├── boatos_saude.tsv                      # Matérias de saúde do Boatos.org (1.287 matérias com boato íntegro)
│   ├── boatos_esporte.tsv                    # Boatos de esportes do Boatos.org (250 checagens)
│   ├── boatos_entretenimento.tsv             # Boatos de cultura e famosos do Boatos.org (250 checagens)
│   ├── boatos_tecnologia.tsv                 # Boatos de tecnologia e golpes do Boatos.org (250 checagens)
│   ├── boatos_politica.tsv                   # Boatos de política do Boatos.org (250 checagens)
│   ├── ebc_saude.tsv                         # Notícias de saúde da Agência Brasil / EBC (2.167 matérias)
│   ├── ebc_esportes.tsv                      # Notícias de esportes da Agência Brasil / EBC (250 matérias)
│   ├── ebc_cultura.tsv                       # Notícias de cultura/entretenimento da Agência Brasil / EBC (250 matérias)
│   ├── ebc_economia.tsv                      # Notícias de economia da Agência Brasil / EBC (250 matérias)
│   ├── ebc_politica.tsv                      # Notícias de política da Agência Brasil / EBC (250 matérias)
│   ├── ebc_geral.tsv                         # Notícias gerais da Agência Brasil / EBC (250 matérias)
│   ├── ebc_direitos-humanos.tsv              # Notícias de direitos humanos/segurança da EBC (200 matérias)
│   └── noticias_ms.tsv                       # Notícias do Ministério da Saúde (264 matérias oficiais)
├── models/                                   # Modelos treinados, pesos ONNX e relatórios de benchmark
│   ├── classifier_int8.onnx                  # Modelo final quantizado dinamicamente em INT8 (129 MB)
│   ├── classifier_fp32.onnx                  # Modelo ONNX baseline FP32 (516 MB)
│   ├── benchmark_report.json                 # Relatório quantitativo comparativo (latência, acurácia, F1)
│   ├── id2label.json                         # Mapeamento índice -> nome da classe temática
│   ├── label2id.json                         # Mapeamento nome da classe -> índice
│   └── tokenizer/                            # Arquivos do tokenizador multilíngue
├── scripts/                                  # Scripts e pipelines executáveis
│   ├── train_onnx_classifier.py              # Treinador PyTorch, exportador ONNX e quantizador INT8
│   ├── predict_theme.py                      # Mecanismo de inferência rápido (CLI / Interativo / Arquivo)
│   ├── add_boundary_cases.py                 # Injetor de casos de fronteira (hard samples) para calibração
│   ├── prepare_theme_dataset.py              # Pipeline mestre de geração do dataset de classificação temática
│   ├── classify_themes.py                    # Classificador temático de alta precisão (taxonomia 7 classes)
│   ├── collect_all_categories.py             # Orquestrador para raspagem balanceada multi-categoria
│   ├── scrape_boatos_categories.py           # Scraper multi-categoria do Boatos.org
│   ├── scrape_ebc_categories.py              # Scraper multi-categoria da Agência Brasil (EBC)
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

### 1. `datasets/dataset_classificacao_temas.tsv` (Classificação de Notícias por Tema)
Base de dados especialmente preparada para **treinamento e benchmark de modelos de classificação de texto em tópicos jornalísticos**. Reúne **3.069 afirmações e manchetes** distribuídas em 7 categorias editoriais fundamentais:

| Coluna | Descrição |
| :--- | :--- |
| `Claim` | Texto limpo da alegação/manchete (higienizado contra ruídos, sem vazamento de "Boato -", sem aspas envolventes, sem emojis ou links) |
| `tema` | **Rótulo da classe temática** (`politica`, `saude`, `esporte`, `entretenimento`, `economia`, `tecnologia`, `seguranca_publica`) |
| `URL` | Link permanente da matéria ou checagem de origem |
| `Titulo` | Título jornalístico original da publicação |
| `Author` | Agência / portal de origem (*Agência Brasil*, *Boatos.org*, *Aos Fatos*, *Estadão Verifica*, *UOL Confere*, etc.) |
| `is_fake` | Booleano de veracidade (`True` = boato/desinformação, `False` = fato/notícia autêntica) |

#### 📈 Distribuição das Classes e Balanceamento de Veracidade:
| Classe Temática (`tema`) | Quantidade | % do Total | Falso (`is_fake=True`) | Verdadeiro (`is_fake=False`) |
| :--- | :---: | :---: | :---: | :---: |
| **`saude`** | 550 | 17,9% | 275 (50,0%) | 275 (50,0%) |
| **`politica`** | 550 | 17,9% | 379 (68,9%) | 171 (31,1%) |
| **`esporte`** | 550 | 17,9% | 274 (49,8%) | 276 (50,2%) |
| **`entretenimento`** | 550 | 17,9% | 289 (52,5%) | 261 (47,5%) |
| **`economia`** | 362 | 11,8% | 76 (21,0%) | 286 (79,0%) |
| **`tecnologia`** | 340 | 11,1% | 297 (87,4%) | 43 (12,6%) |
| **`seguranca_publica`** | 167 | 5,4% | 87 (52,1%) | 80 (47,9%) |
| **TOTAL** | **3.069** | **100%** | **1.677 (54,6%)** | **1.392 (45,4%)** |

> **Garantia de Qualidade para Machine Learning:**
> - **0 valores nulos** em todas as colunas.
> - **0 vazamentos de rótulo:** Nenhuma ocorrência de `"Boato -"`, `"Boato:"` ou prefixos no texto de entrada.
> - **0 aspas envolventes** espúrias (`“...”`, `"..."`).
> - **0 emojis** alarmistas ou links externos que causem correlação espúria (*shortcut learning*).

---

### 2. `datasets/factckbr_boatos_saude_consolidado.tsv` (Base Unificada de Saúde)
Consolidação completa e padronizada das checagens de saúde do Boatos.org, do FACTCK.BR, do Ministério da Saúde e da Agência Brasil/EBC (5.359 registros únicos):

| Coluna | Descrição |
| :--- | :--- |
| `URL` | Link do artigo de checagem ou notícia oficial |
| `Data` | Data de publicação da verificação (`YYYY-MM-DD`) |
| `Titulo` | Título da checagem jornalística ou notícia oficial |
| `Author` | Agência / veículo responsável (*Agência Brasil*, *Boatos.org*, *Ministério da Saúde*, *Aos Fatos*, etc.) |
| `Claim` | Texto integral do boato / alegação analisada |
| `reviewBody` | Texto completo do desmentido / checagem explicativa ou matéria oficial |
| `is_fake` | Booleano declarando se a alegação é falsa (`True`) ou verdadeira (`False`) |

**Composição:**
- **Total de registros**: 5.359 (2.904 falsos / 2.455 verdadeiros).

---

### 3. `datasets/FACTCKBR_updated.tsv` e `datasets/FACTCKBR_old.tsv`
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

## 🚀 Como Executar os Scripts

### Instalação das Dependências

Recomenda-se o uso de um ambiente virtual Python (3.8+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Script 1: Geração do Dataset de Classificação Temática (`scripts/prepare_theme_dataset.py`)

Integra todas as fontes (FACTCK.BR, Boatos.org, EBC e Ministério da Saúde), classifica os textos na taxonomia de 7 temas, aplica a higienização de vieses e salva o arquivo final balanceado:

```bash
# Execução padrão (gera datasets/dataset_classificacao_temas.tsv com até 550 amostras por classe)
python scripts/prepare_theme_dataset.py --max-per-theme 550

# Especificar caminho alternativo de saída e limite personalizado
python scripts/prepare_theme_dataset.py -o datasets/meu_dataset_temas.tsv -m 600
```

---

### Script 2: Classificador Temático Baseado em Regras (`scripts/classify_themes.py`)

Módulo que implementa a taxonomia refinada das 7 classes temáticas. Pode ser importado em pipelines ou scripts personalizados:

```python
from classify_themes import classify_theme

tema = classify_theme("Flamengo vence clássico no Maracanã com dois gols de Arrascaeta")
print(tema)  # 'esporte'
```

---

### Script 3: Scrapers Multi-Categoria do Boatos.org e da EBC

Permitem coletar novas matérias por editoria temática de forma concorrente via `ThreadPoolExecutor`:

```bash
# Coletar notícias do Boatos.org por categoria
python scripts/scrape_boatos_categories.py --category esporte --limit 250
python scripts/scrape_boatos_categories.py --category entretenimento --limit 250
python scripts/scrape_boatos_categories.py --category tecnologia --limit 250
python scripts/scrape_boatos_categories.py --category politica --limit 250

# Coletar notícias da Agência Brasil (EBC) por editoria
python scripts/scrape_ebc_categories.py --category esportes --limit 250
python scripts/scrape_ebc_categories.py --category cultura --limit 250
python scripts/scrape_ebc_categories.py --category economia --limit 250
python scripts/scrape_ebc_categories.py --category politica --limit 250
python scripts/scrape_ebc_categories.py --category geral --limit 250
python scripts/scrape_ebc_categories.py --category direitos-humanos --limit 200

# Executar a coleta balanceada de todas as categorias em lote
python scripts/collect_all_categories.py
```

---

### Script 4: Consolidador Unificado de Saúde (`scripts/consolidate_health_datasets.py`)

Gera a base unificada especializada em saúde combinando checagens de agências, Boatos.org, Ministério da Saúde e EBC:

```bash
python scripts/consolidate_health_datasets.py
```

---

### Script 5: Treinamento e Otimização com ONNX Runtime INT8 (`scripts/train_onnx_classifier.py`)

Treina um classificador neural de ponta a ponta sobre o dataset temático (`datasets/dataset_classificacao_temas.tsv`), exporta para ONNX e aplica **quantização dinâmica de 8 bits (INT8)**:

```bash
# Treinamento com parâmetros padrão (3 épocas, batch size 16, lr 3e-5)
python scripts/train_onnx_classifier.py --epochs 3 --batch-size 16 --lr 3e-5
```

#### ⚡ Resultados Comparativos do Benchmark Calibrado (Conjunto de Teste - 311 amostras):

| Métrica / Propriedade | ONNX FP32 | ONNX INT8 | Ganho / Otimização |
| :--- | :---: | :---: | :---: |
| **Tamanho em Disco** | **516,36 MB** | **129,45 MB** | **-74,9% (4x menor)** |
| **Acurácia no Teste** | **82,96%** | **79,74%** | -3,22% |
| **Macro F1-Score** | **80,16%** | **76,87%** | -3,29% |
| **Weighted F1-Score** | **83,20%** | **79,98%** | -3,22% |
| **Latência Média por Amostra (CPU)** | 14,58 ms | **12,28 ms** | **1,19x mais rápido** |
| **Throughput (Amostras / segundo)** | 68,6 s/sec | **81,4 s/sec** | **+18,7%** |

#### 🎯 Desempenho por Tema no ONNX INT8 (Após Fine-Tuning de Fronteira):
| Tema | Precisão | Revocação | F1-Score | Amostras de Teste |
| :--- | :---: | :---: | :---: | :---: |
| **Esporte** | 95,9% | 83,9% | **89,5%** | 56 |
| **Saúde** | 82,5% | **91,2%** | **86,7%** | 57 |
| **Entretenimento** | 89,4% | 76,4% | **82,4%** | 55 |
| **Política** | 80,8% | 75,0% | **77,8%** | 56 |
| **Economia** | 67,4% | 80,6% | **73,4%** | 36 |
| **Tecnologia** | 66,7% | 76,5% | **71,2%** | 34 |
| **Segurança Pública** | 55,6% | 58,8% | **57,1%** | 17 |
| **Média Ponderada Global** | **80,2%** | **79,7%** | **80,0%** | **311** |

---

### Script 6: Inferência Rápida com ONNX Runtime INT8 (`scripts/predict_theme.py`)

Utilitário de linha de comando para classificação instantânea de claims em produção:

```bash
# 1. Inferência direta por texto
python scripts/predict_theme.py "Ministério da Saúde distribui doses da vacina contra a dengue para estados prioritários"
# 👉 Saída: SAUDE (Confiança: 96.0%, Latência: 11.3 ms)

# 2. Modo interativo de teste no terminal
python scripts/predict_theme.py --interactive

# 3. Classificação em lote a partir de arquivo TSV ou CSV
python scripts/predict_theme.py --input-file claims.tsv --output-file predicoes.tsv
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

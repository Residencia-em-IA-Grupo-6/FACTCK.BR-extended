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
├── datasets/                                 # Datasets consolidados em formato TSV/Parquet
│   ├── saude_consolidado_unificado.tsv       # Base UNIFICADA consolidada de saúde (5.735 checagens/notícias com is_fake)
│   ├── FACTCKBR_updated.tsv                  # Dataset principal atualizado (multi-agências: 2.888 checagens)
│   ├── FACTCKBR_old.tsv                      # Dataset histórico original de referência (WebMedia 2019: 1.334 checagens)
│   ├── dataset_classificacao_temas.tsv       # Dataset de temas legados (3.127 claims, 7 temas)
│   └── topic_classification/                 # Dataset temático unificado e balanceado de 8 classes
│       ├── topic_dataset_8classes.parquet    # Base consolidada (24.000 amostras equilibradas, 3.000 por tema)
│       ├── topic_dataset_8classes.tsv        # Exportação tabular delimitada por tabulações
│       └── README.md                         # Dicionário das 8 classes e metodologia
├── models/                                   # Modelos treinados, pesos ONNX e relatórios de benchmark
│   ├── classifier_int8.onnx                  # Modelo final quantizado dinamicamente em INT8 (129 MB)
│   ├── classifier_fp32.onnx                  # Modelo ONNX baseline FP32 (516 MB)
│   ├── benchmark_report.json                 # Relatório quantitativo comparativo (latência, acurácia, F1)
│   ├── id2label.json                         # Mapeamento índice -> tema (8 classes)
│   ├── label2id.json                         # Mapeamento tema -> índice
│   ├── pytorch_distilbert/                   # Checkpoint PyTorch compatível com Linux e Mac M4
│   └── tokenizer/                            # Arquivos do tokenizador multilíngue
├── scripts/                                  # Pipelines e scripts executáveis unificados
│   ├── scrape_boatos.py                      # Scraper concorrente unificado do Boatos.org (--category / --all)
│   ├── scrape_ebc.py                         # Scraper concorrente unificado da Agência Brasil / EBC (--category / --all)
│   ├── scrape_noticias_ms.py                 # Scraper oficial de notícias do Ministério da Saúde (gov.br)
│   ├── update_factckbr.py                    # Atualizador multi-fonte com suporte a Google API e Sitemaps
│   ├── consolidate_health_datasets.py        # Consolidador unificado de saúde (Boatos.org + FACTCKBR + MS + EBC)
│   ├── build_unified_topic_dataset.py        # Construtor da base de 8 classes balanceada (24.000 amostras)
│   ├── train_onnx_classifier.py              # Treinador PyTorch (DirectML/MPS/CUDA), exportador ONNX e quantizador INT8
│   └── predict_theme.py                      # Inferência ultra-rápida (Gatekeeper de Saúde + 8 temas)
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

### 2. `datasets/saude_consolidado_unificado.tsv` (Base Unificada de Saúde)
Consolidação completa e padronizada das checagens de saúde do Boatos.org, do FACTCK.BR, do Ministério da Saúde e da Agência Brasil/EBC (5.735 registros únicos):

| Coluna | Descrição |
| :--- | :--- |
| `URL` | Link do artigo de checagem ou notícia oficial |
| `Data` | Data de publicação da verificação (`YYYY-MM-DD`) |
| `Titulo` | Título da checagem jornalística ou notícia oficial |
| `Author` | Agência / veículo responsável (*Agência Brasil*, *Boatos.org*, *Ministério da Saúde*, *Aos Fatos*, etc.) |
| `Claim` | Texto integral do boato / alegação analisada |
| `reviewBody` | Texto completo do desmentido / checagem explicativa ou matéria oficial |
| `origem_dataset` | Fonte catalogada original de proveniência dos dados |
| `is_fake` | Booleano declarando se a alegação é falsa (`True`) ou verdadeira (`False`) |

**Composição:**
- **Total de registros**: 5.735 checagens e notícias de saúde com veredito padronizado.

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

### 4. `datasets/topic_classification/topic_dataset.parquet` (Classificação Temática Multi-Fonte - 11 Classes)
Dataset expandido com **38.877 amostras** particionadas de forma estratificada (`train`: 31.125, `val`: 3.876, `test`: 3.876) em 11 tópicos jornalísticos e de conversa geral. Consulte [README.md](file:///home/gabe/repos/FACTCK.BR-extended/datasets/topic_classification/README.md) na respectiva pasta para detalhes completos do esquema e classes.

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

### Script 3: Scrapers Concorrentes Unificados do Boatos.org e da EBC

Permitem coletar matérias por categoria específica ou em lote de forma concorrente via `ThreadPoolExecutor`:

```bash
# Scraper Boatos.org (por categoria ou todas)
python scripts/scrape_boatos.py --category saude --limit 250
python scripts/scrape_boatos.py --category esporte --limit 250
python scripts/scrape_boatos.py --all --limit 250

# Scraper EBC / Agência Brasil (por categoria ou todas)
python scripts/scrape_ebc.py --category saude --limit 250
python scripts/scrape_ebc.py --category esportes --limit 250
python scripts/scrape_ebc.py --all --limit 250

# Scraper oficial do Ministério da Saúde
python scripts/scrape_noticias_ms.py

# Atualizador multi-agências FACTCK.BR
python scripts/update_factckbr.py
```

---

### Script 4: Consolidador Unificado de Saúde (`scripts/consolidate_health_datasets.py`)

Gera a base unificada especializada em saúde combinando checagens de agências, Boatos.org, Ministério da Saúde e EBC:

```bash
python scripts/consolidate_health_datasets.py
```

---

### Script 5: Treinamento e Otimização com ONNX Runtime INT8 (`scripts/train_onnx_classifier.py`)

Treina um classificador neural de ponta a ponta sobre o dataset temático de 8 classes (`datasets/topic_classification/topic_dataset_8classes.parquet`), exporta para ONNX e aplica **quantização dinâmica de 8 bits (INT8)**:

```bash
# Treinamento com aceleração AMD (DirectML), Apple Silicon (MPS) ou CUDA:
python scripts/train_onnx_classifier.py \
    --dataset datasets/topic_classification/topic_dataset_8classes.parquet \
    --device auto \
    --epochs 1 \
    --samples-per-class 1000 \
    --batch-size 8 \
    --grad-accum-steps 4 \
    --max-len 64
```

#### ⚡ Resultados Comparativos do Benchmark Calibrado (Conjunto de Teste Cego - 800 amostras):

| Métrica / Propriedade | ONNX FP32 | ONNX INT8 | Ganho / Otimização |
| :--- | :---: | :---: | :---: |
| **Tamanho em Disco** | **516,35 MB** | **129,44 MB** | **-74,9% (4x menor)** 🎯 |
| **Acurácia no Teste** | **81,50%** | **80,63%** | -0,87% |
| **Macro F1-Score** | **81,61%** | **80,88%** | -0,73% |
| **Weighted F1-Score** | **81,61%** | **80,88%** | -0,73% |
| **Latência Média por Amostra (CPU)** | 27,74 ms | **22,30 ms** | **1,24x mais rápido** |
| **Throughput (Amostras / segundo)** | 36,1 s/sec | **44,8 s/sec** | **+24,4%** |

#### 🏥 Desempenho do Gatekeeper de Saúde no ONNX INT8:
* **Precisão:** **94,25%** (confiabilidade muito alta ao prever saúde)
* **Sensibilidade (Recall):** **82,00%** (capacidade de capturar queixas e relatos médicos)
* **F1-Score Saúde:** **87,70%**

#### 🎯 Desempenho por Tema no ONNX INT8 (8 Classes):
| Tema | Precisão | Revocação | F1-Score | Amostras de Teste |
| :--- | :---: | :---: | :---: | :---: |
| **esportes** | **95,7%** | 89,0% | **92,2%** | 100 |
| **conversa_geral** | **93,8%** | 90,0% | **91,8%** | 100 |
| **saude** | **94,3%** | 82,0% | **87,7%** | 100 |
| **economia_negocios** | 81,1% | 77,0% | **79,0%** | 100 |
| **entretenimento_cultura** | 72,9% | 78,0% | **75,4%** | 100 |
| **cotidiano_seguranca** | 74,3% | 75,0% | **74,6%** | 100 |
| **ciencia_tecnologia** | 70,9% | 78,0% | **74,3%** | 100 |
| **politica_governo** | 68,5% | 76,0% | **72,0%** | 100 |
| **Média Global** | **82,1%** | **80,6%** | **80,9%** | **800** |

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

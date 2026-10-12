# Dataset Balanceado de Tópicos e Gatekeeper de Saúde (8 Classes)

Este dataset reúne **24.000 amostras textuais rigorosamente balanceadas** (3.000 por classe) em português, com divisão fixa estratificada (80% treino, 10% validação, 10% teste).

Foi construído combinando:
1. **Perguntas reais de pacientes e consultas médicas** (`MedPT`).
2. **Desabafos e linguagem de saúde mental** (`MentalHealth-PT`).
3. **Conversas e mensagens informais de redes sociais / tweets** (`pt_tweets`, `told-br`).
4. **Notícias jornalísticas e manchetes** (`Folha de S.Paulo`, `Agência Brasil`).
5. **Alegações e checagens factuais higienizadas** (`FACTCK.BR`, `FakeRecogna`, `Boatos.org`).

---

## Esquema das Classes (8 Classes Funcionais)

| Índice | Classe (`topic`) | Quantidade | Descrição / Domínio |
| :---: | :--- | :---: | :--- |
| **0** | `ciencia_tecnologia` | 3.000 | Tecnologia, gadgets, software, inteligência artificial, internet, inovação |
| **1** | `conversa_geral` | 3.000 | Saudações, chitchat, bate-papo informal de redes sociais, tweets casuais |
| **2** | `cotidiano_seguranca` | 3.000 | Ocorrências policiais, crimes, trânsito, acidentes, defesa civil, cotidiano |
| **3** | `economia_negocios` | 3.000 | Mercado financeiro, empresas, emprego, inflação, salários, investimentos |
| **4** | `entretenimento_cultura` | 3.000 | Famosos, celebridades, filmes, música, reality shows, streaming, arte |
| **5** | `esportes` | 3.000 | Futebol, campeonatos, atletas, transferências, modalidades esportivas |
| **6** | `politica_governo` | 3.000 | Eleições, partidos, congresso, governo, STF, geopolítica e relações internacionais |
| **7** | `saude` | 3.000 | **Domínio Alvo**: Sintomas, remédios, vacinas, doenças, SUS, nutrição, dúvidas médicas |
| **Total** | | **24.000** | **Perfeitamente balanceado (12,5% por classe)** |

---

## Partições Estratificadas (`split`)

- **`train`**: 19.200 amostras (80%)
- **`val`**: 2.400 amostras (10%)
- **`test`**: 2.400 amostras (10%)

---

## Formatos Disponíveis

- `topic_dataset_8classes.parquet` (ou `topic_dataset.parquet`): formato colunar otimizado.
- `topic_dataset_8classes.tsv` (ou `topic_dataset.tsv`): formato tabular delimitado por tabulação.

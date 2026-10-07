#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_boundary_cases.py

Adiciona exemplos de fronteira (hard negatives / cross-domain samples)
ao dataset de classificação temática para calibrar o modelo em casos ambíguos:
- Esporte vs. Saúde (ex: esportes/exercícios com alegações médicas de infarto, arritmia, lesão, prevenção)
- Entretenimento vs. Saúde (ex: dietas de famosos, procedimentos médicos, doenças de celebridades)
- Tecnologia vs. Segurança Pública (ex: golpes cibernéticos, ataques hackers, roubo de dados)
- Política vs. Economia (ex: reformas tributárias, inflação, salário mínimo com contexto governamental)
"""

import pandas as pd
from pathlib import Path

DATASET_PATH = Path("datasets/dataset_classificacao_temas.tsv")

BOUNDARY_SAMPLES = [
    # --- ESPORTE x SAÚDE (Alegações fisiológicas, médicas ou patológicas -> SAUDE) ---
    {
        "Claim": "jogar futebol causa infarto",
        "tema": "saude",
        "Titulo": "Prática de futebol e risco cardiovascular",
        "Author": "Checagem Especializada",
        "is_fake": True,
        "URL": "https://factckbr.org/saude/futebol-infarto"
    },
    {
        "Claim": "jogar futebol intenso sem avaliação médica prévia pode provocar infarto do miocárdio ou morte súbita",
        "tema": "saude",
        "Titulo": "Avaliação cardiológica antes de jogar futebol",
        "Author": "Ministério da Saúde",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/futebol-avaliacao-cardiologica"
    },
    {
        "Claim": "futebol de fim de semana sem preparo físico eleva risco de parada cardíaca e infarto em pessoas sedentárias",
        "tema": "saude",
        "Titulo": "Cardiologistas alertam para futebol de fim de semana",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/futebol-sedentarismo-infarto"
    },
    {
        "Claim": "jogar futebol frequentemente previne doenças cardiovasculares, infarto e controle da pressão arterial",
        "tema": "saude",
        "Titulo": "Benefícios do futebol para a saúde do coração",
        "Author": "Ministério da Saúde",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/futebol-prevencao-infarto"
    },
    {
        "Claim": "jogar bola e praticar esportes coletivos aumenta imunidade e reduz risco de diabetes tipo 2",
        "tema": "saude",
        "Titulo": "Esportes e prevenção ao diabetes",
        "Author": "Ministério da Saúde",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/esportes-diabetes"
    },
    {
        "Claim": "estudo afirma que correr maratona e jogar futebol danifica as artérias do coração e causa infarto fulminante",
        "tema": "saude",
        "Titulo": "É boato que futebol e maratona causam infarto fulminante",
        "Author": "Boatos.org",
        "is_fake": True,
        "URL": "https://factckbr.org/saude/maratona-futebol-infarto-boato"
    },
    {
        "Claim": "fazer musculação diariamente provoca rompimento de ligamentos, lesões na coluna e problemas renais",
        "tema": "saude",
        "Titulo": "Cuidados com excesso de carga na musculação",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/musculacao-lesoes"
    },
    {
        "Claim": "atleta de futebol teve parada respiratória e infarto devido a reação adversa de remédio proibido",
        "tema": "saude",
        "Titulo": "Medicamentos e risco de infarto em atletas",
        "Author": "Aos Fatos",
        "is_fake": True,
        "URL": "https://factckbr.org/saude/atleta-remedio-infarto"
    },
    {
        "Claim": "natação e hidroginástica ajudam no tratamento de asma, bronquite e fortalecimento pulmonar",
        "tema": "saude",
        "Titulo": "Natação no auxílio respiratório",
        "Author": "Ministério da Saúde",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/natacao-asma-pulmao"
    },
    {
        "Claim": "praticar corrida de rua em jejum causa hipoglicemia severa, desmaio e arritmia cardíaca",
        "tema": "saude",
        "Titulo": "Riscos da corrida em jejum sem acompanhamento",
        "Author": "UOL Confere",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/corrida-jejum-arritmia"
    },
    {
        "Claim": "suplementos de creatina e pré-treino tomados por jogadores de futebol causam falência renal e câncer de fígado",
        "tema": "saude",
        "Titulo": "Mitos sobre creatina e lesão renal",
        "Author": "Boatos.org",
        "is_fake": True,
        "URL": "https://factckbr.org/saude/creatina-rim-figado"
    },
    {
        "Claim": "caminhadas diárias de trinta minutos reduzem em até 40% a probabilidade de sofrer um infarto ou AVC",
        "tema": "saude",
        "Titulo": "Caminhada como prevenção ao infarto",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/caminhada-infarto-avc"
    },
    {
        "Claim": "excesso de exercícios físicos causa rabdomiólise e insuficiência renal aguda em praticantes de crossfit",
        "tema": "saude",
        "Titulo": "Rabdomiólise e esforço físico extremo",
        "Author": "Estadão Verifica",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/crossfit-rabdomiolise-rim"
    },
    {
        "Claim": "jogar futebol na chuva atrai pneumonia viral e colapso respiratório imediato",
        "tema": "saude",
        "Titulo": "Chuva não causa pneumonia viral",
        "Author": "Boatos.org",
        "is_fake": True,
        "URL": "https://factckbr.org/saude/futebol-chuva-pneumonia"
    },

    # --- CONTROLE ESPORTE (Ações de jogo, campeonatos, transferências -> ESPORTE) ---
    {
        "Claim": "jogar futebol com regras oficiais da Fifa exige onze jogadores em cada time e noventa minutos de partida",
        "tema": "esporte",
        "Titulo": "Regras oficiais do futebol mundial",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/esporte/regras-futebol-fifa"
    },
    {
        "Claim": "jogar futebol no estádio do Maracanã é o sonho de jovens atletas das categorias de base dos clubes cariocas",
        "tema": "esporte",
        "Titulo": "Categorias de base no Maracanã",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/esporte/maracana-base-futebol"
    },
    {
        "Claim": "Flamengo vence clássico de futebol contra o Fluminense com gol histórico nos acréscimos do segundo tempo",
        "tema": "esporte",
        "Titulo": "Flamengo vence Fluminense no clássico",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/esporte/flamengo-fluminense-classico"
    },
    {
        "Claim": "CBF anuncia novo técnico para comandar a seleção brasileira de futebol nas eliminatórias da Copa do Mundo",
        "tema": "esporte",
        "Titulo": "Novo técnico da seleção brasileira",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/esporte/cbf-selecao-copa"
    },
    {
        "Claim": "Corinthians e Palmeiras empatam sem gols em partida decisiva da semifinal do campeonato paulista de futebol",
        "tema": "esporte",
        "Titulo": "Dérbi paulista termina empatado",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/esporte/corinthians-palmeiras-derbi"
    },
    {
        "Claim": "árbitro de futebol marca pênalti polêmico após consulta ao VAR na final da Libertadores da América",
        "tema": "esporte",
        "Titulo": "Pênalti polêmico na Libertadores",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/esporte/var-libertadores-penalti"
    },

    # --- ENTRETENIMENTO x SAÚDE (Doenças, procedimentos e dietas de celebridades -> SAUDE) ---
    {
        "Claim": "dieta restritiva de cantora famosa à base de água com limão emagrece 10 quilos e cura diabetes",
        "tema": "saude",
        "Titulo": "Dieta da água com limão não cura diabetes",
        "Author": "Boatos.org",
        "is_fake": True,
        "URL": "https://factckbr.org/saude/dieta-limao-diabetes-famosa"
    },
    {
        "Claim": "remédio caseiro recomendado por ator em entrevista na televisão elimina tumores e cura o câncer",
        "tema": "saude",
        "Titulo": "Remédio caseiro de ator não cura câncer",
        "Author": "Aos Fatos",
        "is_fake": True,
        "URL": "https://factckbr.org/saude/ator-remedio-cancer"
    },
    {
        "Claim": "uso abusivo de anabolizantes e cirurgias estéticas radicais por influenciadores causa insuficiência cardíaca",
        "tema": "saude",
        "Titulo": "Riscos de anabolizantes e plásticas excessivas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/anabolizantes-plastica-coracao"
    },
    {
        "Claim": "diagnóstico precoce de câncer de próstata e mama aumenta em 90% as chances de cura completa dos pacientes",
        "tema": "saude",
        "Titulo": "Campanha nacional de diagnóstico precoce",
        "Author": "Ministério da Saúde",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/cancer-diagnostico-precoce"
    },

    # --- TECNOLOGIA x SEGURANÇA PÚBLICA (Golpes, crimes virtuais, quadrilhas -> SEGURANCA_PUBLICA) ---
    {
        "Claim": "polícia civil desmonta quadrilha criminosa especializada em invasão de contas bancárias e golpes pelo WhatsApp",
        "tema": "seguranca_publica",
        "Titulo": "Polícia prende quadrilha do WhatsApp",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/seguranca/quadrilha-whatsapp-presa"
    },
    {
        "Claim": "criminosos armados sequestram empresário após monitorar movimentações em aplicativo de mensagens e GPS",
        "tema": "seguranca_publica",
        "Titulo": "Sequestro com rastreamento eletrônico",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/seguranca/sequestro-gps-app"
    },
    {
        "Claim": "operação da polícia federal cumpre mandados de prisão contra autores de crimes cibernéticos e pornografia infantil",
        "tema": "seguranca_publica",
        "Titulo": "Operação da PF contra crimes virtuais",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/seguranca/pf-crimes-virtuais-prisao"
    },

    # --- TECNOLOGIA PURA (Inovações, inteligência artificial, smartphones -> TECNOLOGIA) ---
    {
        "Claim": "nova atualização de inteligência artificial do ChatGPT e Google melhora tradução simultânea e escrita de código",
        "tema": "tecnologia",
        "Titulo": "Atualização de IA para código e idiomas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/tecnologia/ia-chatgpt-codigo"
    },
    {
        "Claim": "tecnologia 5G e redes de fibra óptica atingem novas capitais brasileiras com maior velocidade de conexão móvel",
        "tema": "tecnologia",
        "Titulo": "Expansão do 5G no território nacional",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/tecnologia/expansao-5g-fibra"
    },

    # --- POLÍTICA x ECONOMIA (Indicadores econômicos decididos pelo governo -> ECONOMIA) ---
    {
        "Claim": "governo federal anuncia corte de impostos sobre produtos da cesta básica para conter inflação e alta dos alimentos",
        "tema": "economia",
        "Titulo": "Corte de impostos e inflação de alimentos",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/economia/impostos-cesta-basica-inflacao"
    },
    {
        "Claim": "Banco Central decide manter a taxa Selic em 10,50% ao ano para controlar expectativas de inflação no mercado",
        "tema": "economia",
        "Titulo": "Copom mantém taxa Selic",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/economia/taxa-selic-copom-inflacao"
    },
    {
        "Claim": "reajuste do salário mínimo para 2026 injetará bilhões de reais no comércio varejista e no consumo das famílias",
        "tema": "economia",
        "Titulo": "Impacto do salário mínimo na economia",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/economia/salario-minimo-comercio"
    },

    # --- POLÍTICA PURA (Eleições, votações no Congresso, mandatos -> POLITICA) ---
    {
        "Claim": "Congresso Nacional aprova nova legislação eleitoral sobre mandatos e financiamento de campanhas partidárias",
        "tema": "politica",
        "Titulo": "Congresso vota regras eleitorais",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/congresso-regras-eleitorais"
    },
    {
        "Claim": "Tribunal Superior Eleitoral homologa prestação de contas dos partidos políticos e define calendário das eleições",
        "tema": "politica",
        "Titulo": "TSE define calendário eleitoral",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/tse-calendario-eleicoes"
    },

    # --- ESCOLAS / EDUCAÇÃO x POLÍTICA (Decisões de prefeitura, decretos, MEC, leis -> POLITICA) ---
    {
        "Claim": "escolas são fechadas no município de são paulo",
        "tema": "politica",
        "Titulo": "Prefeitura de São Paulo determina recesso e fechamento de escolas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/escolas-fechadas-sao-paulo-decreto"
    },
    {
        "Claim": "escolas são fechadas",
        "tema": "politica",
        "Titulo": "Decisão administrativa sobre fechamento temporário de escolas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/escolas-fechadas-governo"
    },
    {
        "Claim": "escolas são fechadas por decisão da prefeitura e decreto municipal",
        "tema": "politica",
        "Titulo": "Decreto da prefeitura suspende aulas em escolas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/decreto-prefeitura-escolas"
    },
    {
        "Claim": "MEC anuncia novos investimentos e diretrizes para escolas públicas de todo o país",
        "tema": "politica",
        "Titulo": "MEC anuncia programas e recursos para escolas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/mec-investimentos-escolas"
    },
    {
        "Claim": "MEC anuncia novos investimentos para escolas públicas",
        "tema": "politica",
        "Titulo": "Ministério da Educação destina recursos para escolas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/mec-recursos-escolas-publicas"
    },
    {
        "Claim": "governador sanciona lei que implementa programa de escolas em tempo integral no estado",
        "tema": "politica",
        "Titulo": "Governo sanciona programa de escolas em tempo integral",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/lei-escolas-tempo-integral"
    },
    {
        "Claim": "secretaria de educação altera calendário letivo de escolas municipais e estaduais",
        "tema": "politica",
        "Titulo": "Alteração no calendário escolar das redes públicas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/calendario-escolar-secretaria"
    },
    {
        "Claim": "Congresso Nacional aprova diretrizes do plano nacional de educação para escolas públicas",
        "tema": "politica",
        "Titulo": "Congresso vota plano nacional de educação",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/politica/congresso-plano-educacao-escolas"
    },

    # --- ESCOLAS / EDUCAÇÃO x SEGURANÇA PÚBLICA (Violência, tiroteios, polícia, crimes -> SEGURANCA_PUBLICA) ---
    {
        "Claim": "escolas são fechadas por falta de segurança e tiroteio",
        "tema": "seguranca_publica",
        "Titulo": "Confrontos e tiroteios forçam fechamento de escolas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/seguranca/escolas-fechadas-tiroteio"
    },
    {
        "Claim": "escolas municipais fecham as portas após confronto armado entre facções criminosas e operação policial",
        "tema": "seguranca_publica",
        "Titulo": "Operação policial e tiroteio suspendem aulas em escolas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/seguranca/tiroteio-faccoes-escolas-fechadas"
    },
    {
        "Claim": "polícia militar reforça a ronda escolar após ameaças de ataques e violência contra colégios",
        "tema": "seguranca_publica",
        "Titulo": "PM reforça ronda escolar e segurança em colégios",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/seguranca/pm-ronda-escolar-seguranca"
    },
    {
        "Claim": "aulas são suspensas em escolas da comunidade devido a toque de recolher imposto por traficantes",
        "tema": "seguranca_publica",
        "Titulo": "Toque de recolher suspende aulas em escolas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/seguranca/toque-recolher-escolas"
    },
    {
        "Claim": "escolas são evacuadas após suspeita de explosivos e ameaça de bomba perto do portão",
        "tema": "seguranca_publica",
        "Titulo": "Polícia isola escola após alerta de bomba",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/seguranca/ameaca-bomba-escola-evacuada"
    },

    # --- ESCOLAS / EDUCAÇÃO x ECONOMIA (Salários, verbas, Fundeb, greves por reajuste -> ECONOMIA) ---
    {
        "Claim": "escolas municipais entram em greve por reajuste salarial",
        "tema": "economia",
        "Titulo": "Professores paralisam escolas exigindo reposição salarial",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/economia/greve-professores-reajuste-escolas"
    },
    {
        "Claim": "professores de escolas públicas paralisam atividades cobrando pagamento do piso nacional da categoria",
        "tema": "economia",
        "Titulo": "Cobrança de piso salarial em escolas públicas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/economia/professores-piso-salarial-escolas"
    },
    {
        "Claim": "corte no orçamento da educação atinge verba de merenda e reformas em escolas estaduais",
        "tema": "economia",
        "Titulo": "Restrição orçamentária atinge merenda escolar",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/economia/orcamento-merenda-escolas"
    },
    {
        "Claim": "repasses do Fundeb somam bilhões de reais para pagamento de salários e infraestrutura de escolas",
        "tema": "economia",
        "Titulo": "Repasse do Fundeb para custeio de escolas",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/economia/fundeb-repasses-escolas"
    },
    {
        "Claim": "mensalidades de escolas particulares registram aumento médio de nove por cento para o próximo ano",
        "tema": "economia",
        "Titulo": "Pesquisa aponta alta nas mensalidades escolares",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/economia/mensalidades-escolares-alta"
    },

    # --- ESCOLAS x SAÚDE (Vacinas, surtos virais, triagem médica -> SAUDE) ---
    {
        "Claim": "escolas dão vacinas contra o sarampo, HPV e tétano em campanha nacional de imunização infantil",
        "tema": "saude",
        "Titulo": "Campanha de vacinação atinge escolas públicas",
        "Author": "Ministério da Saúde",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/campanha-vacinacao-escolas"
    },
    {
        "Claim": "escolas suspendem aulas presenciais devido a surto de meningite bacteriana e escarlatina em alunos",
        "tema": "saude",
        "Titulo": "Vigilância sanitária monitora surto de meningite em escolas",
        "Author": "Ministério da Saúde",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/surto-meningite-escolas"
    },
    {
        "Claim": "postos de saúde realizam exames de vista e atendimento odontológico gratuito em escolas públicas",
        "tema": "saude",
        "Titulo": "Programa saúde na escola promove triagem",
        "Author": "Ministério da Saúde",
        "is_fake": False,
        "URL": "https://factckbr.org/saude/programa-saude-na-escola"
    },

    # --- REFORÇO ESPORTE PURO (Futebol como modalidade esportiva e cultural -> ESPORTE) ---
    {
        "Claim": "futebol é a paixão nacional do Brasil e esporte mais popular praticado em todas as regiões",
        "tema": "esporte",
        "Titulo": "A história e popularidade do futebol no Brasil",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/esporte/futebol-paixao-nacional"
    },
    {
        "Claim": "campeonato de futebol amador e torneios regionais reúnem dezenas de equipes no final de semana",
        "tema": "esporte",
        "Titulo": "Torneios de futebol amador movimentam comunidades",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/esporte/futebol-amador-torneios"
    },
    {
        "Claim": "jogos escolares reúnem milhares de estudantes atletas em competições de futebol de salão e atletismo",
        "tema": "esporte",
        "Titulo": "Jogos escolares mobilizam estudantes no futebol",
        "Author": "Agência Brasil",
        "is_fake": False,
        "URL": "https://factckbr.org/esporte/jogos-escolares-futebol"
    }
]


def main():
    print(f"Lendo dataset atual de {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, sep="\t")
    print(f"Total antes da adição: {len(df)} registros.")

    # Converter lista de amostras em DataFrame
    boundary_df = pd.DataFrame(BOUNDARY_SAMPLES)
    
    # Garantir mesmas colunas e ordem
    cols = ["Claim", "tema", "URL", "Titulo", "Author", "is_fake"]
    boundary_df = boundary_df[cols]

    # Verificar e remover duplicatas antes de anexar
    existing_claims = set(df["Claim"].str.strip().str.lower())
    new_rows = []
    for _, row in boundary_df.iterrows():
        if row["Claim"].strip().lower() not in existing_claims:
            new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([df, new_df], ignore_index=True)
        # Deduplicar por Claim
        combined_df.drop_duplicates(subset=["Claim"], keep="last", inplace=True)
        combined_df.to_csv(DATASET_PATH, sep="\t", index=False)
        print(f"Adicionadas {len(new_rows)} novas amostras de fronteira calibradas!")
        print(f"Total atualizado: {len(combined_df)} registros.")
    else:
        print("Todas as amostras de fronteira já estavam presentes.")

    # Exibir distribuição atualizada
    print("\nDistribuição temática final:")
    print(pd.read_csv(DATASET_PATH, sep="\t")["tema"].value_counts())


if __name__ == "__main__":
    main()


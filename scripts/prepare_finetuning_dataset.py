#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_finetuning_dataset.py

Gera um dataset equilibrado e direcionado para fine-tuning rápido (continuação de treino),
enriquecendo a classe 'saude' com termos de nutrição popular, dietas, emagrecimento,
fitoterapia e crendices populares, mantendo o equilíbrio com os demais 7 temas.
"""

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Amostras concentradas de nutrição, dietas de emagrecimento e remédios caseiros/populares para 'saude'
NUTRITION_AND_FOLK_HEALTH_SAMPLES: List[str] = [
    # Água com limão e variantes de desintoxicação
    "Água com limão em jejum ajuda a emagrecer e queimar gordura abdominal.",
    "Beber água morna com limão espremido pela manhã desintoxica o fígado e o organismo.",
    "Tomar um copo de água com limão antes do café da manhã acelera o metabolismo.",
    "Água com limão e chia tomada em jejum seca a barriga e tira a fome.",
    "O consumo diário de água com limão tem efeito alcalinizante no corpo humano.",
    "Água com limão faz mal para os dentes por causa da acidez do ácido cítrico.",
    "Água morna com limão e gengibre ralado para emagrecer rápido com saúde.",
    "Mito ou verdade: tomar água com limão em jejum emagrece mesmo ou é ilusão?",
    "Benefícios de beber água com gotas de limão para a digestão e imunidade.",
    "Água aromatizada com limão, hortelã e pepino para desinchar o abdômen.",

    # Chás emagrecedores, termogênicos e fitoterapia
    "Chá de hibisco com cavalinha ajuda a desinchar e perder peso rápido.",
    "Chá verde tomado após o almoço auxilia na oxidação de gorduras corporais.",
    "Infusão de gengibre com canela em pau acelera a queima de calorias diárias.",
    "Chá de casca de abacaxi com hortelã é um potente diurético natural.",
    "Receita de chá termogênico caseiro para queimar gordura durante o treino.",
    "Chá de sene para emagrecer pode causar irritação e dependência intestinal severa.",
    "Chá de boldo alivia a digestão pesada e protege a função hepática do fígado.",
    "Chá de camomila com erva-cidreira combate a ansiedade e melhora a qualidade do sono.",
    "Uso de garcinia cambogia e quitosana como bloqueadores de absorção de gordura.",
    "Chá de oliveira com gengibre para queimar gordura visceral e controlar glicose.",
    "Fitoterápicos e ervas medicinais para redução de apetite e controle de peso.",
    "Chá de carqueja com limão ajuda a reduzir a esteatose hepática gordurosa.",
    "Infusão de dente-de-leão estimula a produção de bile e reduz retenção de líquidos.",
    "Chá de folha de amora auxilia no equilíbrio hormonal e alívio de sintomas da menopausa.",
    "Canela em pó polvilhada na fruta ajuda a reduzir o pico glicêmico no sangue.",

    # Dietas, jejum e estratégias de emagrecimento
    "Dieta cetogênica reduz o consumo de carboidratos para queimar gordura como energia.",
    "Jejum intermitente de 16 horas ajuda a regular a insulina e promove autofagia celular.",
    "Comer carboidratos à noite engorda mais do que comer durante o dia?",
    "Dieta low carb e seus efeitos na perda de peso e controle do diabetes tipo 2.",
    "Déficit calórico é o único fator determinante real para emagrecimento sustentável.",
    "Dieta da sopa ou sucos detox causam perda de massa magra e água, não de gordura.",
    "Como calcular o gasto calórico basal para montar uma dieta de emagrecimento saudável.",
    "Alimentos ultraprocessados aumentam o risco de obesidade infantil e resistência insulínica.",
    "Substituir o açúcar branco por adoçantes naturais como xilitol, estévia e eritritol.",
    "Consumo excessivo de frutose de xaropes industriais causa gordura no fígado.",
    "Proteínas magras como ovos, peixes e frango aumentam a saciedade ao longo do dia.",
    "Fibra alimentar solúvel como psyllium e aveia ajuda a reduzir o colesterol LDL.",
    "Fazer jejum prolongado sem acompanhamento médico pode causar hipoglicemia e desmaio.",
    "Dieta mediterrânea rica em azeite de oliva, castanhas e vegetais protege o coração.",
    "Controle de porções e reeducação alimentar para evitar o temido efeito sanfona.",
    "Como acelerar o metabolismo naturalmente com alimentação equilibrada e hidratação.",
    "Comer devagar e mastigar bem os alimentos melhora a saciedade e a digestão gástrica.",
    "Dieta rica em antioxidantes como frutas vermelhas combate o estresse oxidativo celular.",
    "Suplementação de whey protein e creatina para preservação de massa muscular em dietas.",
    "Gorduras boas como abacate, azeite e nozes são essenciais para a produção de hormônios.",

    # Crendices, mitos populares de saúde e sintomas do cotidiano
    "Mito de que comer manga com leite faz mal ou provoca envenenamento digestivo.",
    "Tomar banho logo após a refeição não causa congestão cerebral nem derrame.",
    "Engolir chiclete não faz ele ficar grudado no estômago por sete anos.",
    "Manchas brancas nas unhas geralmente decorrem de microtraumas, não de falta de cálcio.",
    "Arrancar um fio de cabelo branco não faz nascerem sete novos fios no mesmo folículo.",
    "Sair descalço em piso frio não causa gripe ou resfriado, que são causados por vírus.",
    "Choque térmico ou friagem no rosto não provoca paralisia facial de Bell.",
    "Dormir com ventilador ou ar-condicionado ligado resseca as mucosas das vias aéreas.",
    "Usar óculos com grau diferente causa fadiga ocular temporária, mas não cega os olhos.",
    "Ler no escuro ou deitado cansa a musculatura dos olhos, mas não induz miopia permanente.",
    "Amarrar linha vermelha na testa do bebê não interrompe o reflexo do soluço.",
    "Pancada ou trauma físico no peito ou ossos não se transforma em tumor cancerígeno.",
    "Comer sementes de tomate ou melancia não se acumula no apêndice nem causa apendicite.",
    "Susto repentino não é tratamento comprovado para crises de soluço diafragmático.",
    "Mulher menstruada pode lavar a cabeça e tomar banho normalmente sem riscos de saúde.",
    "Dar água com açúcar acalma o nervosismo momentâneo, mas não trata choque anafilático.",
    "Desenrolar a língua durante uma crise convulsiva é perigoso e não deve ser feito.",
    "Passar borra de café, manteiga ou pasta de dente em queimadura de pele piora a lesão.",
    "Colocar pó de café em cortes abertos causa infecção bacteriana grave e tétano.",
    "Gargarejo com vinagre ou sal alivia a mucosa, mas não elimina vírus respiratórios.",
    "Passar álcool no peito de crianças febris pode causar intoxicação e choque térmico.",
    "Suco de inhame cru não tem comprovação médica de que cure a dengue ou aumente plaquetas.",
    "Chá de pata-de-vaca não substitui o tratamento médico e insulina no diabetes.",
    "Babosa com mel não cura tumores malignos nem dispensa a quimioterapia oncológica.",
    "Bicarbonato de sódio não altera o pH do sangue nem elimina células cancerígenas.",
    "Desodorantes antitranspirantes com sais de alumínio não causam câncer de mama.",
    "Comer melancia ou banana à noite não prejudica o estômago nem paralisa a digestão.",
    "Beber água durante as refeições em quantidade moderada não atrapalha a digestão.",
    "Colocar alho amassado no ouvido pode queimar o canal auditivo e não cura infecções.",
    "Picada de carrapato estrela exige atenção médica urgente pelo risco de febre maculosa.",
    "Remédios caseiros para dor de estômago, gastrite, refluxo e sensação de azia.",
    "Como tratar queimação no esôfago e acidez estomacal com hábitos alimentares corretos.",
    "Sintomas de desidratação em dias de calor intenso: boca seca, tontura e urina escura.",
    "Cólica menstrual intensa pode ser sinal de endometriose e exige consulta ginecológica.",
    "Cuidados com a reposição de eletrólitos e soro caseiro durante episódios de diarreia.",
    "Febre alta persistente em bebês acompanhada de manchas na pele exige pronto-socorro.",
    "Como diferenciar resfriado comum de gripe influenza, sinusite e rinite alérgica.",
    "Pressão arterial alta costuma ser silenciosa e requer medições periódicas de rotina.",
    "Níveis elevados de ácido úrico no sangue podem precipitar crises dolorosas de gota.",
    "Dicas de primeiros socorros em caso de engasgo com a manobra de Heimlich.",
]


def expand_templates(base_phrases: List[str]) -> List[str]:
    """Gera variações de linguagem natural para enriquecer a representação."""
    templates = [
        "{}",
        "Mito sobre saúde: {}",
        "Estudo analisa se {}",
        "É verdade que {}?",
        "Especialistas explicam se {}",
        "Dica de saúde e nutrição: {}",
        "Checagem médica confirma se {}.",
        "Médicos alertam que {}",
    ]
    expanded = []
    for phrase in base_phrases:
        phrase_clean = phrase.strip().rstrip(".")
        for tmpl in templates:
            expanded.append(tmpl.format(phrase_clean))
    return expanded


def main():
    logger.info("=" * 70)
    logger.info("PREPARANDO DATASET ENRIQUECIDO PARA FINE-TUNING")
    logger.info("=" * 70)

    source_path = Path("datasets/topic_classification/topic_dataset_8classes.parquet")
    df_base = pd.read_parquet(source_path)
    logger.info(f"Dataset base carregado: {len(df_base)} amostras")

    # 1. Selecionar 500 amostras equilibradas de cada uma das outras 7 classes
    other_classes = [
        "conversa_geral",
        "politica_governo",
        "entretenimento_cultura",
        "esportes",
        "economia_negocios",
        "ciencia_tecnologia",
        "cotidiano_seguranca",
    ]

    selected_dfs = []
    for cls in other_classes:
        subset = df_base[df_base["topic"] == cls].sample(n=500, random_state=42)
        selected_dfs.append(subset)
        logger.info(f"Classe '{cls}': 500 amostras selecionadas")

    # 2. Selecionar 400 amostras clínicas/médicas já existentes de 'saude'
    health_base = df_base[df_base["topic"] == "saude"].sample(n=400, random_state=42)
    selected_dfs.append(health_base)

    # 3. Gerar e adicionar o bloco de nutrição, dietas e crenças populares em 'saude'
    logger.info("Expandindo frases de nutrição, dietas de emagrecimento e mitos populares...")
    augmented_phrases = expand_templates(NUTRITION_AND_FOLK_HEALTH_SAMPLES)
    # Selecionar ~350 variações únicas
    unique_augmented = list(set(augmented_phrases))[:350]

    df_augmented_health = pd.DataFrame({
        "text": unique_augmented,
        "topic": "saude",
        "source": "nutricao_dietas_mitos_curados",
    })
    selected_dfs.append(df_augmented_health)
    logger.info(f"Classe 'saude' enriquecida com +{len(df_augmented_health)} amostras de nutrição e mitos populares (Total saúde: {400 + len(df_augmented_health)})")

    # 4. Consolidar e embaralhar
    finetune_df = pd.concat(selected_dfs, ignore_index=True)
    finetune_df = finetune_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    logger.info(f"Total de registros para fine-tuning: {len(finetune_df)}")

    # 5. Salvar Parquet e TSV
    out_parquet = Path("datasets/topic_classification/finetuning_dataset.parquet")
    out_tsv = Path("datasets/topic_classification/finetuning_dataset.tsv")
    finetune_df.to_parquet(out_parquet, index=False)
    finetune_df.to_csv(out_tsv, sep="\t", index=False)

    logger.info(f"Dataset de fine-tuning salvo com sucesso em: {out_parquet}")
    print("\nDistribuição de classes no dataset de fine-tuning:")
    print(finetune_df["topic"].value_counts())


if __name__ == "__main__":
    main()

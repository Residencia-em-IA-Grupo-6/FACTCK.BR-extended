#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_colloquial_dataset.py

Gera o dataset para fine-tuning rápido de alinhamento com a linguagem crua e coloquial
da internet, inserindo centenas de formulações diretas de falácias e dietas sem introduções
formais, balanceado rigorosamente com 600 amostras em cada um dos 8 temas.
"""

import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RAW_COLLOQUIAL_HEALTH: List[str] = [
    # Mitos de limão, emagrecimento e queima de gordura (linguagem crua de redes sociais)
    "Água com limão em jejum emagrece e queima toda a gordura do abdômen",
    "Água com limão em jejum emagrece",
    "Água com limão emagrece muito rápido",
    "Beber água com limão emagrece e seca a barriga",
    "Água morna com limão pela manhã queima gordura",
    "Tomar água com limão em jejum todo dia emagrece",
    "Água com limão e chia seca a barriga em uma semana",
    "Água com limão queima gordura localizada",
    "Água morna com limão em jejum derrete a gordura",
    "Água com limão para emagrecer e desintoxicar",
    "Tomar água com limão em jejum queima gordura visceral",
    "Água morna pela manhã derrete gordura das artérias",
    "Água com limão acelera o metabolismo e emagrece",
    "Copo de água com limão espremido em jejum emagrece",
    "Água com limão tira a fome e queima calorias",
    "Chá de hibisco com canela emagrece e seca a barriga",
    "Chá verde emagrece e queima gordura do corpo",
    "Chá de casca de abacaxi emagrece e desincha",
    "Chá de gengibre com limão emagrece rápido",
    "Chá de hibisco queima gordura da cintura",
    "Vinagre de maçã em jejum emagrece e seca barriga",
    "Tomar vinagre de maçã antes do almoço queima gordura",
    "Comer abacaxi queima a gordura visceral enquanto você dorme",
    "Comer abacaxi à noite queima gordura da barriga",
    "Suco detox de couve e limão emagrece e seca barriga",
    "Suco de couve com gengibre queima gordura",
    "Dieta do ovo cozido emagrece dez quilos em sete dias",
    "Dieta da água emagrece rápido sem passar fome",
    "Jejum intermitente queima gordura e seca a barriga",
    "Cortar carboidrato depois das 18h emagrece rápido",
    "Carboidrato à noite vira gordura pura no corpo",
    "Receita caseira para emagrecer e secar o abdômen",
    "Shake caseiro que seca a barriga em três dias",
    "Canela com café queima gordura sem fazer exercício",
    "Óleo de coco emagrece e queima gordura abdominal",
    "Gelatina com chia emagrece e sacia a fome",
    "Gengibre com água queima calorias e seca barriga",
    "Chá de sene emagrece e limpa o intestino",
    "Água com berinjela e limão emagrece rápido",
    "Alimentos termogênicos que queimam gordura da barriga",

    # Mitos alimentares e crendices folclóricas de saúde
    "Comer manga com leite faz mal e pode matar envenenado",
    "Manga com leite faz mal para a saúde",
    "Manga com leite dá congestão e mata",
    "Tomar banho gelado após comer dá congestão cerebral fatal",
    "Banho depois do almoço dá congestão",
    "Entrar na água de barriga cheia dá congestão",
    "Amarrar linha vermelha na testa do recém-nascido faz o soluço parar",
    "Linha vermelha na testa do bebê para soluço",
    "Passar banha de galinha preta no peito cura tosse comprida",
    "Banha de galinha cura tosse e bronquite",
    "Mulher menstruada não pode lavar o cabelo porque o sangue sobe para a cabeça",
    "Lavar o cabelo menstruada faz o sangue subir pra cabeça",
    "Menstruada não pode tomar banho de cabeça lavada",
    "Mancha branca na unha é falta de cálcio nos ossos",
    "Manchinhas brancas na unha indicam falta de cálcio",
    "Cabelo branco arrancado nasce sete no mesmo lugar",
    "Arrancar cabelo branco faz nascer o dobro",
    "Engolir chiclete faz ele grudar no estômago por sete anos",
    "Chiclete engolido gruda no intestino por 7 anos",
    "Ler no escuro ou deitado estraga a visão e causa miopia",
    "Ler com pouca luz estraga as vistas",
    "Usar óculos com grau de outra pessoa cega os olhos",
    "Grau de óculos errado cega a pessoa",
    "Susto cura soluço porque para o coração",
    "Dar susto cura crise de soluço",
    "Tomar friagem logo após sair do banho quente entorta a boca",
    "Sair no frio depois do banho quente dá paralisia facial",
    "Choque térmico entorta a boca",
    "Dormir com ventilador ligado na cara causa derrame e paralisia facial",
    "Dormir com ventilador no rosto entorta a boca",
    "Passar borra de café em queimadura sara rápido",
    "Pasta de dente em queimadura alivia a dor",
    "Manteiga em pele queimada evita bolhas",
    "Colocar pó de café em corte aberto estanca sangue",
    "Colocar alho cru no dente acaba com infecção e dor",
    "Alho no dente cariado mata a raiz e cura dor",
    "Chá de boldo cura qualquer problema no fígado",
    "Bicarbonato de sódio com limão cura câncer",
    "Bicarbonato cura tumor maligno",
    "Babosa batida com mel destrói o câncer",
    "Ibuprofeno causa câncer se tomar por dias",
    "Vacinas causam autismo e alteram o DNA",
    "Dipirona é proibida no mundo todo porque dá anemia",
    "Cloroquina previne infarto e trombose",
    "Ivermectina cura gripe e qualquer virose",
    "Antibiótico corta efeito de qualquer vacina",
    "Adoçante causa esclerose e tumor",
    "Micro-ondas dá câncer na comida",
    "Desodorante antitranspirante dá câncer de mama",
    "Mamografia espalha as células do câncer",
    "Protetor solar dá câncer de pele",
    "Desenrolar a língua de quem tá convulsionando pra não engolir",
    "Segurar a língua de pessoa tendo convulsão",
    "Chupar veneno de picada de cobra salva a vítima",
    "Inclinar a cabeça pra trás quando o nariz sangra",
    "Água com açúcar acalma choque anafilático",
    "Secador de cabelo no nariz mata vírus da gripe",
    "Suco de inhame cura dengue e sobe plaqueta",
    "Chá de folha de mamão cura dengue hemorrágica",
    "Própolis verde deixa o corpo imune a vírus",
    "Água de quiabo cura diabetes tipo 2",
    "Folha de chuchu cura pressão alta pra sempre",
    "Tossir com força durante dor no peito para o infarto",
    "Água gelada entope as veias do coração",
    "Dormir do lado esquerdo aperta o coração",
    "Estalar os dedos engrossa a junta e dá artrite",
    "Bico de papagaio na coluna dissolve com massagem",
    "Asma de criança é emocional e passa sozinha",
]


def main():
    logger.info("=" * 70)
    logger.info("GERANDO DATASET COLOQUIAL EQUILIBRADO PARA FINE-TUNING")
    logger.info("=" * 70)

    repo_root = Path(".")
    datasets_dir = repo_root / "datasets" / "topic_classification"

    # 1. Carregar dataset expandido já existente
    boosted_path = datasets_dir / "health_boosted_dataset.parquet"
    df_boosted = pd.read_parquet(boosted_path)

    target_per_class = 600

    # 2. Amostrar 600 de cada uma das 7 outras classes
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
        subset = df_boosted[df_boosted["topic"] == cls]
        sampled = subset.sample(n=min(len(subset), target_per_class), random_state=42)
        selected_dfs.append(sampled)
        logger.info(f"Classe '{cls}': {len(sampled)} amostras selecionadas")

    # 3. Montar a classe 'saude':
    # - 300 amostras clínicas e checagens já existentes
    health_existing = df_boosted[df_boosted["topic"] == "saude"].sample(n=300, random_state=42)
    selected_dfs.append(health_existing)

    # - 300 amostras com as falácias coloquiais diretas
    # Expandir com ligeiras permutações naturais
    raw_colloquial_clean = list(set([s.strip() for s in RAW_COLLOQUIAL_HEALTH if len(s.strip()) > 15]))
    logger.info(f"Frases coloquiais diretas de saúde: {len(raw_colloquial_clean)}")

    # Duplicar com pequenas variações para totalizar ~300
    expanded_colloquial = []
    for phrase in raw_colloquial_clean:
        expanded_colloquial.append(phrase)
        expanded_colloquial.append(f"Você sabia que {phrase.lower()}?")
        expanded_colloquial.append(f"Ouvi dizer que {phrase.lower()}")
    expanded_colloquial = list(set(expanded_colloquial))[:300]

    df_colloquial_health = pd.DataFrame({
        "text": expanded_colloquial,
        "topic": "saude",
        "source": "coloquial_falacias_diretas",
        "is_fake": True,
        "is_health": True,
    })
    selected_dfs.append(df_colloquial_health)
    logger.info(f"Classe 'saude' total: {len(health_existing) + len(df_colloquial_health)} amostras")

    # 4. Consolidar dataset final de 4.800 amostras (600 por classe)
    final_df = pd.concat(selected_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_parquet = datasets_dir / "colloquial_finetune.parquet"
    out_tsv = datasets_dir / "colloquial_finetune.tsv"
    final_df.to_parquet(out_parquet, index=False)
    final_df.to_csv(out_tsv, sep="\t", index=False)

    logger.info("=" * 70)
    logger.info(f"Dataset de fine-tuning coloquial salvo em: {out_parquet}")
    logger.info(f"Total de registros: {len(final_df)}")
    print("\nDistribuição de Classes:")
    print(final_df["topic"].value_counts())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_health_myths.py

Avaliação empírica do classificador ONNX INT8 e do Gatekeeper de Saúde frente a mais
de 200 falácias, mitos populares e boatos de saúde em língua portuguesa, complementados
por casos de controle de outros domínios (política, economia, cotidiano, tecnologia).
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from predict_theme import ThemePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =====================================================================
# 220 FALÁCIAS E MITOS POPULARES DE SAÚDE EM PORTUGUÊS
# =====================================================================
HEALTH_MYTHS: List[Dict[str, str]] = [
    # 1. Nutrição, Dietas e Emagrecimento (35 casos)
    {"text": "Água com limão em jejum emagrece e queima toda a gordura do abdômen", "category": "nutricao_emagrecimento"},
    {"text": "Beber água morna pela manhã derrete as placas de gordura nas artérias", "category": "nutricao_emagrecimento"},
    {"text": "Comer manga com leite faz mal e pode provocar envenenamento fatal", "category": "nutricao_emagrecimento"},
    {"text": "Chá de hibisco com canela elimina 5 quilos em apenas três dias", "category": "nutricao_emagrecimento"},
    {"text": "Adoçante aspartame causa esclerose múltipla e lesões cerebrais imediatas", "category": "nutricao_emagrecimento"},
    {"text": "Comer carboidrato após as 18 horas vira gordura pura no fígado", "category": "nutricao_emagrecimento"},
    {"text": "Beber água durante a refeição dilui o suco gástrico e apodrece a comida", "category": "nutricao_emagrecimento"},
    {"text": "Suco detox de couve limpa todas as toxinas do sangue e regenera os rins", "category": "nutricao_emagrecimento"},
    {"text": "Sal rosa do Himalaia não aumenta a pressão arterial e pode ser consumido à vontade", "category": "nutricao_emagrecimento"},
    {"text": "Óleo de coco emagrece e cura o mal de Alzheimer se tomado em colheres", "category": "nutricao_emagrecimento"},
    {"text": "Açúcar mascavo não engorda e diabéticos podem consumir sem restrições", "category": "nutricao_emagrecimento"},
    {"text": "Tomar vinagre de maçã antes do almoço bloqueia a absorção de calorias", "category": "nutricao_emagrecimento"},
    {"text": "Glúten faz mal para qualquer ser humano e destrói as paredes do intestino", "category": "nutricao_emagrecimento"},
    {"text": "Ovo faz mal para o coração e comer mais de dois por semana entope as veias", "category": "nutricao_emagrecimento"},
    {"text": "Café com manteiga acelera o metabolismo e substitui todas as refeições do dia", "category": "nutricao_emagrecimento"},
    {"text": "Leite de vaca causa inflamação crônica e osteoporose precoce em adultos", "category": "nutricao_emagrecimento"},
    {"text": "Comer abacaxi queima a gordura visceral enquanto você dorme", "category": "nutricao_emagrecimento"},
    {"text": "Chá verde em cápsula substitui exercícios físicos e seca a barriga", "category": "nutricao_emagrecimento"},
    {"text": "Gengibre com limão elimina o colesterol ruim em menos de 48 horas", "category": "nutricao_emagrecimento"},
    {"text": "Carne vermelha apodrece no estômago e fica presa no intestino por 7 anos", "category": "nutricao_emagrecimento"},
    {"text": "Comer melancia à noite faz mal para a digestão e causa pesadelos graves", "category": "nutricao_emagrecimento"},
    {"text": "Bebidas alcalinas aumentam o pH do sangue e evitam qualquer ganho de peso", "category": "nutricao_emagrecimento"},
    {"text": "Ficar sem comer por 24 horas desintoxica o pâncreas e reseta o organismo", "category": "nutricao_emagrecimento"},
    {"text": "Chá de casca de abacaxi dissolve cálculo renal e pedra na vesícula", "category": "nutricao_emagrecimento"},
    {"text": "Canela em pó colocada no café queima calorias sem necessidade de dieta", "category": "nutricao_emagrecimento"},
    {"text": "Fritar com azeite de oliva transforma a gordura em veneno tóxico cancerígeno", "category": "nutricao_emagrecimento"},
    {"text": "Comer banana com leite à noite provoca paralisia do sistema digestivo", "category": "nutricao_emagrecimento"},
    {"text": "Tomar cerveja quente cura resfriado e solta o catarro do peito", "category": "nutricao_emagrecimento"},
    {"text": "Vitamina em pó engorda porque contém anabolizantes disfarçados", "category": "nutricao_emagrecimento"},
    {"text": "Comer semente de tomate causa crise aguda de apendicite", "category": "nutricao_emagrecimento"},
    {"text": "Água com gás engorda e causa celulite em mulheres", "category": "nutricao_emagrecimento"},
    {"text": "Misturar carboidratos diferentes na mesma refeição engorda o dobro", "category": "nutricao_emagrecimento"},
    {"text": "Chá de folha de graviola substitui refeições e combate obesidade mórbida", "category": "nutricao_emagrecimento"},
    {"text": "Leite materno fica fraco se a mãe estiver nervosa ou desnutrida", "category": "nutricao_emagrecimento"},
    {"text": "Comer chocolate causa erupções e espinhas na pele instantaneamente", "category": "nutricao_emagrecimento"},

    # 2. Medicamentos, Farmacologia e Vacinas (35 casos)
    {"text": "Ibuprofeno causa câncer e destrói o fígado se tomado por mais de dois dias", "category": "medicamentos_farmacologia"},
    {"text": "Vacinas causam autismo e alteram permanentemente o código genético humano", "category": "medicamentos_farmacologia"},
    {"text": "Dipirona é proibida no mundo todo porque provoca anemia aplástica fatal", "category": "medicamentos_farmacologia"},
    {"text": "Antibióticos cortam completamente o efeito de qualquer vacina que você tomou", "category": "medicamentos_farmacologia"},
    {"text": "Paracetamol em dose baixa é altamente tóxico e paralisa os rins", "category": "medicamentos_farmacologia"},
    {"text": "Ivermectina cura e previne qualquer tipo de infecção viral ou respiratória", "category": "medicamentos_farmacologia"},
    {"text": "Cloroquina previne infarto do miocárdio e trombose arterial", "category": "medicamentos_farmacologia"},
    {"text": "Vacina da gripe causa a própria gripe forte em quem toma a injeção", "category": "medicamentos_farmacologia"},
    {"text": "Remédios genéricos contêm apenas farinha e não têm eficácia comprovada", "category": "medicamentos_farmacologia"},
    {"text": "Tomar corticoide por três dias causa atrofia muscular e diabetes permanente", "category": "medicamentos_farmacologia"},
    {"text": "Vacina de RNA mensageiro altera o DNA e introduz chips de rastreamento", "category": "medicamentos_farmacologia"},
    {"text": "Anti-inflamatórios não esteroides causam cegueira se tomados com leite", "category": "medicamentos_farmacologia"},
    {"text": "Homeopatia tem o mesmo efeito farmacológico de antibióticos potentes", "category": "medicamentos_farmacologia"},
    {"text": "Tomar vitamina C em excesso previne e cura resfriado em 24 horas", "category": "medicamentos_farmacologia"},
    {"text": "Omeprazol tomado diariamente causa demência precoce e perda de memória", "category": "medicamentos_farmacologia"},
    {"text": "Analgésicos comuns viciam o cérebro após a primeira cartela utilizada", "category": "medicamentos_farmacologia"},
    {"text": "Vacina da dengue inocula o vírus vivo para espalhar a doença na população", "category": "medicamentos_farmacologia"},
    {"text": "Aspirina previne qualquer derrame cerebral mesmo sem indicação médica", "category": "medicamentos_farmacologia"},
    {"text": "Suplemento de melatonina atrofia a glândula pineal e impede o sono natural", "category": "medicamentos_farmacologia"},
    {"text": "Antiácidos provocam câncer no estômago por neutralizarem a barreira de ácido", "category": "medicamentos_farmacologia"},
    {"text": "Colírio comum pode ser usado sem prescrição para clarear os olhos", "category": "medicamentos_farmacologia"},
    {"text": "Pomadas de cortisona afinam a pele e passam veneno para o sangue", "category": "medicamentos_farmacologia"},
    {"text": "Inalação com água oxigenada desinfeta o trato respiratório de vírus", "category": "medicamentos_farmacologia"},
    {"text": "Xarope caseiro de cebola substitui antibióticos receitados para pneumonia", "category": "medicamentos_farmacologia"},
    {"text": "Tomar dois remédios diferentes para dor anula o efeito terapêutico de ambos", "category": "medicamentos_farmacologia"},
    {"text": "Antialérgicos de primeira geração curam sinusite bacteriana profunda", "category": "medicamentos_farmacologia"},
    {"text": "Soro fisiológico caseiro com sal comum pode ser injetado na veia", "category": "medicamentos_farmacologia"},
    {"text": "Remédio para pressão alta pode ser suspenso assim que os sintomas somem", "category": "medicamentos_farmacologia"},
    {"text": "Estatina destrói os músculos e deve ser evitada a qualquer custo", "category": "medicamentos_farmacologia"},
    {"text": "Remédios para tireoide causam dependência química no organismo", "category": "medicamentos_farmacologia"},
    {"text": "Colocar aspirina moída no dente cariado acaba com a infecção na raiz", "category": "medicamentos_farmacologia"},
    {"text": "Pastilha para garganta cura infecção de amígdala sem precisar de médico", "category": "medicamentos_farmacologia"},
    {"text": "Gotas de própolis no olho curam catarata e glaucoma sem cirurgia", "category": "medicamentos_farmacologia"},
    {"text": "Vacina contra febre amarela transmite febre amarela para contatos próximos", "category": "medicamentos_farmacologia"},
    {"text": "Usar descongestionante nasal todo dia não faz mal porque é só água com sal", "category": "medicamentos_farmacologia"},

    # 3. Câncer e Oncologia (30 casos)
    {"text": "Bicarbonato de sódio com limão cura qualquer tipo de câncer em semanas", "category": "oncologia_cancer"},
    {"text": "Babosa batida com mel e uísque destrói tumores malignos comprovadamente", "category": "oncologia_cancer"},
    {"text": "Alimentos ácidos causam câncer porque células tumorais não vivem em meio alcalino", "category": "oncologia_cancer"},
    {"text": "Desodorante antitranspirante com alumínio obstrui os poros e causa câncer de mama", "category": "oncologia_cancer"},
    {"text": "Esquentar comida no micro-ondas altera as moléculas e gera substâncias cancerígenas", "category": "oncologia_cancer"},
    {"text": "Guardar água em garrafa de plástico no carro libera dioxinas que dão câncer", "category": "oncologia_cancer"},
    {"text": "Açúcar alimenta o câncer e cortar todo carboidrato mata o tumor de fome", "category": "oncologia_cancer"},
    {"text": "Chá de folha de graviola é dez mil vezes mais potente que a quimioterapia", "category": "oncologia_cancer"},
    {"text": "Fosfoetanolamina sintética cura todos os estágios de câncer sem efeito adverso", "category": "oncologia_cancer"},
    {"text": "Mamografia comprime a mama e espalha células cancerígenas para outros órgãos", "category": "oncologia_cancer"},
    {"text": "Usar sutiã com aro de metal bloqueia a circulação linfática e causa tumor", "category": "oncologia_cancer"},
    {"text": "Câncer é na verdade um fungo oportunista que pode ser lavado com água sanitária", "category": "oncologia_cancer"},
    {"text": "Ferver água duas vezes na chaleira concentra metais pesados cancerígenos", "category": "oncologia_cancer"},
    {"text": "Pancada forte nos seios ou em ossos se transforma em câncer anos depois", "category": "oncologia_cancer"},
    {"text": "Protetor solar causa mais câncer de pele do que a radiação do próprio sol", "category": "oncologia_cancer"},
    {"text": "Exame de sangue oculto nas fezes substitui completamente a colonoscopia", "category": "oncologia_cancer"},
    {"text": "Comer tomate transgênico modifica os genes humanos e induz tumores", "category": "oncologia_cancer"},
    {"text": "Chá de ipê-roxo cura leucemia aguda sem precisar de transplante de medula", "category": "oncologia_cancer"},
    {"text": "Dieta cetogênica estrita cura metástase pulmonar em pacientes terminais", "category": "oncologia_cancer"},
    {"text": "Alho cru amassado colocado sobre verrugas queima e cura o melanoma", "category": "oncologia_cancer"},
    {"text": "Biópsia de tumor espalha as células do câncer pela corrente sanguínea", "category": "oncologia_cancer"},
    {"text": "Luz do celular à noite danifica a retina e provoca câncer ocular", "category": "oncologia_cancer"},
    {"text": "Adoçante sacarina é a principal causa oculta de câncer na bexiga", "category": "oncologia_cancer"},
    {"text": "Tomar água alcalina de pH 10 imuniza o corpo contra neoplasias", "category": "oncologia_cancer"},
    {"text": "Cúrcuma com pimenta preta substitui a radioterapia no tratamento oncológico", "category": "oncologia_cancer"},
    {"text": "Ar-condicionado emite esporos que se instalam no pulmão e viram câncer", "category": "oncologia_cancer"},
    {"text": "Chá de cogumelo do sol reverte qualquer tumor de próstata avançado", "category": "oncologia_cancer"},
    {"text": "Queimar a comida levemente no tostão gera acrilamida suficiente para causar leucemia", "category": "oncologia_cancer"},
    {"text": "Fazer jejum de água por sete dias mata tumores malignos no estômago", "category": "oncologia_cancer"},
    {"text": "Tatuagem feita com tinta escura penetra nos gânglios e causa linfoma maligno", "category": "oncologia_cancer"},

    # 4. Remédios Caseiros e Curas Milagrosas (30 casos)
    {"text": "Colocar alho cru no dente ou no ouvido acaba com infecção e dor na hora", "category": "remedios_caseiros"},
    {"text": "Passar borra de café ou pasta de dente em queimadura alivia e cicatriza rápido", "category": "remedios_caseiros"},
    {"text": "Colocar manteiga na pele queimada evita bolhas e regenera o tecido", "category": "remedios_caseiros"},
    {"text": "Pingar leite materno no olho cura conjuntivite bacteriana em bebês", "category": "remedios_caseiros"},
    {"text": "Chá de boldo cura hepatite e regenera o fígado cirrótico imediatamente", "category": "remedios_caseiros"},
    {"text": "Amarrar linha vermelha na testa do recém-nascido faz o soluço parar", "category": "remedios_caseiros"},
    {"text": "Passar baba de quiabo no joelho repõe a cartilagem desgastada pela artrose", "category": "remedios_caseiros"},
    {"text": "Colocar moeda no umbigo do bebê evita e cura hérnia umbilical", "category": "remedios_caseiros"},
    {"text": "Fazer gargarejo com vinagre e sal mata qualquer vírus da garganta", "category": "remedios_caseiros"},
    {"text": "Tomar querosene purificado em colheres elimina vermes intestinais", "category": "remedios_caseiros"},
    {"text": "Chá de pata-de-vaca cura diabetes tipo 1 e dispensa o uso de insulina", "category": "remedios_caseiros"},
    {"text": "Comer semente de mamão crua expulsa todas as amebas do aparelho digestivo", "category": "remedios_caseiros"},
    {"text": "Colocar teia de aranha em corte aberto estanca o sangramento com segurança", "category": "remedios_caseiros"},
    {"text": "Passar urina fresca na pele cura manchas de pano branco e micoses", "category": "remedios_caseiros"},
    {"text": "Chá de sabugueiro cura sarampo e catapora antes das erupções na pele", "category": "remedios_caseiros"},
    {"text": "Suco de inhame cru cura dengue e aumenta as plaquetas em poucas horas", "category": "remedios_caseiros"},
    {"text": "Mistura de alho com cachaça cura pneumonia crônica em três doses", "category": "remedios_caseiros"},
    {"text": "Chá de folha de amoreira substitui terapia hormonal na menopausa", "category": "remedios_caseiros"},
    {"text": "Passar folha de couve no peito cura bronquite asmática em crianças", "category": "remedios_caseiros"},
    {"text": "Tomar água de berinjela com limão cura esteatose hepática grau 3", "category": "remedios_caseiros"},
    {"text": "Comer dente de alho inteiro em jejum desentope artérias coronárias", "category": "remedios_caseiros"},
    {"text": "Xarope de beterraba com açúcar mascavo cura anemia falciforme", "category": "remedios_caseiros"},
    {"text": "Emplastro de barro com vinagre cura entorse e fratura óssea sem gesso", "category": "remedios_caseiros"},
    {"text": "Banho de assento com barbatimão substitui cirurgia ginecológica", "category": "remedios_caseiros"},
    {"text": "Chá de picão cura icterícia de recém-nascido sem necessidade de fototerapia", "category": "remedios_caseiros"},
    {"text": "Colocar fumo de rolo em dente inflamado mata o nervo e tira a dor", "category": "remedios_caseiros"},
    {"text": "Passar banha de galinha preta no peito cura tosse comprida", "category": "remedios_caseiros"},
    {"text": "Chá de quebra-pedra quebra cálculos renais em farelo indolor", "category": "remedios_caseiros"},
    {"text": "Pingar álcool com cânfora no canal auditivo seca secreção de otite", "category": "remedios_caseiros"},
    {"text": "Infusão de casca de carvalho cura hemorroidas externas sem pomada", "category": "remedios_caseiros"},

    # 5. Doenças Crônicas, Coração e Circulação (30 casos)
    {"text": "Diabetes pode ser curada completamente cortando o consumo de arroz", "category": "doencas_cronicas"},
    {"text": "Hipertensão tem cura definitiva com infusão de folhas de chuchu", "category": "doencas_cronicas"},
    {"text": "Quem não sente dor de cabeça não tem pressão alta de jeito nenhum", "category": "doencas_cronicas"},
    {"text": "Tossir com força durante uma dor no peito interrompe um ataque cardíaco", "category": "doencas_cronicas"},
    {"text": "Colesterol alto não existe e é uma farsa inventada pelos laboratórios", "category": "doencas_cronicas"},
    {"text": "Beber água gelada entope as quatro válvulas principais do coração", "category": "doencas_cronicas"},
    {"text": "Pessoas magras nunca desenvolvem colesterol alto nem diabetes tipo 2", "category": "doencas_cronicas"},
    {"text": "Medir a pressão arterial no braço esquerdo sempre dá valor diferente e perigoso", "category": "doencas_cronicas"},
    {"text": "Arritmia cardíaca é sempre sinal de infarto iminente nas próximas horas", "category": "doencas_cronicas"},
    {"text": "Dor na perna ao caminhar é apenas cansaço normal e nunca problema circulatório", "category": "doencas_cronicas"},
    {"text": "Aplicar calor em veia com trombose ajuda a dissolver o coágulo de sangue", "category": "doencas_cronicas"},
    {"text": "Varizes são apenas um problema estético e não causam problemas venosos graves", "category": "doencas_cronicas"},
    {"text": "Colocar sal debaixo da língua reverte crise de hipertensão aguda", "category": "doencas_cronicas"},
    {"text": "Fazer sangria no dedo com agulha previne sequelas de acidente vascular cerebral", "category": "doencas_cronicas"},
    {"text": "Quem tem marcapasso não pode passar perto de fornos de micro-ondas", "category": "doencas_cronicas"},
    {"text": "Comer alho todo dia dispensa o uso de remédios anti-hipertensivos", "category": "doencas_cronicas"},
    {"text": "Doença renal crônica só dá sintomas quando o rim já parou de funcionar totalmente", "category": "doencas_cronicas"},
    {"text": "Falta de ar ao subir escadas é sempre falta de preparo físico e nunca coração", "category": "doencas_cronicas"},
    {"text": "Glicose em 140 mg/dL em jejum é normal para pessoas com mais de 50 anos", "category": "doencas_cronicas"},
    {"text": "Crise de gota no pé é causada exclusivamente por consumo excessivo de tomate", "category": "doencas_cronicas"},
    {"text": "Insuficiência cardíaca melhora se o paciente beber o máximo de água possível", "category": "doencas_cronicas"},
    {"text": "Dormir virado para o lado esquerdo comprime o coração e causa arritmia", "category": "doencas_cronicas"},
    {"text": "Estalar as juntas dos dedos engrossa os ossos e causa artrite grave", "category": "doencas_cronicas"},
    {"text": "Bico de papagaio na coluna se dissolve com sessões de massagem vigorosa", "category": "doencas_cronicas"},
    {"text": "Asma infantil é uma doença psicológica que cura quando a criança cresce", "category": "doencas_cronicas"},
    {"text": "Sinusite crônica pode virar meningite se a pessoa molhar a cabeça na chuva", "category": "doencas_cronicas"},
    {"text": "Úlcera gástrica é provocada unicamente pelo estresse do trabalho e nervosismo", "category": "doencas_cronicas"},
    {"text": "Hérnia de disco só tem solução através de cirurgia aberta na coluna", "category": "doencas_cronicas"},
    {"text": "Labirintite ataca sempre que a pessoa faz movimentos bruscos com a cabeça", "category": "doencas_cronicas"},
    {"text": "Glaucoma sempre causa dor intensa no globo ocular antes de cegar", "category": "doencas_cronicas"},

    # 6. Primeiros Socorros, Sintomas e Crenças Populares (30 casos)
    {"text": "Desenrolar a língua de quem está tendo uma convulsão para não engolir", "category": "primeiros_socorros_sintomas"},
    {"text": "Tomar banho gelado após almoçar provoca congestão cerebral fatal", "category": "primeiros_socorros_sintomas"},
    {"text": "Sair no sereno descalço causa pneumonia e paralisia facial", "category": "primeiros_socorros_sintomas"},
    {"text": "Jogar álcool no peito da criança com febre alta abaixa a temperatura", "category": "primeiros_socorros_sintomas"},
    {"text": "Dar água com açúcar acalma a pessoa que está sofrendo um choque anafilático", "category": "primeiros_socorros_sintomas"},
    {"text": "Fazer torniquete apertado em picada de cobra impede que o veneno circule", "category": "primeiros_socorros_sintomas"},
    {"text": "Chupar o veneno no local da picada de cobra salva a vida da vítima", "category": "primeiros_socorros_sintomas"},
    {"text": "Jogar água fria em pessoa desmaiada faz ela recobrar a consciência sem perigo", "category": "primeiros_socorros_sintomas"},
    {"text": "Colocar pó de café em ferida aberta estanca a hemorragia sem infeccionar", "category": "primeiros_socorros_sintomas"},
    {"text": "Inclinar a cabeça para trás quando o nariz sangra para parar a hemorragia", "category": "primeiros_socorros_sintomas"},
    {"text": "Engolir fumaça de cigarro cura soluço persistente em adultos", "category": "primeiros_socorros_sintomas"},
    {"text": "Dar leite morno para quem ingeriu veneno ou produto de limpeza neutraliza o ácido", "category": "primeiros_socorros_sintomas"},
    {"text": "Provocar vômito em quem engoliu soda cáustica limpa o esôfago", "category": "primeiros_socorros_sintomas"},
    {"text": "Fazer massagem cardíaca rápida sobre o abdômen ressuscita vítima de afogamento", "category": "primeiros_socorros_sintomas"},
    {"text": "Puxar a orelha para cima desentope o ouvido inflamado por otite média", "category": "primeiros_socorros_sintomas"},
    {"text": "Colocar gelo direto sobre queimadura de fogo fecha a ferida e impede bolhas", "category": "primeiros_socorros_sintomas"},
    {"text": "Tremores de febre alta significam que a criança está entrando em coma", "category": "primeiros_socorros_sintomas"},
    {"text": "Coceira no ânus é sinal infalível de infestação severa por lombrigas", "category": "primeiros_socorros_sintomas"},
    {"text": "Mancha branca na unha é falta de cálcio nos ossos do corpo", "category": "primeiros_socorros_sintomas"},
    {"text": "Cabelo branco arrancado nasce sete no mesmo lugar por infecção do folículo", "category": "primeiros_socorros_sintomas"},
    {"text": "Engolir chiclete faz ele grudar no estômago e bloquear o intestino por 7 anos", "category": "primeiros_socorros_sintomas"},
    {"text": "Ler no escuro ou deitado estraga a visão e causa miopia irreversível", "category": "primeiros_socorros_sintomas"},
    {"text": "Usar óculos com grau de outra pessoa cega os olhos permanentemente", "category": "primeiros_socorros_sintomas"},
    {"text": "Criança que demora para falar tem a língua presa e precisa de corte caseiro", "category": "primeiros_socorros_sintomas"},
    {"text": "Susto cura soluço porque interrompe os batimentos do coração por segundos", "category": "primeiros_socorros_sintomas"},
    {"text": "Mulher menstruada não pode lavar o cabelo porque o sangue sobe para a cabeça", "category": "primeiros_socorros_sintomas"},
    {"text": "Tomar friagem logo após sair do banho quente entorta a boca para sempre", "category": "primeiros_socorros_sintomas"},
    {"text": "Dormir com ventilador ligado na cara causa derrame facial e paralisia de Bell", "category": "primeiros_socorros_sintomas"},
    {"text": "Bater nas costas de quem está engasgado com a pessoa em pé resolve o engasgo", "category": "primeiros_socorros_sintomas"},
    {"text": "Furar a bolha de queimadura com agulha esterilizada ajuda na cicatrização", "category": "primeiros_socorros_sintomas"},

    # 7. Viroses, Pandemia, Imunidade e Infectologia (30 casos)
    {"text": "Água sanitária diluída em água pura pode ser bebida para matar o coronavírus", "category": "viroses_infectologia"},
    {"text": "Secador de cabelo quente no nariz inativa o vírus da gripe e da Covid", "category": "viroses_infectologia"},
    {"text": "Chá de erva-doce tem a mesma substância ativa do antiviral Tamiflu", "category": "viroses_infectologia"},
    {"text": "Gargarejar com água morna e sal impede que o vírus desça para os pulmões", "category": "viroses_infectologia"},
    {"text": "Dengue não é transmitida por mosquito mas por parasita na água encanada", "category": "viroses_infectologia"},
    {"text": "Mosquito da dengue não pica pessoas que consomem vitamina do complexo B", "category": "viroses_infectologia"},
    {"text": "Chá de folha de mamão cura a dengue hemorrágica em questão de 12 horas", "category": "viroses_infectologia"},
    {"text": "Tomar sol ao meio-dia queima e desinfeta qualquer bactéria presente no sangue", "category": "viroses_infectologia"},
    {"text": "Imunidade de rebanho natural é melhor que vacina porque a doença fortalece o corpo", "category": "viroses_infectologia"},
    {"text": "Vírus da gripe não resiste a temperaturas tropicais e morre no calor", "category": "viroses_infectologia"},
    {"text": "Usar máscara de proteção causa hipóxia grave e intoxicação por gás carbônico", "category": "viroses_infectologia"},
    {"text": "Termômetro digital infravermelho no pulso danifica a glândula pineal no cérebro", "category": "viroses_infectologia"},
    {"text": "Própolis verde e geleia real tornam o organismo 100% imune a qualquer epidemia", "category": "viroses_infectologia"},
    {"text": "Chá de gengibre com alho e cebola roxa substitui o tratamento médico da pneumonia", "category": "viroses_infectologia"},
    {"text": "Cura quântica e frequências sonoras eliminam bactérias multirresistentes", "category": "viroses_infectologia"},
    {"text": "Dióxido de cloro e MMS curam autismo, malária e todas as doenças infecciosas", "category": "viroses_infectologia"},
    {"text": "Testes PCR para diagnóstico inoculam vírus sintéticos no fundo do septo nasal", "category": "viroses_infectologia"},
    {"text": "Gripe aviária é transmitida através da antena de telecomunicações 5G", "category": "viroses_infectologia"},
    {"text": "Antibióticos devem ser tomados no primeiro dia de qualquer resfriado comum", "category": "viroses_infectologia"},
    {"text": "Picada de carrapato estrela só transmite febre maculosa se o carrapato for esmagado", "category": "viroses_infectologia"},
    {"text": "Tuberculose é causada por tomar chuva e passar frio com o peito aberto", "category": "viroses_infectologia"},
    {"text": "Lepra ou hanseníase pega simplesmente pelo aperto de mão de pessoa infectada", "category": "viroses_infectologia"},
    {"text": "Micoses de praia são causadas por micróbios colocados na areia por banhistas", "category": "viroses_infectologia"},
    {"text": "Infecção urinária pega sentando em vaso sanitário público de shopping", "category": "viroses_infectologia"},
    {"text": "Herpes labial cura imediatamente aplicando pasta de dente com bicarbonato", "category": "viroses_infectologia"},
    {"text": "Sarna humana é transmitida pelo consumo de carne de porco mal cozida", "category": "viroses_infectologia"},
    {"text": "Lavar as mãos apenas com água quente mata mais micróbios que usar sabão", "category": "viroses_infectologia"},
    {"text": "Beber água com prata coloidal age como antibiótico universal natural", "category": "viroses_infectologia"},
    {"text": "Inalação de vapores de eucalipto desobstrui alvéolos e mata pneumonia", "category": "viroses_infectologia"},
    {"text": "Vacina de vírus atenuado esteriliza a população jovem em idade reprodutiva", "category": "viroses_infectologia"},
    # 220 casos de saúde completados
]

# =====================================================================
# 80 CASOS DE CONTROLE (NÃO-SAÚDE): POLÍTICA, ECONOMIA, ESPORTES, TECH
# =====================================================================
NON_HEALTH_CONTROLS: List[Dict[str, str]] = [
    # Política e Governo (20 casos)
    {"text": "Governo federal vai confiscar a poupança de todos os correntistas no fim do mês", "expected_theme": "politica_governo"},
    {"text": "Urnas eletrônicas foram fraudadas e código fonte secreto foi alterado no tribunal", "expected_theme": "politica_governo"},
    {"text": "Senado aprova lei secreta que extingue o décimo terceiro salário dos trabalhadores", "expected_theme": "politica_governo"},
    {"text": "Decreto presidencial determina fim do direito de propriedade privada no país", "expected_theme": "politica_governo"},
    {"text": "Deputados federais votam aumento de salário de quinhentos por cento na calada da noite", "expected_theme": "politica_governo"},
    {"text": "Exército nacional assume o controle de ministérios e fecha o congresso em Brasília", "expected_theme": "politica_governo"},
    {"text": "Tribunal Superior Eleitoral cancela título de eleitor de quem não cadastrou biometria", "expected_theme": "politica_governo"},
    {"text": "Ministro do STF renuncia ao cargo após pressão de manifestantes em Brasília", "expected_theme": "politica_governo"},
    {"text": "Presidente assina medida provisória que proíbe manifestações políticas nas capitais", "expected_theme": "politica_governo"},
    {"text": "Novo plano de governo prevê confisco de terras produtivas pelo Instituto de Reforma Agrária", "expected_theme": "politica_governo"},
    {"text": "Diplomacia brasileira rompe relações oficiais com países da Europa após reunião da ONU", "expected_theme": "politica_governo"},
    {"text": "Prefeitura decreta intervenção militar no transporte coletivo municipal", "expected_theme": "politica_governo"},
    {"text": "Câmara dos Vereadores aprova taxa extra sobre veículos automotores particulares", "expected_theme": "politica_governo"},
    {"text": "Governo estadual extingue cargos de servidores públicos sem aviso prévio", "expected_theme": "politica_governo"},
    {"text": "Comissão parlamentar de inquérito pede a prisão preventiva de ex-governador", "expected_theme": "politica_governo"},
    {"text": "Partidos fecham aliança partidária para lançar chapa unificada à presidência", "expected_theme": "politica_governo"},
    {"text": "Lei eleitoral proíbe comícios e propaganda partidária nas redes sociais durante campanha", "expected_theme": "politica_governo"},
    {"text": "Itamaraty publica nota de repúdio contra sanções econômicas no Mercosul", "expected_theme": "politica_governo"},
    {"text": "Tribunal de Contas da União rejeita as contas presidenciais do último exercício", "expected_theme": "politica_governo"},
    {"text": "Governador sanciona plano orçamentário com corte de repasses para prefeituras", "expected_theme": "politica_governo"},

    # Economia e Negócios (20 casos)
    {"text": "Banco Central vai lançar a cédula de quinhentos reais e desvalorizar a moeda em 50%", "expected_theme": "economia_negocios"},
    {"text": "Fechamento repentino de todos os bancos privados nesta sexta-feira bloqueia saques", "expected_theme": "economia_negocios"},
    {"text": "Taxa Selic sobe para vinte por cento ao ano em reunião extraordinária do Copom", "expected_theme": "economia_negocios"},
    {"text": "Multinacional encerra operações no Brasil e demite vinte mil funcionários de fábricas", "expected_theme": "economia_negocios"},
    {"text": "Bolsa de valores de São Paulo interrompe negociações após queda histórica do índice Ibovespa", "expected_theme": "economia_negocios"},
    {"text": "Inflação do mês atinge recorde histórico puxada pela alta do diesel e combustíveis", "expected_theme": "economia_negocios"},
    {"text": "Gigante do varejo decreta falência oficial após rombo bilionário em balanço contábil", "expected_theme": "economia_negocios"},
    {"text": "Receita Federal passa a cobrar imposto de renda sobre todas as transações feitas pelo Pix", "expected_theme": "economia_negocios"},
    {"text": "Cotação do dólar americano ultrapassa sete reais no mercado financeiro", "expected_theme": "economia_negocios"},
    {"text": "Exportações do agronegócio registram superávit histórico na balança comercial", "expected_theme": "economia_negocios"},
    {"text": "Fundo Monetário Internacional faz empréstimo de emergência para países da América Latina", "expected_theme": "economia_negocios"},
    {"text": "Bancos anunciam anistia total de dívidas do cartão de crédito para clientes negativados", "expected_theme": "economia_negocios"},
    {"text": "Investimento em títulos do tesouro direto passa a ter isenção total de impostos", "expected_theme": "economia_negocios"},
    {"text": "Startups de tecnologia abrem capital na bolsa de Nova York captando bilhões", "expected_theme": "economia_negocios"},
    {"text": "Operação de fusão entre companhias aéreas é barrada pelo conselho de defesa econômica", "expected_theme": "economia_negocios"},
    {"text": "Petrobras anuncia redução nos preços da gasolina refinada nas distribuidoras", "expected_theme": "economia_negocios"},
    {"text": "Setor de serviços cresce acima do esperado no terceiro trimestre consecutivo", "expected_theme": "economia_negocios"},
    {"text": "Companhia siderúrgica constrói novo complexo industrial no porto de Santos", "expected_theme": "economia_negocios"},
    {"text": "Montadora de automóveis anuncia investimento em veículos híbridos nacionais", "expected_theme": "economia_negocios"},
    {"text": "Comércio eletrônico bate recorde de faturamento durante a semana de promoções", "expected_theme": "economia_negocios"},

    # Esportes (20 casos)
    {"text": "Flamengo e Palmeiras empatam clássico eletrizante no Maracanã com dois gols no final", "expected_theme": "esportes"},
    {"text": "Seleção Brasileira confirma novo técnico europeu para o ciclo da Copa do Mundo", "expected_theme": "esportes"},
    {"text": "Atacante da seleção sofre lesão no ligamento cruzado e fica fora da temporada", "expected_theme": "esportes"},
    {"text": "Final da Copa Libertadores da América terá transmissão exclusiva em canal aberto", "expected_theme": "esportes"},
    {"text": "Piloto brasileiro conquista a pole position no GP de Fórmula 1 em Interlagos", "expected_theme": "esportes"},
    {"text": "Clube paulista anuncia a contratação de astro do futebol europeu por três temporadas", "expected_theme": "esportes"},
    {"text": "Campeonato Brasileiro de Futebol tem rodada decisiva na disputa pelo título nacional", "expected_theme": "esportes"},
    {"text": "Comitê Olímpico Internacional inclui novas modalidades nos Jogos Olímpicos de Los Angeles", "expected_theme": "esportes"},
    {"text": "Tenista brasileiro avança para as quartas de final em torneio Grand Slam de Roland Garros", "expected_theme": "esportes"},
    {"text": "Equipe de basquete conquista o troféu da NBA após vitória épica no jogo sete", "expected_theme": "esportes"},
    {"text": "Vôlei feminino do Brasil conquista a medalha de ouro no campeonato mundial", "expected_theme": "esportes"},
    {"text": "Atleta quebra recorde mundial nos cem metros rasos durante etapa da Liga Diamante", "expected_theme": "esportes"},
    {"text": "Federação de futebol suspende árbitro após polêmica com o uso do árbitro de vídeo", "expected_theme": "esportes"},
    {"text": "Ginasta brasileira conquista nota máxima e medalha de ouro no salto individual", "expected_theme": "esportes"},
    {"text": "Maratona internacional de São Paulo reúne milhares de corredores nas ruas da capital", "expected_theme": "esportes"},
    {"text": "Boxeador brasileiro vence por nocaute e fatura cinturão mundial dos pesos médios", "expected_theme": "esportes"},
    {"text": "Estádio de futebol é interditado pela defesa civil por problemas na cobertura", "expected_theme": "esportes"},
    {"text": "Surfista brasileiro vence etapa do circuito mundial na praia de Saquarema", "expected_theme": "esportes"},
    {"text": "Torcida organizada do clube é punida com perda de mando de campo por tumulto", "expected_theme": "esportes"},
    {"text": "Nadador conquista três medalhas nos Jogos Pan-Americanos no Chile", "expected_theme": "esportes"},

    # Ciência, Tecnologia e Segurança/Cotidiano (20 casos)
    {"text": "A Terra é plana e as fotos espaciais da NASA são geradas por computação gráfica", "expected_theme": "ciencia_tecnologia"},
    {"text": "Rede 5G altera o DNA humano e transmite sinais para controle mental da população", "expected_theme": "ciencia_tecnologia"},
    {"text": "Carro elétrico explode imediatamente se for lavado em lava-rápido ou tomar chuva", "expected_theme": "ciencia_tecnologia"},
    {"text": "Telescópio espacial James Webb captura imagens inéditas de galáxias primordiais", "expected_theme": "ciencia_tecnologia"},
    {"text": "Inteligência artificial adquire consciência própria e desliga servidores do data center", "expected_theme": "ciencia_tecnologia"},
    {"text": "Cientistas conseguem sintetizar novo material supercondutor em temperatura ambiente", "expected_theme": "ciencia_tecnologia"},
    {"text": "Sonda espacial da missão Artemis completa com sucesso a órbita ao redor da Lua", "expected_theme": "ciencia_tecnologia"},
    {"text": "Hackers invadem sistema bancário internacional e sequestram bancos de dados", "expected_theme": "ciencia_tecnologia"},
    {"text": "Novo processador quântico realiza cálculo complexo em milionésimos de segundo", "expected_theme": "ciencia_tecnologia"},
    {"text": "Astrônomos detectam sinal de rádio repetitivo vindo de sistema estelar distante", "expected_theme": "ciencia_tecnologia"},
    {"text": "Gangue do palhaço em van branca está sequestrando crianças perto de escolas públicas", "expected_theme": "cotidiano_seguranca"},
    {"text": "Perfume borrifado em estacionamento de shopping faz vítimas desmaiarem em segundos", "expected_theme": "cotidiano_seguranca"},
    {"text": "Polícia Militar realiza megaoperação desarticulando quadrilha de roubo de cargas", "expected_theme": "cotidiano_seguranca"},
    {"text": "Incêndio atinge galpão industrial na região metropolitana sem deixar vítimas feridas", "expected_theme": "cotidiano_seguranca"},
    {"text": "Câmeras de segurança registram assalto a joalheria no centro comercial da cidade", "expected_theme": "cotidiano_seguranca"},
    {"text": "Defesa Civil emite alerta vermelho de temporais e risco de deslizamento de terra", "expected_theme": "cotidiano_seguranca"},
    {"text": "Linha do metrô é paralisada após falha elétrica no sistema de sinalização de trens", "expected_theme": "cotidiano_seguranca"},
    {"text": "Operação conjunta resgata animais silvestres mantidos em cativeiro ilegal em chácara", "expected_theme": "cotidiano_seguranca"},
    {"text": "Acidente entre dois caminhões interdita pista principal da rodovia Presidente Dutra", "expected_theme": "cotidiano_seguranca"},
    {"text": "Moradores organizam mutirão solidário para revitalizar praça comunitária no bairro", "expected_theme": "cotidiano_seguranca"},
]


def main():
    logger.info("=" * 70)
    logger.info("AVALIAÇÃO DO GATEKEEPER DE SAÚDE EM FALÁCIAS POPULARES (ONNX INT8)")
    logger.info("=" * 70)

    predictor = ThemePredictor(model_dir="models", prefer_int8=True)
    logger.info(f"Modelo carregado: {predictor.model_type} ({predictor.model_path})")

    results: List[Dict] = []
    latencies: List[float] = []

    logger.info(f"Avaliando {len(HEALTH_MYTHS)} falácias de saúde e {len(NON_HEALTH_CONTROLS)} controles de outros domínios...")

    # 1. Processar Falácias de Saúde (True Positives esperados para 'saude')
    for item in HEALTH_MYTHS:
        text = item["text"]
        cat = item["category"]
        pred = predictor.predict(text, max_length=64)
        latencies.append(pred["latency_ms"])

        is_health_predicted = pred["is_health"]
        health_prob = pred["health_probability"]
        predicted_theme = pred["theme"]
        confidence = pred["confidence"]

        results.append({
            "text": text,
            "true_label": "saude",
            "is_health_ground_truth": True,
            "subcategory": cat,
            "predicted_theme": predicted_theme,
            "is_health_predicted": is_health_predicted,
            "health_probability": health_prob,
            "confidence": confidence,
            "latency_ms": pred["latency_ms"],
        })

    # 2. Processar Controles Não-Saúde (True Negatives esperados para 'saude')
    for item in NON_HEALTH_CONTROLS:
        text = item["text"]
        expected_theme = item["expected_theme"]
        pred = predictor.predict(text, max_length=64)
        latencies.append(pred["latency_ms"])

        is_health_predicted = pred["is_health"]
        health_prob = pred["health_probability"]
        predicted_theme = pred["theme"]
        confidence = pred["confidence"]

        results.append({
            "text": text,
            "true_label": expected_theme,
            "is_health_ground_truth": False,
            "subcategory": f"controle_{expected_theme}",
            "predicted_theme": predicted_theme,
            "is_health_predicted": is_health_predicted,
            "health_probability": health_prob,
            "confidence": confidence,
            "latency_ms": pred["latency_ms"],
        })

    df = pd.DataFrame(results)

    # =====================================================================
    # MÉTRICAS DO GATEKEEPER DE SAÚDE
    # =====================================================================
    health_mask = df["is_health_ground_truth"] == True
    non_health_mask = df["is_health_ground_truth"] == False

    tp = int((df[health_mask]["is_health_predicted"] == True).sum())
    fn = int((df[health_mask]["is_health_predicted"] == False).sum())
    tn = int((df[non_health_mask]["is_health_predicted"] == False).sum())
    fp = int((df[non_health_mask]["is_health_predicted"] == True).sum())

    total_health = len(HEALTH_MYTHS)
    total_non_health = len(NON_HEALTH_CONTROLS)
    total_samples = len(df)

    recall = tp / total_health if total_health > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / total_non_health if total_non_health > 0 else 0.0
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    mean_health_prob_on_myths = float(df[health_mask]["health_probability"].mean())
    median_health_prob_on_myths = float(df[health_mask]["health_probability"].median())
    mean_health_prob_on_controls = float(df[non_health_mask]["health_probability"].mean())

    mean_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # =====================================================================
    # QUEBRA POR SUBCATEGORIA DE FALÁCIAS DE SAÚDE
    # =====================================================================
    subcat_stats = {}
    for cat, group in df[health_mask].groupby("subcategory"):
        cat_tp = int((group["is_health_predicted"] == True).sum())
        cat_total = len(group)
        cat_recall = cat_tp / cat_total if cat_total > 0 else 0.0
        cat_mean_prob = float(group["health_probability"].mean())
        subcat_stats[cat] = {
            "total": cat_total,
            "detected": cat_tp,
            "recall": round(cat_recall, 4),
            "mean_health_prob": round(cat_mean_prob, 4),
        }

    # =====================================================================
    # CASOS DE FALHA (FALSE NEGATIVES / FALSE POSITIVES)
    # =====================================================================
    false_negatives = df[health_mask & (df["is_health_predicted"] == False)][
        ["text", "subcategory", "predicted_theme", "confidence", "health_probability"]
    ].to_dict(orient="records")

    false_positives = df[non_health_mask & (df["is_health_predicted"] == True)][
        ["text", "true_label", "predicted_theme", "confidence", "health_probability"]
    ].to_dict(orient="records")

    # Montar relatório consolidado
    eval_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": predictor.model_type,
        "sample_counts": {
            "total": total_samples,
            "health_myths": total_health,
            "non_health_controls": total_non_health,
        },
        "gatekeeper_metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall_sensitivity": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp,
        },
        "confidence_levels": {
            "mean_health_prob_on_myths": round(mean_health_prob_on_myths, 4),
            "median_health_prob_on_myths": round(median_health_prob_on_myths, 4),
            "mean_health_prob_on_controls": round(mean_health_prob_on_controls, 4),
        },
        "latency_ms": {
            "mean": round(mean_latency, 2),
            "median": round(median_latency, 2),
            "p95": round(p95_latency, 2),
        },
        "subcategory_breakdown": subcat_stats,
        "false_negatives_count": len(false_negatives),
        "false_positives_count": len(false_positives),
        "false_negatives_sample": false_negatives[:10],
        "false_positives_sample": false_positives[:10],
    }

    # Salvar resultados
    output_dir = Path("models")
    report_json_path = output_dir / "health_myths_eval_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)

    df_path = output_dir / "health_myths_eval_details.tsv"
    df.to_csv(df_path, sep="\t", index=False)

    # =====================================================================
    # EXIBIÇÃO NO CONSOLE
    # =====================================================================
    print("\n" + "=" * 75)
    print("           RELATÓRIO DE PRECISÃO: TESTE COM FALÁCIAS DE SAÚDE")
    print("=" * 75)
    print(f"Total de Amostras Avaliadas : {total_samples}")
    print(f"Falácias e Mitos de Saúde   : {total_health}")
    print(f"Controles de Outros Temas   : {total_non_health}")
    print(f"Modelo / Engine             : {predictor.model_type}")
    print(f"Latência Média por Inferência: {mean_latency:.2f} ms")
    print("-" * 75)
    print("MATRIZ DE CONFUSÃO DO GATEKEEPER DE SAÚDE:")
    print(f"  Verdadeiros Positivos (Mitos de Saúde Detectados) : {tp:3d} / {total_health} ({recall*100:.1f}%)")
    print(f"  Falsos Negativos (Mitos de Saúde Não Capturados)  : {fn:3d} / {total_health} ({(1-recall)*100:.1f}%)")
    print(f"  Verdadeiros Negativos (Outros Temas Rejeitados)   : {tn:3d} / {total_non_health} ({specificity*100:.1f}%)")
    print(f"  Falsos Positivos (Outros Temas Rotulados Saúde)   : {fp:3d} / {total_non_health} ({(1-specificity)*100:.1f}%)")
    print("-" * 75)
    print("MÉTRICAS OFICIAIS DO GATEKEEPER:")
    print(f"  Sensibilidade (Recall) : {recall*100:6.2f}% (Capacidade de capturar falácias médicas)")
    print(f"  Precisão               : {precision*100:6.2f}% (Confiabilidade quando afirma ser saúde)")
    print(f"  Especificidade         : {specificity*100:6.2f}% (Capacidade de ignorar temas não-médicos)")
    print(f"  Acurácia Global        : {accuracy*100:6.2f}%")
    print(f"  F1-Score               : {f1*100:6.2f}%")
    print(f"  Probabilidade Média em Mitos de Saúde: {mean_health_prob_on_myths*100:.1f}%")
    print("-" * 75)
    print("DESEMPENHO POR SUBCATEGORIA DE FALÁCIA DE SAÚDE:")
    print(f"{'Subcategoria':<30} | {'Detectados':<12} | {'Recall':<10} | {'Prob Média':<10}")
    print("-" * 75)
    for cat, s in subcat_stats.items():
        print(f"{cat:<30} | {s['detected']:2d} / {s['total']:2d}      | {s['recall']*100:>6.1f}%   | {s['mean_health_prob']*100:>6.1f}%")
    print("=" * 75)

    if false_negatives:
        print(f"\nExemplos de Falácias de Saúde Não Detectadas (Falsos Negativos, total={len(false_negatives)}):")
        for i, fn_case in enumerate(false_negatives[:5], 1):
            print(f"  {i}. \"{fn_case['text']}\" -> Previsto como '{fn_case['predicted_theme']}' ({fn_case['confidence']*100:.1f}%) [P(saude)={fn_case['health_probability']*100:.1f}%]")

    if false_positives:
        print(f"\nExemplos de Outros Temas Confundidos com Saúde (Falsos Positivos, total={len(false_positives)}):")
        for i, fp_case in enumerate(false_positives[:5], 1):
            print(f"  {i}. \"{fp_case['text']}\" -> Tema Real: '{fp_case['true_label']}' [P(saude)={fp_case['health_probability']*100:.1f}%]")

    print(f"\nRelatório completo salvo em: {report_json_path}")
    print(f"Tabela detalhada com cada predição salva em: {df_path}\n")


if __name__ == "__main__":
    main()

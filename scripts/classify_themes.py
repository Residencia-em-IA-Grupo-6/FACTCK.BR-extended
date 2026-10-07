#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_themes.py

Classificador temático de alta precisão baseado em taxonomia jornalística
para catalogar alegações e notícias em 7 classes:
  1. politica
  2. saude
  3. esporte
  4. entretenimento
  5. economia
  6. tecnologia
  7. seguranca_publica
"""

import re
from typing import Dict, List, Optional, Tuple

THEME_TAXONOMY: Dict[str, List[str]] = {
    "esporte": [
        r"\b(futebol|copa do mundo|jogador(es)?|jogadora[s]?|gols?|flamengo|corinthians|palmeiras|santos fc|s[ãa]o paulo fc|vasco da gama|gr[êe]mio|internacional|cruzeiro|atl[ée]tico[- ]mg|botafogo|fluminense)\b",
        r"\b(neymar|messi|cristiano ronaldo|\bcr7\b|pel[ée]|maradona|tite|ancelotti|vini jr|vinicius junior|mbapp[ée]|rebeca andrade|rayssa leal|gabigol|arrascaeta)\b",
        r"\b(olimp[ií]adas?|jogos ol[ií]mpicos|mundial de clubes|libertadores|champions league|brasileir[ãa]o|s[ée]rie [ab]|copa do brasil|est[áa]dio|maracan[ãa]|allianz parque|morumbi|itaquer[ãa]o)\b",
        r"\b(basquete|v[ôo]lei(bol)?|t[êe]nis|nata[çc][ãa]o|gin[áa]stica art[ií]stica|f[oó]rmula 1|\bf1\b|automobilismo|surfe|skate|jud[ôo]|atleta[s]?|arbitragem|p[êe]nalti|cart[ãa]o vermelho)\b",
    ],
    "entretenimento": [
        r"\b(ator(es)?|atriz(es)?|cantor(a|es|as)?|m[úu]sica[s]?|[áa]lbum|shows?|turn[êe]|videoclipe|cinema|filmes?|novela[s]?|teatro|oscar|grammy|emmy|festival de cannes)\b",
        r"\b(anitta|xuxa|faust[ãa]o|silvio santos|roberto carlos|mar[ií]lia mendon[çc]a|gusttavo lima|lu[ií]sa sonza|virg[ií]nia|whindersson|pabllo vittar|caetano veloso|gilberto gil|chico buarque)\b",
        r"\b(rede globo|tv globo|sbt|\brecord\b|band|netflix|globoplay|streaming|reality show|\bbbb\b|big brother|a fazenda|famoso[sa]|celebridade[s]?|fofoca|hollywood)\b",
    ],
    "tecnologia": [
        r"\b(whatsapp|telegram|facebook|instagram|tiktok|twitter|\bx\b (antigo twitter)|redes? socia[li]s?|youtube)\b",
        r"\b(celular(es)?|smartphones?|computador(es)?|wi-fi|\b5g\b|aplicativo[s]?|\bapp[s]?\b|intelig[êe]ncia artificial|\bia\b|chatgpt|openai|software)\b",
        r"\b(hackers?|ataque cibern[ée]tico|vazamento de dados|golpe do pix|golpe virtual|phishing|criptomoeda[s]?|bitcoins?|deepfake|malware|v[ií]rus de computador)\b",
    ],
    "seguranca_publica": [
        r"\b(pol[ií]cia militar|pol[ií]cia civil|pol[ií]cia federal|\bpm\b|\bprf\b|bope|delegacia|delegado[a]?|batalh[ãa]o|secretaria de seguran[çc]a|guarda municipal|for[çc]a nacional)\b",
        r"\b(assassinat[oa]|homic[ií]dio|latroc[ií]nio|chacina|tiroteio|bala perdida|assalto[s]?|roubo[s]?|sequestro[s]?|estupro|feminic[ií]dio|viol[êe]ncia dom[ée]stica|agress[ãa]o)\b",
        r"\b(fac[çc][ãa]o|fac[çc][õo]es|primeiro comando da capital|\bpcc\b|comando vermelho|\bcv\b|mil[ií]cia[s]?|miliciano[s]?|tr[áa]fico de drogas?|traficante[s]?|crime organizado)\b",
        r"\b(pres[oa]s?|pris[ãa]o|pris[õo]es|cadeia|pres[ií]dio|penitenci[áa]ria|mandado de pris[ãa]o|apreens[ãa]o de (armas|drogas|fuzis)|fuzil|pistola|disparos? de arma|opera[çc][ãa]o policial|investiga[çc][ãa]o policial)\b",
    ],
    "economia": [
        r"\b(infla[çc][ãa]o|\bpib\b|taxa selic|taxa de juros|c[âa]mbio|d[óo]lar|euro|moeda estrangeira|desemprego|mercado financeiro|bolsa de valores|ibovespa|a[çc][õo]es|investimentos?)\b",
        r"\b(gasolina|diesel|combust[ií]ve[li]s?|tarifa de energia|conta de luz|g[áa]s de cozinha|cesta b[áa]sica|sal[áa]rio m[ií]nimo|reajuste salarial|com[ée]rcio|varejo|ind[úu]stria)\b",
        r"\b(imposto[s]?|tributo[s]?|reforma tribut[áa]ria|receita federal|minist[ée]rio da fazenda|banco central|banco do brasil|caixa econ[ôo]mica|cr[ée]dito|inadimpl[êe]ncia)\b",
        r"\b(fgts|\binss\b|aposentadoria[s]?|pens[ãa]o|reforma da previd[êe]ncia|or[çc]amento p[úu]blico|d[íi]vida p[úu]blica|bolsa fam[ií]lia|aux[ií]lio brasil|arrecada[çc][ãa]o|tesouro nacional)\b",
    ],
    "saude": [
        r"\b(c[âa]ncer|tumor(es)?|leucemia|diabete[st]?|insulina|infarto|avc|derrame|sarampo|febre amarela|dengue|zika|chikungunya|mal[aá]ria|leptospirose|tuberculose|aids|\bhiv\b|covid(-19)?|coronav[ií]rus|gripe|h1n1)\b",
        r"\b(vacina(s|[çc][ãa]o|do[sa]?|r)?|imuniza([çc][ãa]o|do[sa]?|r)?|medicamento[s]?|rem[ée]dio[s]?|f[áa]rmaco[s]?|antibi[oó]tico[s]?|cirurgia[s]?|quimioterapia)\b",
        r"\b(sistema [úu]nico de sa[úu]de|\bsus\b|minist[ée]rio da sa[úu]de|anvisa|organiza[çc][ãa]o mundial da sa[úu]de|\boms\b|hospital(ar|ares)?|\buti\b|leitos?|postos? de sa[úu]de|m[ée]dic[oa]s?|enfermeir[oa]s?)\b",
    ],
    "politica": [
        r"\b(governo federal|presid[êe]ncia da rep[úu]blica|pal[áa]cio do planalto|supremo tribunal federal|\bstf\b|congresso nacional|c[âa]mara dos deputados|senado federal|tribunal superior eleitoral|\btse\b|\btcu\b)\b",
        r"\b(lula|jair bolsonaro|michel temer|dilma rousseff|s[ée]rgio moro|deputad[oa]s?|senador(es)?|governador(es)?|prefeito[s]?|ministr[oa]s? de estado)\b",
        r"\b(elei[çc][õo]es?|voto[s]?|urna eletr[ôo]nica|campanha eleitoral|partido[s]? pol[ií]tico[s]?|partido dos trabalhadores|\bpt\b|\bpl\b|\bpsdb\b|\bpsol\b|\bmdb\b|uni[ãa]o brasil|bancada|mandato|impeachment|projeto de lei|\bcpi\b)\b",
    ],
}

COMPILED_TAXONOMY = {
    theme: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for theme, patterns in THEME_TAXONOMY.items()
}

# Ordem de precedência para evitar que termos genéricos de política engulam temas especializados
THEME_PRECEDENCE = [
    "esporte",
    "entretenimento",
    "tecnologia",
    "saude",
    "seguranca_publica",
    "economia",
    "politica",
]


def classify_theme(text: str) -> Optional[str]:
    """
    Classifica um texto em uma das 7 classes temáticas.
    Retorna o nome do tema ou None caso não atinja nenhum critério.
    """
    if not text:
        return None

    # 1. Contagem ponderada de matches por categoria
    scores = {theme: 0 for theme in THEME_PRECEDENCE}
    for theme in THEME_PRECEDENCE:
        for regex in COMPILED_TAXONOMY[theme]:
            matches = regex.findall(text)
            scores[theme] += len(matches)

    # 2. Selecionar o tema com maior pontuação positiva
    max_score = max(scores.values())
    if max_score == 0:
        return None

    # Em caso de empate, aplicar precedência temática (domínios específicos têm prioridade sobre política genérica)
    for theme in THEME_PRECEDENCE:
        if scores[theme] == max_score:
            return theme

    return None


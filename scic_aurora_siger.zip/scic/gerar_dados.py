# -*- coding: utf-8 -*-
"""
gerar_dados.py - Gera a base SIMULADA da Aurora Siger (dados_aurora_siger.csv).

Esta base é fictícia, mas coerente: cada módulo tem uma latência "real" (observada)
que depende da carga da rede, da corrente e do tipo do módulo. A latência "prevista"
é uma estimativa simples do operador, que NÃO conhece a deriva (piora lenta) de
alguns módulos, e por isso erra mais neles.
Semente fixa (42) => o arquivo gerado é sempre o mesmo.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# id do tipo (vira o 1º dígito hexadecimal do código do sensor), latência base (ms),
# tensão de operação (V), faixa de corrente (A)
TIPOS = {
    "comunicacao":    {"id": 0x3, "base": 40, "faixa_i": (1.5, 6.0)},
    "armazenamento":  {"id": 0x6, "base": 65, "faixa_i": (0.8, 3.5)},
    "habitacao":      {"id": 0x1, "base": 55, "faixa_i": (1.0, 5.0)},
    "agricultura":    {"id": 0x2, "base": 85, "faixa_i": (0.8, 3.0)},
    "laboratorio":    {"id": 0x4, "base": 70, "faixa_i": (1.0, 4.0)},
    "suporte_medico": {"id": 0x5, "base": 50, "faixa_i": (1.0, 4.5)},
}

# nome, tipo, tensão (V), essencial (1/0), deriva de latência (ms por ciclo)
MODULOS = [
    ("Comunicação Principal",   "comunicacao",    24, 1, 0.0),
    ("Comunicação Reserva",     "comunicacao",    24, 0, 3.0),   # piora com o tempo
    ("Controle de Antenas",     "comunicacao",    48, 0, 0.0),
    ("Comando Central",         "armazenamento",  12, 1, 0.0),
    ("Armazenamento de Dados",  "armazenamento",  12, 1, 0.0),
    ("Habitação Alfa",          "habitacao",      24, 1, 0.0),
    ("Habitação Beta",          "habitacao",      24, 0, 0.0),
    ("Agricultura Hidropônica", "agricultura",    12, 0, 4.0),   # piora com o tempo
    ("Laboratório Geológico",   "laboratorio",    24, 0, 0.0),
    ("Suporte Médico",          "suporte_medico", 24, 1, 0.0),
]
N_CICLOS = 8

linhas = []
for ciclo in range(1, N_CICLOS + 1):
    for idx, (nome, tipo, tensao, essencial, deriva) in enumerate(MODULOS):
        t = TIPOS[tipo]
        corrente = round(rng.uniform(*t["faixa_i"]), 2)
        carga = round(rng.uniform(30, 90), 1)
        # latência real: base + efeito da carga + efeito da corrente + deriva + ruído (+ pico raro)
        lat_obs = t["base"] + 0.45 * carga + 2.2 * corrente + deriva * ciclo + rng.normal(0, 4)
        if rng.random() < 0.08:
            lat_obs += rng.uniform(30, 60)                     # pico (falha momentânea)
        # estimativa do operador: simples, sem deriva, com pequeno ruído
        lat_prev = t["base"] + 0.40 * carga + 2.0 * corrente + rng.normal(0, 2.5)
        lat_obs, lat_prev = round(lat_obs, 1), round(lat_prev, 1)

        erro_rel = abs(lat_obs - lat_prev) / lat_obs
        if rng.random() < 0.04:
            status, prioridade, msg = "manutencao", 2, "Manutenção programada do enlace"
        elif erro_rel >= 0.25 or lat_obs >= 140:
            status = "alerta"
            prioridade = 3 if (essencial or erro_rel >= 0.40) else 2
            msg = ("Latência muito acima do previsto" if erro_rel >= 0.40
                   else "Latência acima do previsto")
        else:
            status, prioridade, msg = "ativo", 1, "Operação normal"

        linhas.append({
            "ciclo": ciclo, "modulo": nome, "tipo": tipo,
            "codigo_sensor": f"{t['id']:X}{0x0A0 + idx * 0x11:03X}",   # ex.: 30A0 (hex)
            "latencia_observada_ms": lat_obs, "latencia_prevista_ms": lat_prev,
            "tensao_v": tensao, "corrente_a": corrente, "carga_rede_pct": carga,
            "essencial": essencial, "status": status,
            "prioridade": prioridade, "mensagem_alerta": msg,
        })

df = pd.DataFrame(linhas)
df.to_csv("dados_aurora_siger.csv", index=False, encoding="utf-8")
print(df.shape, df["status"].value_counts().to_dict())

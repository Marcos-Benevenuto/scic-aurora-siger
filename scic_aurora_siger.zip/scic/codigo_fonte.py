# -*- coding: utf-8 -*-
"""
SCIC - Sistema de Comunicação Interplanetária da Colônia (Aurora Siger)
=======================================================================
Protótipo em Python que:
  1. carrega/cadastra dados simulados dos módulos da colônia (CSV + Pandas);
  2. calcula indicadores e erros numéricos (absoluto e relativo);
  3. simula a evolução da latência com o método de Euler;
  4. treina um modelo simples de previsão e avalia (MAE, MSE, RMSE, R², AIC, BIC);
  5. prioriza alertas com uma HEAP (fila de prioridade) implementada à mão;
  6. busca registros por prefixo com uma TRIE implementada à mão;
  7. relaciona o sistema a E/S, bases numéricas e eletricidade básica;
  8. discute gerenciamento inteligente (manutenção preditiva e redundância).

Execução:
    python codigo_fonte.py          -> menu interativo
    python codigo_fonte.py --auto   -> roda todas as etapas em sequência (demo)
"""

import argparse
import math
import os
import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # salva figuras em arquivo (sem abrir janela)
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------------
PASTA = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(PASTA, "dados_aurora_siger.csv")
PASTA_GRAFICOS = os.path.join(PASTA, "graficos_ou_imagens")
LIMITE_ACEITAVEL = 15.0     # erro relativo (%) até aqui: aceitável
LIMITE_PREOCUPANTE = 30.0   # acima daqui: preocupante (entre os dois: atenção)
COLUNAS = ["ciclo", "modulo", "tipo", "codigo_sensor", "latencia_observada_ms",
           "latencia_prevista_ms", "tensao_v", "corrente_a", "carga_rede_pct",
           "essencial", "status", "prioridade", "mensagem_alerta"]


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# ----------------------------------------------------------------------------
# 1. Carga e cadastro de dados
# ----------------------------------------------------------------------------
def carregar_dados(caminho=ARQUIVO_DADOS):
    """Lê o CSV, confere as colunas e cria as colunas calculadas."""
    df = pd.read_csv(caminho, encoding="utf-8")
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {faltando}")
    df = df.dropna(subset=["latencia_observada_ms", "latencia_prevista_ms"])  # limpeza
    df = calcular_erros(df)
    df["potencia_w"] = (df["tensao_v"] * df["corrente_a"]).round(2)           # P = V * I
    return df.reset_index(drop=True)


def calcular_erros(df):
    """Erro absoluto = |previsto - observado|; relativo = absoluto / |observado| (em %)."""
    df = df.copy()
    df["erro_abs_ms"] = (df["latencia_prevista_ms"] - df["latencia_observada_ms"]).abs()
    df["erro_rel_pct"] = df["erro_abs_ms"] / df["latencia_observada_ms"].abs() * 100
    return df


def classificar_erro(pct):
    if pct < LIMITE_ACEITAVEL:
        return "aceitável"
    if pct < LIMITE_PREOCUPANTE:
        return "atenção"
    return "preocupante"


def cadastrar_registro(df):
    """Cadastra um novo registro pelo terminal e grava no CSV."""
    titulo("CADASTRAR NOVO REGISTRO")
    try:
        novo = {
            "ciclo": int(input("Ciclo (inteiro): ")),
            "modulo": input("Nome do módulo: ").strip(),
            "tipo": input("Tipo (comunicacao/armazenamento/habitacao/agricultura/"
                          "laboratorio/suporte_medico): ").strip(),
            "codigo_sensor": input("Código do sensor (hex, ex.: 30A0): ").strip().upper(),
            "latencia_observada_ms": float(input("Latência observada (ms): ")),
            "latencia_prevista_ms": float(input("Latência prevista (ms): ")),
            "tensao_v": float(input("Tensão (V): ")),
            "corrente_a": float(input("Corrente (A): ")),
            "carga_rede_pct": float(input("Carga da rede (%): ")),
            "essencial": int(input("Módulo essencial? (1=sim, 0=não): ")),
            "status": input("Status (ativo/alerta/manutencao): ").strip().lower(),
            "prioridade": int(input("Prioridade (1 a 3): ")),
            "mensagem_alerta": input("Mensagem resumida: ").strip(),
        }
    except ValueError:
        print("Valor inválido. Cadastro cancelado.")
        return df
    base = pd.read_csv(ARQUIVO_DADOS, encoding="utf-8")
    base = pd.concat([base, pd.DataFrame([novo])], ignore_index=True)
    base.to_csv(ARQUIVO_DADOS, index=False, encoding="utf-8")
    print("Registro salvo em", os.path.basename(ARQUIVO_DADOS))
    return carregar_dados()


# ----------------------------------------------------------------------------
# 2. Consulta de registros
# ----------------------------------------------------------------------------
def consultar_registros(df, auto=False):
    titulo("CONSULTA DE REGISTROS")
    print(f"Base carregada: {len(df)} registros, {df['modulo'].nunique()} módulos, "
          f"{df['ciclo'].nunique()} ciclos.")
    print("\nContagem por status:")
    print(df["status"].value_counts().to_string())
    cols = ["ciclo", "modulo", "codigo_sensor", "latencia_observada_ms",
            "latencia_prevista_ms", "status", "prioridade"]
    if auto:
        print("\nExemplo: registros com status 'alerta':")
        print(df.loc[df["status"] == "alerta", cols].to_string(index=False))
        return
    filtro = input("\nFiltrar por [m]ódulo, [c]iclo, [s]tatus ou ENTER para os 10 primeiros: ").strip().lower()
    if filtro == "m":
        t = normalizar(input("Parte do nome do módulo: "))
        res = df[df["modulo"].map(normalizar).str.contains(t, regex=False)]
    elif filtro == "c":
        res = df[df["ciclo"] == int(input("Ciclo: "))]
    elif filtro == "s":
        res = df[df["status"] == input("Status (ativo/alerta/manutencao): ").strip().lower()]
    else:
        res = df.head(10)
    print(res[cols].to_string(index=False) if len(res) else "Nenhum registro encontrado.")


# ----------------------------------------------------------------------------
# 3. Indicadores e erros numéricos
# ----------------------------------------------------------------------------
def indicadores_e_erros(df, verbose=True):
    resumo = df.groupby("modulo").agg(
        lat_media_ms=("latencia_observada_ms", "mean"),
        erro_abs_medio_ms=("erro_abs_ms", "mean"),
        erro_rel_medio_pct=("erro_rel_pct", "mean"),
    ).round(2)
    resumo["classificacao"] = resumo["erro_rel_medio_pct"].map(classificar_erro)
    resumo = resumo.sort_values("erro_rel_medio_pct", ascending=False)

    disponibilidade = (df["status"] == "ativo").mean() * 100     # % de registros ativos
    pct_alerta = (df["status"] == "alerta").mean() * 100
    if not verbose:
        return resumo, disponibilidade, pct_alerta

    titulo("INDICADORES E ERROS NUMÉRICOS")
    print(f"Disponibilidade (registros ativos): {disponibilidade:.1f}%")
    print(f"Registros em alerta: {pct_alerta:.1f}%")
    print(f"Latência média geral: {df['latencia_observada_ms'].mean():.1f} ms")
    print("\nErro por módulo (ordenado do pior para o melhor):")
    print(resumo.to_string())

    print(f"\nCritério: erro relativo < {LIMITE_ACEITAVEL:.0f}% = aceitável; "
          f"até {LIMITE_PREOCUPANTE:.0f}% = atenção; acima = preocupante.")
    print("Por que ERRO RELATIVO? Um erro de 10 ms é pequeno para um módulo de ~110 ms,")
    print("mas seria grande para um de ~40 ms. O relativo permite comparar escalas diferentes.")

    pior = df.loc[df["erro_rel_pct"].idxmax()]
    print(f"\nMaior erro individual: {pior['modulo']} (ciclo {int(pior['ciclo'])}) -> "
          f"abs = {pior['erro_abs_ms']:.1f} ms, rel = {pior['erro_rel_pct']:.1f}%")

    print("\n--- Ponto flutuante e precisão numérica ---")
    soma = 0.1 + 0.2
    print(f"0.1 + 0.2 = {soma!r}  (igual a 0.3? {soma == 0.3})")
    print(f"math.isclose(0.1 + 0.2, 0.3) = {math.isclose(soma, 0.3)}")
    print("Computadores guardam decimais em binário; 0.1 e 0.2 não têm representação exata.")
    print("Por isso comparamos com tolerância (isclose) e arredondamos (round) na exibição.")
    return resumo, disponibilidade, pct_alerta


# ----------------------------------------------------------------------------
# 4. Simulação com método de Euler
# ----------------------------------------------------------------------------
def simulacao_euler(df, verbose=True):
    """
    Modelo: após um pico, a latência L volta ao valor normal L_alvo:
        dL/dt = -k * (L - L_alvo)
    Euler: L(t+h) = L(t) + h * f(L(t)).  Solução exata: L_alvo + (L0 - L_alvo) * e^(-k t)
    Comparar os dois mostra o ERRO DE TRUNCAMENTO do método numérico.
    """
    linha = df.loc[df["latencia_observada_ms"].idxmax()]
    L0, alvo, k = float(linha["latencia_observada_ms"]), float(linha["latencia_prevista_ms"]), 0.5
    h, T = 1.0, 6.0
    ts = np.arange(0, T + h, h)
    euler = [L0]
    for _ in ts[1:]:
        euler.append(euler[-1] + h * (-k * (euler[-1] - alvo)))
    exata = alvo + (L0 - alvo) * np.exp(-k * ts)
    if verbose:
        titulo("SIMULAÇÃO: RECUPERAÇÃO DA LATÊNCIA (MÉTODO DE EULER)")
        print(f"Módulo: {linha['modulo']} (ciclo {int(linha['ciclo'])}) | pico = {L0:.1f} ms | "
              f"alvo = {alvo:.1f} ms | k = {k} | passo h = {h}")
        print(f"{'t':>3} {'Euler':>9} {'Exata':>9} {'Erro abs':>9} {'Erro rel %':>11}")
        for t, e, x in zip(ts, euler, exata):
            print(f"{t:>3.0f} {e:>9.2f} {x:>9.2f} {abs(e - x):>9.3f} {abs(e - x) / abs(x) * 100:>11.2f}")
        print("O erro existe porque Euler aproxima a curva por retas. Passo menor => erro menor.")
    return ts, np.array(euler), exata


# ----------------------------------------------------------------------------
# 5. Modelo simples de previsão e avaliação
# ----------------------------------------------------------------------------
def metricas(y, pred):
    mse = mean_squared_error(y, pred)
    return {"MAE": mean_absolute_error(y, pred), "MSE": mse,
            "RMSE": math.sqrt(mse), "R2": r2_score(y, pred)}


def aic_bic(y, pred, k):
    """AIC/BIC para regressão com erro gaussiano; k = nº de parâmetros estimados."""
    n = len(y)
    rss = float(np.sum((np.asarray(y) - np.asarray(pred)) ** 2))
    return n * math.log(rss / n) + 2 * k, n * math.log(rss / n) + k * math.log(n)


def modelo_e_metricas(df, verbose=True):
    """Prevê a latência observada a partir de corrente, carga da rede e tipo do módulo."""
    X = pd.get_dummies(df[["tipo", "corrente_a", "carga_rede_pct"]], columns=["tipo"], dtype=float)
    y = df["latencia_observada_ms"]
    # divisão 60% treino / 20% validação / 20% teste
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42)

    candidatos = {"Baseline (média)": DummyRegressor(strategy="mean"),
                  "Regressão linear": LinearRegression()}
    for alpha in (0.01, 0.1, 1.0, 10.0):                      # "grid search" manual do Ridge
        candidatos[f"Ridge (alpha={alpha})"] = Ridge(alpha=alpha)

    tabela = {}
    for nome, mdl in candidatos.items():
        mdl.fit(X_tr, y_tr)
        tabela[nome] = metricas(y_va, mdl.predict(X_va))
    tab = pd.DataFrame(tabela).T
    melhor = tab["RMSE"].idxmin()                              # escolhe pela validação

    final = candidatos[melhor]
    final.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))   # reajusta com treino+validação
    pred_te = final.predict(X_te)
    m_te = metricas(y_te, pred_te)
    m_operador = metricas(y_te, df.loc[y_te.index, "latencia_prevista_ms"])  # estimativa do operador

    lin = LinearRegression().fit(X_tr, y_tr)
    aic_l, bic_l = aic_bic(y_tr, lin.predict(X_tr), X_tr.shape[1] + 1)
    base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    aic_b, bic_b = aic_bic(y_tr, base.predict(X_tr), 1)

    if verbose:
        titulo("MODELO SIMPLES DE PREVISÃO DE LATÊNCIA")
        print(f"Registros: treino={len(X_tr)}, validação={len(X_va)}, teste={len(X_te)}")
        print("Variáveis de entrada: corrente_a, carga_rede_pct e tipo do módulo (one-hot).")
        print("\nValidação (escolha do modelo):")
        print(tab.round(3).to_string())
        print(f"\nModelo escolhido (menor RMSE na validação): {melhor}")
        print("\nTeste final (dados nunca vistos):")
        print(f"  Modelo : MAE={m_te['MAE']:.2f}  MSE={m_te['MSE']:.2f}  "
              f"RMSE={m_te['RMSE']:.2f}  R²={m_te['R2']:.3f}")
        print(f"  Operador (latência prevista no CSV): MAE={m_operador['MAE']:.2f}  "
              f"MSE={m_operador['MSE']:.2f}  RMSE={m_operador['RMSE']:.2f}  R²={m_operador['R2']:.3f}")
        print(f"\nAIC/BIC (treino): linear = {aic_l:.1f}/{bic_l:.1f} | baseline = {aic_b:.1f}/{bic_b:.1f}"
              "  (menor é melhor)")
        print("\nInterpretação:")
        razao = m_te["RMSE"] / m_te["MAE"]
        print(f"  - RMSE/MAE = {razao:.2f}. Quanto maior que 1, mais os erros se concentram em "
              "poucos registros (picos).")
        print("  - R² alto não significa modelo perfeito: ele só diz quanto da variação foi explicada;")
        print("    com poucos dados, ou com picos raros, o número pode enganar. Olhe todas as métricas.")
        print("  - MAE está em ms (fácil de explicar); RMSE penaliza erros grandes; MSE está em ms².")
    return {"melhor": melhor, "teste": m_te, "operador": m_operador, "tabela": tab,
            "y_te": y_te, "pred_te": pred_te}


# ----------------------------------------------------------------------------
# 6. HEAP para priorização de alertas (implementada à mão)
# ----------------------------------------------------------------------------
class HeapAlertas:
    """
    Heap binária de prioridade guardada em lista. Os itens são tuplas
    (-score, ciclo, modulo, mensagem). Como a heap é de MÍNIMO e usamos -score,
    o topo (posição 0) é sempre o alerta de MAIOR score. Empates: ciclo menor (mais antigo).
    Relações de índices: pai = (i-1)//2, filhos = 2i+1 e 2i+2.
    """

    def __init__(self):
        self.itens = []
        self.comparacoes = 0               # contador usado para comparar com lista simples

    def __len__(self):
        return len(self.itens)

    def _menor(self, i, j):
        self.comparacoes += 1
        return self.itens[i] < self.itens[j]

    def inserir(self, item):
        self.itens.append(item)
        self._subir(len(self.itens) - 1)

    def _subir(self, i):                   # heapify-up
        while i > 0:
            pai = (i - 1) // 2
            if self._menor(i, pai):
                self.itens[i], self.itens[pai] = self.itens[pai], self.itens[i]
                i = pai
            else:
                break

    def remover_topo(self):
        if not self.itens:
            return None
        topo = self.itens[0]
        ultimo = self.itens.pop()
        if self.itens:
            self.itens[0] = ultimo
            self._descer(0)
        return topo

    def _descer(self, i):                  # heapify-down
        n = len(self.itens)
        while True:
            esq, dir_, menor = 2 * i + 1, 2 * i + 2, i
            if esq < n and self._menor(esq, menor):
                menor = esq
            if dir_ < n and self._menor(dir_, menor):
                menor = dir_
            if menor == i:
                break
            self.itens[i], self.itens[menor] = self.itens[menor], self.itens[i]
            i = menor


def calcular_score(linha):
    """
    Critério de urgência (0 a ~100):
      prioridade (1 a 3) x 30  +  20 se módulo essencial  +  0,5 x erro relativo (máx. 50)
      + 15 se latência observada >= 140 ms.
    """
    score = linha["prioridade"] * 30
    score += 20 if linha["essencial"] == 1 else 0
    score += 0.5 * min(linha["erro_rel_pct"], 100)
    score += 15 if linha["latencia_observada_ms"] >= 140 else 0
    return round(score, 1)


def priorizar_alertas(df, verbose=True, mostrar_estado=True):
    alertas = df[df["status"] == "alerta"]
    heap = HeapAlertas()
    for _, lin in alertas.iterrows():
        heap.inserir((-calcular_score(lin), int(lin["ciclo"]), lin["modulo"], lin["mensagem_alerta"]))
    ordem = []
    copia_itens = list(heap.itens)
    while len(heap):
        ordem.append(heap.remover_topo())
    if verbose:
        titulo("PRIORIZAÇÃO DE ALERTAS COM HEAP")
        print("Critério: score = prioridade x 30 + 20 (essencial) + 0,5 x erro relativo % "
              "+ 15 (latência >= 140 ms)")
        print(f"Alertas na base: {len(alertas)}")
        if mostrar_estado:
            print("\nEstado interno da heap (lista; posição 0 = mais urgente):")
            for pos, it in enumerate(copia_itens[:5]):
                print(f"  [{pos}] score={-it[0]:>5} | ciclo {it[1]} | {it[2]}")
            print("  ... (pai de i = (i-1)//2; filhos = 2i+1 e 2i+2)")
        print("\nAlertas na ordem de atendimento (remoções sucessivas do topo):")
        for n, (neg, ciclo, mod, msg) in enumerate(ordem, 1):
            print(f"  {n:>2}º  score={-neg:>5} | ciclo {ciclo} | {mod:<24} | {msg}")
        if ordem:
            print(f"\nAlerta MAIS URGENTE agora: {ordem[0][2]} (ciclo {ordem[0][1]}, score {-ordem[0][0]})")
    return ordem


def comparar_heap_com_lista(n=1000):
    """Conta comparações para tirar todos os alertas em ordem: heap x lista simples."""
    rng = np.random.default_rng(0)
    scores = [(-float(s), i, "m", "x") for i, s in enumerate(rng.uniform(0, 100, n))]
    heap = HeapAlertas()
    for s in scores:
        heap.inserir(s)
    while len(heap):
        heap.remover_topo()
    lista, comp_lista = list(scores), 0
    while lista:                             # lista simples: varre tudo para achar o menor
        melhor = 0
        for i in range(1, len(lista)):
            comp_lista += 1
            if lista[i] < lista[melhor]:
                melhor = i
        lista.pop(melhor)
    titulo(f"HEAP x LISTA SIMPLES ({n} alertas)")
    print(f"Comparações com heap ......: {heap.comparacoes:>9,}  (~ n log n)".replace(",", "."))
    print(f"Comparações com lista .....: {comp_lista:>9,}  (~ n²/2)".replace(",", "."))
    print("A heap acha o mais urgente em O(1) (topo) e reorganiza em O(log n);")
    print("a lista precisa varrer tudo (O(n)) a cada alerta atendido.")
    return heap.comparacoes, comp_lista


# ----------------------------------------------------------------------------
# 7. TRIE para busca por prefixo (implementada à mão)
# ----------------------------------------------------------------------------
def normalizar(texto):
    """Minúsculas e sem acentos: 'Comunicação' -> 'comunicacao'."""
    sem = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in sem if not unicodedata.combining(c)).lower().strip()


class NoTrie:
    def __init__(self):
        self.filhos = {}        # letra -> NoTrie
        self.registros = []     # o que foi cadastrado terminando neste nó


class Trie:
    """Árvore de prefixos: cada nível é uma letra; palavras com o mesmo início compartilham caminho."""

    def __init__(self):
        self.raiz = NoTrie()

    def inserir(self, chave, registro):
        no = self.raiz
        for letra in normalizar(chave):
            no = no.filhos.setdefault(letra, NoTrie())
        no.registros.append(registro)

    def buscar_prefixo(self, prefixo):
        no = self.raiz
        for letra in normalizar(prefixo):          # desce só pelo caminho do prefixo
            if letra not in no.filhos:
                return []
            no = no.filhos[letra]
        achados, pilha = [], [no]                  # coleta tudo abaixo do nó do prefixo
        while pilha:
            atual = pilha.pop()
            achados.extend(atual.registros)
            pilha.extend(atual.filhos.values())
        return sorted(achados)


COMANDOS = ["chavear_reserva", "calibrar_antena", "comando_reiniciar_enlace", "comunicar_base",
            "consultar_status", "controle_energia_reduzir", "gerar_relatorio", "limpar_alertas"]


def construir_tries(df):
    t_mod, t_sen, t_cmd, t_alr = Trie(), Trie(), Trie(), Trie()
    for nome in df["modulo"].unique():
        t_mod.inserir(nome, nome)
    for _, lin in df.drop_duplicates("codigo_sensor").iterrows():
        t_sen.inserir(lin["codigo_sensor"], f"{lin['codigo_sensor']} -> {lin['modulo']}")
    for cmd in COMANDOS:
        t_cmd.inserir(cmd, cmd)
    for _, lin in df[df["status"] == "alerta"].iterrows():        # palavras-chave dos alertas
        for palavra in str(lin["mensagem_alerta"]).split():
            t_alr.inserir(palavra, palavra.lower())
    return {"modulos": t_mod, "sensores": t_sen, "comandos": t_cmd, "alertas": t_alr}


def busca_por_prefixo(df, auto=False):
    titulo("BUSCA POR PREFIXO (TRIE)")
    tries = construir_tries(df)
    if auto:
        for base, pref in [("modulos", "com"), ("modulos", "hab"), ("sensores", "30"),
                           ("comandos", "co"), ("alertas", "lat")]:
            res = sorted(set(tries[base].buscar_prefixo(pref)))
            print(f"[{base}] prefixo '{pref}' -> {res}")
        print("\nPor que trie? O custo da busca depende do tamanho do PREFIXO, não de quantos")
        print("registros existem; uma lista exigiria testar item por item.")
        return
    print("Bases: modulos, sensores, comandos, alertas")
    while True:
        base = input("\nBase (ENTER para voltar): ").strip().lower()
        if not base:
            return
        if base not in tries:
            print("Base inválida.")
            continue
        pref = input("Prefixo: ")
        res = sorted(set(tries[base].buscar_prefixo(pref)))
        print("Resultados:", res if res else "nenhum")


# ----------------------------------------------------------------------------
# 8. Dispositivos, bases numéricas e eletricidade
# ----------------------------------------------------------------------------
def dispositivos_bases_eletricidade(df):
    titulo("DISPOSITIVOS, BASES NUMÉRICAS E ELETRICIDADE")
    print("ENTRADA (simulada): medidores de tensão e corrente, sensores de latência e de "
          "carga de rede em cada módulo.")
    print("SAÍDA: terminal (este menu), relatório e gráficos em PNG; em campo seria um painel/monitor.")
    print("INTERFACES (conceitual): rede/Wi-Fi para os módulos, USB para manutenção local, "
          "Bluetooth para diagnóstico de curto alcance.")

    print("\n--- Código do sensor em hexadecimal, decimal e binário ---")
    codigo = df.loc[df["tipo"] == "comunicacao", "codigo_sensor"].iloc[0]
    valor = int(codigo, 16)                      # hexadecimal -> decimal
    print(f"Código: 0x{codigo}  ->  decimal {valor}  ->  binário {valor:016b}")
    print(f"Volta: {valor:016b} (bin) = {int(f'{valor:016b}', 2)} (dec) = {valor:X} (hex)")
    tipo_id, numero = valor >> 12, valor & 0xFFF          # deslocamento de bits e máscara
    print(f"Decodificação: 1º dígito hex (tipo) = {tipo_id}  |  número do sensor = 0x{numero:03X} ({numero})")
    print("Convenção: 1=habitação 2=agricultura 3=comunicação 4=laboratório 5=suporte médico 6=armazenamento")

    print("\n--- Tensão, corrente, potência e lei de Ohm (valores médios por módulo) ---")
    med = df.groupby("modulo").agg(V=("tensao_v", "mean"), I=("corrente_a", "mean")).round(2)
    med["P (W) = V x I"] = (med["V"] * med["I"]).round(2)
    med["R (ohm) = V / I"] = (med["V"] / med["I"]).round(2)
    print(med.to_string())
    pot = med["P (W) = V x I"].sum()
    print(f"\nPotência média total dos módulos: {pot:.1f} W ({pot / 1000:.3f} kW)")
    ref = med.loc["Comunicação Principal"]
    print(f"Exemplo: transmissor da Comunicação Principal: {ref['V']:.0f} V x {ref['I']:.2f} A "
          f"= {ref['P (W) = V x I']:.2f} W de potência.")
    print("Em uma colônia com energia limitada, cada watt economizado em comunicação sobra para "
          "habitação e suporte médico.")


# ----------------------------------------------------------------------------
# 9. Gerenciamento inteligente da comunicação
# ----------------------------------------------------------------------------
def tendencias_por_modulo(df):
    """Inclinação (ms por ciclo) da reta ajustada à latência de cada módulo (np.polyfit grau 1)."""
    linhas = []
    for nome, g in df.groupby("modulo"):
        a, b = np.polyfit(g["ciclo"], g["latencia_observada_ms"], 1)
        ult = int(df["ciclo"].max())
        linhas.append({"modulo": nome, "tendencia_ms_por_ciclo": round(a, 2),
                       "projecao_ciclo_+4_ms": round(a * (ult + 4) + b, 1)})
    return pd.DataFrame(linhas).sort_values("tendencia_ms_por_ciclo", ascending=False)


def gerenciamento_inteligente(df, verbose=True):
    tend = tendencias_por_modulo(df)
    suspeitos = tend[tend["tendencia_ms_por_ciclo"] > 2.0]
    # redundância: quando a principal falha/está em manutenção, a reserva está ativa?
    p = df[df["modulo"] == "Comunicação Principal"].set_index("ciclo")["status"]
    r = df[df["modulo"] == "Comunicação Reserva"].set_index("ciclo")["status"]
    falhas = p[p != "ativo"]
    cobertas = [c for c in falhas.index if r.get(c) == "ativo"]
    if not verbose:
        return suspeitos, falhas, cobertas
    titulo("GERENCIAMENTO INTELIGENTE DA COMUNICAÇÃO")
    print("MANUTENÇÃO PREDITIVA - tendência da latência por módulo (reta ajustada):")
    print(tend.to_string(index=False))
    print(f"\nMódulos com piora sustentada (> 2 ms/ciclo): {', '.join(suspeitos['modulo']) or 'nenhum'}")
    print("Esses módulos devem ser inspecionados ANTES de virarem alerta crítico.")
    print("\nREDUNDÂNCIA - Comunicação Principal x Reserva:")
    print(f"Ciclos em que a principal NÃO estava ativa: {list(falhas.index) or 'nenhum'}")
    print(f"Desses, a reserva estava ativa em: {cobertas or 'nenhum'} -> o chaveamento era possível.")
    print("\nCONEXÃO COM OS CONCEITOS:")
    print("- Sensores/medidores inteligentes: geram corrente, tensão, carga e latência por ciclo.")
    print("- Monitoramento contínuo: o erro relativo acusa anomalias (previsto x observado).")
    print("- Automação: a heap indica o alerta mais urgente; regras podem acionar o comando 'chavear_reserva'.")
    print("- Armazenamento e enlaces redundantes: histórico dos ciclos + reserva sustentam a base.")
    print("- Microrredes/redes inteligentes: o consumo (P = V x I) pode ser balanceado entre módulos.")
    print("- Humano no circuito: a equipe valida as decisões automáticas antes de agir.")
    return suspeitos, falhas, cobertas


# ----------------------------------------------------------------------------
# 10. Análise final e gráficos
# ----------------------------------------------------------------------------
def gerar_graficos(df, res_modelo):
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    caminhos = []

    resumo = df.groupby("modulo")["erro_rel_pct"].mean().sort_values()
    cores = ["#c0392b" if v >= LIMITE_PREOCUPANTE else "#e67e22" if v >= LIMITE_ACEITAVEL else "#27ae60"
             for v in resumo]
    plt.figure(figsize=(9, 5))
    plt.barh(resumo.index, resumo.values, color=cores)
    plt.axvline(LIMITE_ACEITAVEL, ls="--", c="gray"); plt.axvline(LIMITE_PREOCUPANTE, ls="--", c="black")
    plt.xlabel("Erro relativo médio (%)"); plt.title("Erro relativo médio por módulo")
    plt.tight_layout()
    c = os.path.join(PASTA_GRAFICOS, "erro_relativo_por_modulo.png"); plt.savefig(c, dpi=120); plt.close()
    caminhos.append(c)

    plt.figure(figsize=(9, 5))
    for nome in ["Comunicação Principal", "Comunicação Reserva", "Agricultura Hidropônica"]:
        g = df[df["modulo"] == nome]
        plt.plot(g["ciclo"], g["latencia_observada_ms"], marker="o", label=nome)
    plt.xlabel("Ciclo"); plt.ylabel("Latência observada (ms)")
    plt.title("Evolução da latência (módulos selecionados)"); plt.legend(); plt.tight_layout()
    c = os.path.join(PASTA_GRAFICOS, "latencia_por_ciclo.png"); plt.savefig(c, dpi=120); plt.close()
    caminhos.append(c)

    y, p = res_modelo["y_te"], res_modelo["pred_te"]
    plt.figure(figsize=(6, 6))
    plt.scatter(y, p)
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    plt.plot(lim, lim, "k--"); plt.xlabel("Latência observada (ms)"); plt.ylabel("Latência prevista pelo modelo (ms)")
    plt.title("Teste: previsto x observado"); plt.tight_layout()
    c = os.path.join(PASTA_GRAFICOS, "previsto_vs_observado.png"); plt.savefig(c, dpi=120); plt.close()
    caminhos.append(c)
    return caminhos


def analise_final(df, res_modelo=None):
    titulo("ANÁLISE FINAL DOS RESULTADOS")
    resumo, disp, pct_alerta = indicadores_e_erros(df, verbose=False)
    if res_modelo is None:
        res_modelo = modelo_e_metricas(df, verbose=False)
    ordem = priorizar_alertas(df, verbose=False)
    suspeitos, falhas, cobertas = gerenciamento_inteligente(df, verbose=False)
    m, op = res_modelo["teste"], res_modelo["operador"]

    print(f"1) Disponibilidade: {disp:.1f}% dos registros ativos; {pct_alerta:.1f}% em alerta.")
    pior = resumo.iloc[0]
    print(f"2) Pior módulo em erro relativo: {resumo.index[0]} "
          f"({pior['erro_rel_medio_pct']:.1f}% - {pior['classificacao']}).")
    print(f"3) Modelo ({res_modelo['melhor']}): MAE={m['MAE']:.2f} ms, RMSE={m['RMSE']:.2f} ms, R²={m['R2']:.3f}.")
    print(f"   Estimativa do operador no mesmo teste: MAE={op['MAE']:.2f} ms, RMSE={op['RMSE']:.2f} ms, R²={op['R2']:.3f}.")
    melhora = "melhor" if m["MAE"] < op["MAE"] else "pior"
    print(f"   -> O modelo teve MAE {melhora} que a estimativa do operador.")
    if ordem:
        print(f"4) Alerta mais urgente: {ordem[0][2]} (ciclo {ordem[0][1]}, score {-ordem[0][0]}).")
    print(f"5) Manutenção preditiva: inspecionar {', '.join(suspeitos['modulo']) or 'nenhum módulo'}.")
    print(f"6) Redundância: {len(cobertas)} de {len(falhas)} indisponibilidades da principal tinham reserva ativa.")
    print("\nLimitações: base pequena e simulada; modelo linear simples; picos são raros e difíceis de prever.")
    print("Decisão final é humana: o sistema apoia, não substitui, a equipe da Aurora Siger.")
    for c in gerar_graficos(df, res_modelo):
        print("Gráfico salvo:", os.path.relpath(c, PASTA))


# ----------------------------------------------------------------------------
# Menu
# ----------------------------------------------------------------------------
MENU = """
==================== SCIC - AURORA SIGER ====================
 1) Carregar dados / cadastrar novo registro
 2) Consultar registros
 3) Indicadores e erros numéricos (absoluto e relativo)
 4) Simulação da latência (método de Euler)
 5) Previsão simples e métricas (MAE, MSE, RMSE, R², AIC, BIC)
 6) Priorizar alertas (heap)  [+ comparação heap x lista]
 7) Busca por prefixo (trie)
 8) Dispositivos, bases numéricas e eletricidade
 9) Gerenciamento inteligente da comunicação
10) Análise final (gera gráficos)
 0) Sair
=============================================================
"""


def executar_tudo(df):
    consultar_registros(df, auto=True)
    indicadores_e_erros(df)
    simulacao_euler(df)
    res = modelo_e_metricas(df)
    priorizar_alertas(df)
    comparar_heap_com_lista()
    busca_por_prefixo(df, auto=True)
    dispositivos_bases_eletricidade(df)
    gerenciamento_inteligente(df)
    analise_final(df, res)


def main():
    parser = argparse.ArgumentParser(description="SCIC - Aurora Siger")
    parser.add_argument("--auto", action="store_true", help="executa todas as etapas sem menu")
    args = parser.parse_args()

    df = carregar_dados()
    print(f"Dados carregados: {len(df)} registros de {os.path.basename(ARQUIVO_DADOS)}.")
    if args.auto:
        executar_tudo(df)
        return

    res_modelo = None
    while True:
        print(MENU)
        op = input("Escolha uma opção: ").strip()
        try:
            if op == "1":
                if input("[c]arregar de novo ou [n]ovo registro? ").strip().lower() == "n":
                    df = cadastrar_registro(df)
                else:
                    df = carregar_dados(); print("Dados recarregados:", len(df), "registros.")
            elif op == "2":
                consultar_registros(df)
            elif op == "3":
                indicadores_e_erros(df)
            elif op == "4":
                simulacao_euler(df)
            elif op == "5":
                res_modelo = modelo_e_metricas(df)
            elif op == "6":
                priorizar_alertas(df); comparar_heap_com_lista()
            elif op == "7":
                busca_por_prefixo(df)
            elif op == "8":
                dispositivos_bases_eletricidade(df)
            elif op == "9":
                gerenciamento_inteligente(df)
            elif op == "10":
                analise_final(df, res_modelo)
            elif op == "0":
                print("Encerrando o SCIC. Até a próxima missão!")
                break
            else:
                print("Opção inválida. Digite um número do menu.")
        except (ValueError, KeyError) as erro:
            print("Entrada inválida:", erro)
        input("\nPressione ENTER para voltar ao menu...")


if __name__ == "__main__":
    main()

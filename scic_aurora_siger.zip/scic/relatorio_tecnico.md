# Relatório Técnico – SCIC (Aurora Siger)

## 1. Contexto da solução
A Aurora Siger é uma colônia com módulos de habitação, agricultura, comunicação, laboratório,
suporte médico e armazenamento de dados. A comunicação entre esses módulos precisa ser confiável e
com baixa latência. O SCIC é um protótipo que (a) organiza os dados de operação, (b) mede o quanto
as estimativas de latência erram, (c) prevê a latência com um modelo simples, (d) decide qual alerta
atender primeiro e (e) permite localizar módulos, sensores e comandos rapidamente.

## 2. Descrição dos dados
Base **simulada** (`dados_aurora_siger.csv`), gerada por `gerar_dados.py` com semente fixa:
10 módulos x 8 ciclos = **80 registros**, 13 colunas. A latência observada depende do tipo do módulo,
da carga da rede e da corrente, com ruído e picos raros. Dois módulos (Comunicação Reserva e
Agricultura Hidropônica) têm uma piora lenta embutida, para permitir a análise de manutenção
preditiva. A latência prevista é uma estimativa simples do operador, que não conhece essa piora.
Distribuição de status: 66 ativos, 10 alertas, 4 em manutenção. O código do sensor é hexadecimal
de 4 dígitos: o primeiro identifica o tipo do módulo e os três seguintes o número do sensor.
Limitação: por ser simulada, a base reflete as regras que nós mesmos definimos.

## 3. Indicadores e análise de erros
- Disponibilidade (registros ativos): **82,5%**; registros em alerta: **12,5%**; latência média: **95,3 ms**.
- Erro absoluto = |previsto − observado| (ms). Erro relativo = erro absoluto / observado (%).
  O relativo permite comparar módulos com escalas de latência diferentes.
- Critério adotado: **< 15% aceitável; 15–30% atenção; > 30% preocupante**. Em uma colônia, erros acima
  de 30% indicam que a estimativa não serve para planejar rotas de comunicação nem consumo de energia.
- Piores módulos (erro relativo médio): Comunicação Reserva (21,1%) e Agricultura Hidropônica (17,3%),
  ambos "atenção" e justamente os que têm piora lenta. Melhor: Habitação Alfa (2,0%).
  Maior erro individual: Habitação Beta, ciclo 4 (46,7 ms, 35,7%), um pico.
- **Ponto flutuante:** `0.1 + 0.2` resulta em `0.30000000000000004`, pois 0,1 e 0,2 não têm representação
  binária exata. Por isso usamos `math.isclose` para comparar e `round` para exibir.
- **Método de Euler:** simulamos a volta da latência do pico (150,7 ms, Agricultura, ciclo 8) ao valor
  normal (113,1 ms) com dL/dt = −k(L − L_alvo), k = 0,5, h = 1. Comparado à solução exata, o erro relativo
  máximo foi 3,5% (erro de truncamento). Passo menor reduz o erro.

## 4. Modelo simples e métricas
Regressão (Ridge, alpha = 0,1, escolhido pela validação) para prever a latência observada a partir de
corrente, carga da rede e tipo do módulo. Divisão **60/20/20** (48 treino, 16 validação, 16 teste).
Comparamos baseline (média), regressão linear e Ridge com quatro valores de alpha (grid search manual).

| Validação | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (média) | 13,28 | 16,78 | −0,003 |
| Regressão linear | 6,57 | 7,66 | 0,791 |
| Ridge (alpha=0,1) | 6,58 | 7,60 | 0,794 |

**Teste final:** modelo MAE = 7,83 ms, MSE = 118,30, RMSE = 10,88 ms, R² = 0,645.
Estimativa do operador no mesmo teste: MAE = 9,03, RMSE = 14,42, R² = 0,375.
AIC/BIC (treino): linear 252,2/269,0 contra baseline 287,3/289,2 (menor é melhor).

**Interpretação.** O modelo supera a estimativa do operador, mas um único número não basta:
(i) RMSE/MAE = 1,39, ou seja, há erros maiores concentrados em poucos registros (picos);
(ii) R² caiu de ~0,79 na validação para 0,645 no teste, sinal de que, com apenas 16 registros por
conjunto, as métricas variam bastante; (iii) R² alto não garante modelo perfeito, só indica quanto da
variação foi explicada. Os picos aleatórios não são previsíveis por esse modelo.

## 5. Heap: priorização de alertas
- **Representação:** cada alerta é a tupla `(-score, ciclo, módulo, mensagem)` dentro de uma lista que
  funciona como heap binária (pai = (i−1)//2; filhos = 2i+1 e 2i+2), com `heapify-up` na inserção e
  `heapify-down` na remoção.
- **Critério:** score = prioridade × 30 + 20 (módulo essencial) + 0,5 × erro relativo (%) + 15 (latência ≥ 140 ms).
- **Seleção:** como a heap é de mínimo e guardamos −score, o topo é o alerta mais urgente. Empate: o mais antigo.
- **Resultado:** 10 alertas; o primeiro a atender é **Armazenamento de Dados (ciclo 1, score 125,1)**,
  seguido de Comando Central (ciclo 2, score 123,9). Ambos são essenciais.
- **Vantagem sobre lista simples:** com 1000 alertas, esvaziar a heap exigiu 17.175 comparações contra
  499.500 da lista simples (≈ n log n contra n²/2). O mais urgente está sempre no topo (O(1)).

## 6. Trie: busca por prefixo
Árvore em que cada nível é uma letra (sem acento e sem diferenciar maiúsculas). Palavras com o mesmo
início compartilham o caminho, então o custo da busca depende do tamanho do prefixo, não do número de
registros. Exemplos: "com" retorna Comando Central, Comunicação Principal e Comunicação Reserva;
"30" retorna os sensores 30A0, 30B1 e 30C2 (todos de comunicação); "co" em comandos retorna
`comando_reiniciar_enlace`, `comunicar_base`, `consultar_status` e `controle_energia_reduzir`.

## 7. Dispositivos, bases numéricas e eletricidade
- **Entrada:** medidores de tensão/corrente e sensores de latência e carga (simulados). **Saída:** terminal,
  relatório e gráficos PNG (em campo, um painel). **Interfaces (conceitual):** rede/Wi-Fi, USB, Bluetooth.
- **Bases:** o sensor `0x30A0` = 12448 em decimal = `0011000010100000` em binário. Com deslocamento de bits
  (`>> 12`) obtemos o tipo (3 = comunicação) e com máscara (`& 0xFFF`) o número do sensor (0x0A0 = 160).
- **Eletricidade:** P = V × I e R = V / I. O transmissor da Comunicação Principal (24 V × 3,86 A) consome
  92,64 W; o Controle de Antenas (48 V) é o maior consumidor, com 191,52 W. A potência média somada dos
  módulos é 742,6 W. Reduzir consumo da comunicação libera energia para habitação e suporte médico.

## 8. Gerenciamento inteligente da comunicação
- **Sensores e medidores inteligentes** produzem a tensão, a corrente, a carga e a latência que alimentam o sistema.
- **Monitoramento contínuo:** o erro relativo entre previsto e observado denuncia anomalias.
- **Automação:** a heap diz qual alerta é mais urgente e regras podem acionar `chavear_reserva`, sempre com
  validação humana.
- **Redundância e armazenamento:** a Comunicação Principal esteve fora de operação em 1 ciclo (ciclo 2) e a
  reserva estava ativa nele, então o chaveamento era possível. O histórico por ciclo permite auditar as falhas.
- **Manutenção preditiva:** a reta ajustada à latência mostra piora de ~4,2 ms/ciclo na Agricultura
  Hidropônica e ~3,6 ms/ciclo na Comunicação Reserva; projeta-se 163,3 ms para a Agricultura em quatro ciclos.
  O Controle de Antenas (4,4 ms/ciclo) também foi sinalizado, mas por causa de um pico no último ciclo;
  é um alerta para inspeção, não uma conclusão de defeito.
- **Redes inteligentes e microrredes:** o consumo por módulo (P = V × I) pode ser balanceado entre módulos
  em horários de maior carga de comunicação.

## 9. Reflexão social, cultural e sustentável
1. **Sustentabilidade:** comunicar com eficiência reduz retransmissões e consumo elétrico. Cada watt a menos
   nos transmissores sobra para sistemas vitais e reduz o desperdício de recursos da colônia.
2. **Conhecimentos indígenas:** muitos povos indígenas planejam o uso de recursos pensando em ciclos e em
   regeneração. O mesmo princípio inspira o sistema: preferir manutenção preventiva e consumo moderado a esgotar
   um módulo até a falha.
3. **Diversidade e viés:** o score de prioridade favorece módulos "essenciais". Quem define o que é essencial
   deve ouvir todas as comunidades da colônia, para que habitação, saúde e agricultura não sejam preteridas por
   critérios técnicos que ignorem necessidades humanas. Mensagens de alerta devem usar linguagem neutra e respeitosa.
4. **Transparência e responsabilidade humana:** os critérios (score, limites de erro) estão explícitos no código
   e neste relatório, e a decisão final de agir sobre um alerta é sempre da equipe humana.

## 10. Limitações e melhorias
- Base pequena e simulada: as métricas variam com a divisão dos dados; usar mais ciclos e validação cruzada.
- Modelo linear simples: não captura picos; testar modelos não lineares e mais variáveis (temperatura, distância do enlace).
- Pesos do score definidos manualmente: calibrar com a equipe de operação.
- Heap e trie em memória: persistir os alertas em arquivo para recuperar o estado entre execuções.
- Dados reais por telemetria e painel gráfico (fora do escopo desta fase).

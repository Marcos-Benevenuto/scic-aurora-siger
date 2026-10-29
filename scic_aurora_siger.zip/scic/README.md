# SCIC – Sistema de Comunicação Interplanetária da Colônia (Aurora Siger)

Protótipo em Python que organiza dados operacionais e de comunicação da colônia Aurora Siger,
avalia a confiabilidade de previsões de latência, prioriza alertas com **heap** e permite
buscas por prefixo com **trie**.

## Arquivos da entrega
| Arquivo | Descrição |
|---|---|
| `codigo_fonte.py` | Sistema principal (menu no terminal) |
| `dados_aurora_siger.csv` | Base simulada: 10 módulos x 8 ciclos = 80 registros |
| `gerar_dados.py` | Script (extra) que gerou a base com semente fixa |
| `relatorio_tecnico.md` | Relatório técnico completo |
| `link_video.txt` | Link do vídeo (YouTube, "Não listado") |
| `graficos_ou_imagens/` | Gráficos gerados na execução |

## Dependências
Python 3.9+ e as bibliotecas estudadas na fase: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
Nenhuma biblioteca adicional é necessária.

```
pip install numpy pandas matplotlib scikit-learn
```

## Como executar
```
python codigo_fonte.py          # menu interativo
python codigo_fonte.py --auto   # roda todas as etapas em sequência (demonstração)
```
Execute dentro da pasta do projeto. O CSV é lido da mesma pasta do código.

## Menu
1. Carregar dados / cadastrar novo registro (o cadastro grava no CSV)
2. Consultar registros (filtro por módulo, ciclo ou status)
3. Indicadores e erros numéricos (absoluto e relativo, ponto flutuante)
4. Simulação da latência (método de Euler)
5. Previsão simples e métricas (MAE, MSE, RMSE, R², AIC, BIC)
6. Priorizar alertas (heap) e comparação heap x lista
7. Busca por prefixo (trie): módulos, sensores, comandos, palavras de alerta
8. Dispositivos, bases numéricas e eletricidade (P = V x I, lei de Ohm)
9. Gerenciamento inteligente (manutenção preditiva e redundância)
10. Análise final (gera os gráficos em `graficos_ou_imagens/`)

## Colunas do CSV
`ciclo, modulo, tipo, codigo_sensor (hex), latencia_observada_ms, latencia_prevista_ms,
tensao_v, corrente_a, carga_rede_pct, essencial, status, prioridade, mensagem_alerta`

A heap e a trie foram implementadas manualmente (sem `heapq`) para demonstrar o funcionamento.

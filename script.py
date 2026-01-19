import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter

# Initialize connection
mt5.initialize()

symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M15  # 15 minutos
num_candles = 50  # Número de candles a analisar
num_tops_bottoms = 10
precision = 0.0010  # agrupar por esse intervalo de preço

# Ensure symbol is available
mt5.symbol_select(symbol, True)

# Get 5 candles
rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)

mt5.symbol_select(symbol, True)

# Verificar se recebeu dados
if rates is None or len(rates) == 0:
    print("Nenhum dado retornado.")
    mt5.shutdown()
    quit()

# Convert to DataFrame
df = pd.DataFrame(rates)

# Convert open time to datetime
df['open_time'] = pd.to_datetime(df['time'], unit='s')

# Calculate close time (open_time + timeframe duration)
# For 1-minute candles, add 1 minute
df['close_time'] = df['open_time'] + timedelta(minutes=1)

print(df[['open_time', 'close_time', 'open', 'close']])

# Recuperar a máxima e a mínima do período
maxima = df['high'].max()
minima = df['low'].min()

print(f"Máxima (High) nas últimas {num_candles} velas: {maxima}")
print(f"Mínima (Low) nas últimas {num_candles} velas: {minima}")

# Ordenar por high e low para pegar os N maiores e menores valores
top_highs = df.sort_values(by='high', ascending=False).head(num_tops_bottoms)[['time', 'high']]
bottom_lows = df.sort_values(by='low', ascending=True).head(num_tops_bottoms)[['time', 'low']]

print(f"Top {num_tops_bottoms} máximas:")
print(top_highs)

print(f"\nTop {num_tops_bottoms} mínimas:")
print(bottom_lows)


# Arredonda máximas para o "bin" mais próximo (ex: 1.097 -> 1.0970)
df['high_grouped'] = (df['high'] // precision) * precision

# Contar frequência
group_counts = Counter(df['high_grouped'])

# Obter as N faixas com mais ocorrências
mais_agrupadas = group_counts.most_common(num_tops_bottoms)

# Transformar em lista formatada
resultados = [{'preco': round(preco, 5), 'toques': count} for preco, count in mais_agrupadas]

# Mostrar resultado
print("Zonas de máximas mais agrupadas:")
for item in resultados:
    print(f"Preço: {item['preco']} | Toques: {item['toques']}")

# # Plotar scatter plot
# plt.figure(figsize=(12, 6))
# plt.scatter(df['time'], df['high'], color='red', label='Máximas', s=10, alpha=0.6)
# plt.scatter(df['time'], df['low'], color='blue', label='Mínimas', s=10, alpha=0.6)
# plt.scatter(df['time'], df['close'], color='gray', label='Fechamento', s=5, alpha=0.4)
#
# plt.title(f"Nuvem de pontos - {symbol} ({num_candles} candles)")
# plt.xlabel("Tempo")
# plt.ylabel("Preço")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Shutdown connection
mt5.shutdown()

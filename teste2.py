import MetaTrader5 as mt5
import pandas as pd
import time

from datetime import datetime
import pytz  # Para lidar com fusos horários, importante para dados de trading

# --- 1. Configuração Inicial ---
# ATENÇÃO: Este script é um exemplo conceitual e não deve ser usado em uma conta de trading real
# sem testes exaustivos em uma conta demo e uma compreensão completa dos riscos envolvidos.
# A detecção de divergência é simplificada e a marcação no gráfico é conceitual.

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M30
LOT_SIZE = 0.01  # Tamanho do lote para cada operação
MAGIC_NUMBER = 123456  # Número mágico para identificar as ordens do seu robô
RISK_PERCENT = 0.01  # 1% de risco por operação (para cálculo de SL)
TP_RR_RATIO = 2.0  # Relação Risco-Recompensa de 2:1

# Parâmetros da Estratégia "Reversão à Média com Divergência"
BB_PERIOD = 20
BB_DEVIATIONS = 2.0
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
MACD_FAST_EMA = 12
MACD_SLOW_EMA = 26
MACD_SIGNAL_SMA = 9
ATR_PERIOD = 14

# --- Conectar ao MetaTrader 5 ---
# Certifique-se de que o MetaTrader 5 está aberto e logado.
# Instale a biblioteca MetaTrader5: pip install MetaTrader5
# Instale a biblioteca pandas: pip install pandas
# Instale a biblioteca pandas_ta: pip install pandas_ta
# Instale a biblioteca pytz: pip install pytz

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    # Tente encontrar o terminal MT5 se não inicializar
    # mt5.initialize(path="C:\\Program Files\\MetaTrader 5\\terminal64.exe") # Substitua pelo seu caminho
    quit()

print("Conectado ao MetaTrader 5")

# Configurar o fuso horário para o MetaTrader (geralmente UTC)
timezone = pytz.timezone("Etc/UTC")

# Configurar o símbolo
selected = mt5.symbol_select(SYMBOL, True)
if not selected:
    print(f"Falha ao selecionar {SYMBOL}. Verifique se o símbolo está disponível no seu Market Watch.")
    mt5.shutdown()
    quit()


# --- 2. Funções de Indicadores ---
def calculate_indicators(data_frame):
    """Calcula os indicadores técnicos necessários para a estratégia."""
    # Bandas de Bollinger
    # O pandas_ta adiciona as colunas diretamente ao DataFrame
    bandas = calcular_bollinger_bands(precos=data_frame['close'], periodo=BB_PERIOD, desvio=BB_DEVIATIONS)

    # RSI
    data_frame.rsi(close=data_frame['close'], length=RSI_PERIOD, append=True)

    # MACD
    data_frame.macd(close=data_frame['close'], fast=MACD_FAST_EMA, slow=MACD_SLOW_EMA, signal=MACD_SIGNAL_SMA,
                       append=True)

    # ATR
    data_frame.atr(high=data_frame['high'], low=data_frame['low'], close=data_frame['close'], length=ATR_PERIOD,
                      append=True)

    return data_frame

def calcular_bollinger_bands(precos, periodo=20, desvio=2):
    """
    Calcula as Bandas de Bollinger.

    Parâmetros:
    - precos (pd.Series): Série com os preços (fechamento, por exemplo).
    - periodo (int): Período da média móvel e do desvio padrão.
    - desvio (float): Multiplicador do desvio padrão.

    Retorna:
    - pd.DataFrame com as colunas: ['sma', 'upper', 'lower']
    """
    sma = precos.rolling(window=periodo).mean().dropna()
    std = precos.rolling(window=periodo).std().dropna()
    upper = sma + (desvio * std)
    lower = sma - (desvio * std)

    bandas = pd.DataFrame({
        'sma': sma,
        'upper': upper,
        'lower': lower
    }, index=precos.index)

    return bandas.dropna()


# --- 3. Função de Detecção de Divergência (Simplificada/Placeholder) ---
# ATENÇÃO: Esta é uma implementação MUITO simplificada e NÃO ROBUSTA de detecção de divergência.
# A detecção de divergência real é complexa e envolve a identificação de picos e vales (swing points)
# no preço e no indicador, e a comparação de suas tendências.
# Para um robô de trading real, esta lógica precisaria ser muito mais sofisticada,
# possivelmente usando algoritmos de detecção de picos/vales ou bibliotecas de análise técnica mais avançadas.
def detect_divergence(data_frame):
    """
    Detecta divergência de alta ou baixa entre o preço e o MACD.
    Esta é uma implementação simplificada para fins de demonstração.
    """
    # Precisamos de dados suficientes para que os indicadores sejam calculados e para análise de padrões
    if len(data_frame) < max(BB_PERIOD, RSI_PERIOD, MACD_SLOW_EMA, ATR_PERIOD) + 5:
        return {'bullish': False, 'bearish': False}

    # Pegar os últimos candles para análise
    # A análise de divergência geralmente olha para os últimos 2-3 swing points
    # Aqui, estamos simplificando para os últimos 5 candles para ilustrar o conceito.
    recent_data = data_frame.iloc[-5:]

    # Condições para divergência de alta (preço faz mínima mais baixa, MACD faz mínima mais alta)
    # Exemplo simplificado:
    bullish_divergence = False
    # Certifique-se de que a coluna do MACD existe antes de tentar acessá-la
    macd_col_name = f'MACD_{MACD_FAST_EMA}_{MACD_SLOW_EMA}_{MACD_SIGNAL_SMA}'
    if macd_col_name in recent_data.columns and len(recent_data) >= 2:
        if recent_data['close'].iloc[-1] < recent_data['close'].iloc[-2] and \
                recent_data[macd_col_name].iloc[-1] > recent_data[macd_col_name].iloc[-2]:
            bullish_divergence = True

    # Condições para divergência de baixa (preço faz máxima mais alta, MACD faz máxima mais baixa)
    # Exemplo simplificado:
    bearish_divergence = False
    if macd_col_name in recent_data.columns and len(recent_data) >= 2:
        if recent_data['close'].iloc[-1] > recent_data['close'].iloc[-2] and \
                recent_data[macd_col_name].iloc[-1] < recent_data[macd_col_name].iloc[-2]:
            bearish_divergence = True

    return {'bullish': bullish_divergence, 'bearish': bearish_divergence}


# --- 4. Funções de Gerenciamento de Ordens ---
def send_order(symbol, order_type, lot, price, sl, tp, magic, comment):
    """Envia uma ordem de mercado com SL e TP."""
    point = mt5.symbol_info(symbol).point
    deviation = 20  # Desvio máximo permitido do preço em pontos

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,  # Good Till Cancel
        "type_filling": mt5.ORDER_FILLING_FOC,  # Fill Or Kill
    }

    # Adicionar SL e TP se forem válidos (não zero)
    # O MetaTrader5 espera que SL e TP sejam preços absolutos, não diferenciais.
    # Certifique-se de que os preços de SL/TP estão corretos e que o SL não está entre o preço e o TP para ordens de mercado.
    if sl > 0:
        request["sl"] = sl
    if tp > 0:
        request["tp"] = tp

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Falha ao enviar ordem: {result.comment}, retcode={result.retcode}")
        print(f"Request: {request}")
    else:
        print(
            f"Ordem enviada com sucesso! Ticket: {result.order}, Tipo: {order_type_to_string(order_type)}, Volume: {lot}")
    return result


def close_position(ticket, symbol, volume, position_type):
    """Fecha uma posição aberta."""
    # Obter o preço atual para fechar a ordem
    if position_type == mt5.POSITION_TYPE_BUY: # Corrigido de ORDER_TYPE_BUY para POSITION_TYPE_BUY
        close_price = mt5.symbol_info_tick(symbol).bid
    else:  # mt5.POSITION_TYPE_SELL # Corrigido de ORDER_TYPE_SELL para POSITION_TYPE_SELL
        close_price = mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL if position_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY, # Tipo de ordem inversa para fechar
        "position": ticket,
        "price": close_price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "close by robot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Falha ao fechar posição {ticket}: {result.comment}, retcode={result.retcode}")
    else:
        print(f"Posição {ticket} fechada com sucesso!")
    return result


def get_open_positions(magic_number=None):
    """Retorna uma lista de posições abertas pelo robô."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []

    filtered_positions = []
    for pos in positions:
        if magic_number is None or pos.magic == magic_number:
            filtered_positions.append(pos)
    return filtered_positions


def order_type_to_string(order_type):
    """Converte o tipo de ordem MT5 para string legível."""
    if order_type == mt5.ORDER_TYPE_BUY:
        return "BUY"
    elif order_type == mt5.ORDER_TYPE_SELL:
        return "SELL"
    # Para tipos de posição (POSITION_TYPE_BUY/SELL)
    elif order_type == mt5.POSITION_TYPE_BUY:
        return "BUY_POSITION"
    elif order_type == mt5.POSITION_TYPE_SELL:
        return "SELL_POSITION"
    else:
        return str(order_type)


# --- 5. Loop Principal de Trading ---
def trading_loop():
    print("\nIniciando loop de trading...")
    last_checked_candle_time = None

    while True:
        # Obter os últimos candles
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 200)
        if rates is None or len(rates) == 0:
            print("Nenhum dado de preço recebido. Tentando novamente em 30 segundos.")
            time.sleep(30)
            continue

        # Converter para DataFrame
        data_frame = pd.DataFrame(rates)
        data_frame['time'] = pd.to_datetime(data_frame['time'], unit='s')
        data_frame.set_index('time', inplace=True)
        # Renomear colunas para minúsculas para compatibilidade com pandas_ta
        data_frame.columns = [col.lower() for col in data_frame.columns]

        # Verificar se um novo candle fechou
        current_candle_time = data_frame.index[-1]
        if current_candle_time == last_checked_candle_time:
            time.sleep(10)
            continue

        last_checked_candle_time = current_candle_time
        print(f"\nAnalisando novo candle fechado: {current_candle_time}")

        # Calcular indicadores
        data_frame = calculate_indicators(data_frame)

        # Pegar os dados do último candle fechado e do candle anterior
        current_candle = data_frame.iloc[-1]
        previous_candle = data_frame.iloc[-2] # Acessa o penúltimo candle, que é o fechado anterior

        # Obter informações do símbolo para cálculo de preço e stop loss
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            print(f"Falha ao obter informações do símbolo {SYMBOL}")
            time.sleep(30)
            continue

        # Obter preços de bid e ask em tempo real para execução
        tick_info = mt5.symbol_info_tick(SYMBOL)
        ask_price = tick_info.ask
        bid_price = tick_info.bid

        # Verificar posições abertas pelo robô
        open_positions = get_open_positions(MAGIC_NUMBER)

        # --- Lógica da Estratégia "Reversão à Média com Divergência" ---
        # Apenas operar se não houver posições abertas para evitar múltiplas entradas
        if not open_positions:
            # Detecção de Divergência (Placeholder - Lógica simplificada)
            divergence_signals = detect_divergence(data_frame)

            # Condições de Compra
            buy_signal = False
            # Verificar se as colunas dos indicadores existem antes de acessá-las
            bb_lower_col = f'BBL_{BB_PERIOD}_{BB_DEVIATIONS}.0'
            rsi_col = f'RSI_{RSI_PERIOD}'
            macd_col = f'MACD_{MACD_FAST_EMA}_{MACD_SLOW_EMA}_{MACD_SIGNAL_SMA}'

            # Garantir que os dados dos candles anteriores e atuais estão disponíveis e não são NaN
            if bb_lower_col in data_frame.columns and rsi_col in data_frame.columns and macd_col in data_frame.columns:
                if not pd.isna(previous_candle[bb_lower_col]) and \
                   not pd.isna(previous_candle[rsi_col]) and \
                   not pd.isna(previous_candle[macd_col]):

                    if previous_candle['close'] < previous_candle[bb_lower_col] and \
                       previous_candle[rsi_col] < RSI_OVERSOLD and \
                       divergence_signals['bullish'] and \
                       current_candle['close'] > previous_candle[bb_lower_col]:
                        buy_signal = True

            # Condições de Venda
            sell_signal = False
            bb_upper_col = f'BBU_{BB_PERIOD}_{BB_DEVIATIONS}.0'

            if bb_upper_col in data_frame.columns and rsi_col in data_frame.columns and macd_col in data_frame.columns:
                if not pd.isna(previous_candle[bb_upper_col]) and \
                   not pd.isna(previous_candle[rsi_col]) and \
                   not pd.isna(previous_candle[macd_col]):

                    if previous_candle['close'] > previous_candle[bb_upper_col] and \
                       previous_candle[rsi_col] > RSI_OVERBOUGHT and \
                       divergence_signals['bearish'] and \
                       current_candle['close'] < previous_candle[bb_upper_col]:
                        sell_signal = True

            # Executar Ordem de Compra
            if buy_signal:
                print(f"Sinal de COMPRA detectado em {current_candle.name}!")
                # Calcular Stop Loss e Take Profit
                # O pandas_ta usa 'atr' em minúsculas
                atr_val = current_candle['atr'] if 'atr' in current_candle else (current_candle['high'] - current_candle['low']) * 0.5 # Fallback se ATR não calculado
                sl_price = current_candle['low'] - (atr_val * 1.5)
                tp_price = ask_price + (abs(ask_price - sl_price) * TP_RR_RATIO)

                print(f"Tentando comprar {LOT_SIZE} lotes de {SYMBOL} @ {ask_price:.5f} SL: {sl_price:.5f} TP: {tp_price:.5f}")
                send_order(SYMBOL, mt5.ORDER_TYPE_BUY, LOT_SIZE, ask_price, sl_price, tp_price, MAGIC_NUMBER, "BUY_DIVERGENCE")

            # Executar Ordem de Venda
            elif sell_signal:
                print(f"Sinal de VENDA detectado em {current_candle.name}!")
                atr_val = current_candle['atr'] if 'atr' in current_candle else (current_candle['high'] - current_candle['low']) * 0.5 # Fallback se ATR não calculado
                sl_price = current_candle['high'] + (atr_val * 1.5)
                tp_price = bid_price - (abs(sl_price - bid_price) * TP_RR_RATIO)

                print(f"Tentando vender {LOT_SIZE} lotes de {SYMBOL} @ {bid_price:.5f} SL: {sl_price:.5f} TP: {tp_price:.5f}")
                send_order(SYMBOL, mt5.ORDER_TYPE_SELL, LOT_SIZE, bid_price, sl_price, tp_price, MAGIC_NUMBER, "SELL_DIVERGENCE")
        else:
            for position in open_positions:
                print(f"Posição aberta: Ticket={position.ticket}, Tipo={order_type_to_string(position.type)}, Volume={position.volume}, Preço Abertura={position.price_open:.5f}, SL={position.sl:.5f}, TP={position.tp:.5f}")

        time.sleep(30)


# --- Executar o Robô ---
if __name__ == "__main__":
    try:
        trading_loop()
    except KeyboardInterrupt:
        print("\nRobô interrompido pelo usuário.")
    finally:
        mt5.shutdown()
        print("Conexão com MetaTrader 5 encerrada.")

import time
from collections import Counter, defaultdict
from datetime import timedelta, datetime, date
from enum import Enum

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from scipy.stats import linregress


class TipoPadrao(Enum):
    MARTELO = 'Martelo'
    ENFORCADO = 'Enforcado'
    DOJI = 'Doji'
    ENGOLFO_ALTA = 'Engolfo de Alta'
    ENGOLFO_BAIXA = 'Engolfo de Baixa'
    ESTRELA_MANHA = 'Estrela da Manhã'
    ESTRELA_NOITE = 'Estrela da Noite'
    INDEFINIDO = 'INDEFINIDO'


class TipoSinal(Enum):
    COMPRA = 'COMPRA'
    VENDA = 'VENDA'
    INDEFINIDO = 'INDEFINIDO'


timeframe_to_minutes = {
    # Minutos
    mt5.TIMEFRAME_M1: 1,  # TIMEFRAME_M1
    mt5.TIMEFRAME_M2: 2,  # TIMEFRAME_M2
    mt5.TIMEFRAME_M3: 3,  # TIMEFRAME_M3
    mt5.TIMEFRAME_M4: 4,  # TIMEFRAME_M4
    mt5.TIMEFRAME_M5: 5,  # TIMEFRAME_M5
    mt5.TIMEFRAME_M6: 6,  # TIMEFRAME_M6
    mt5.TIMEFRAME_M10: 10,  # TIMEFRAME_M10
    mt5.TIMEFRAME_M12: 12,  # TIMEFRAME_M12
    mt5.TIMEFRAME_M15: 15,  # TIMEFRAME_M15
    mt5.TIMEFRAME_M20: 20,  # TIMEFRAME_M20
    mt5.TIMEFRAME_M30: 30,  # TIMEFRAME_M30

    # Horas (usando bitwise OR com 0x4000 = 16384)
    mt5.TIMEFRAME_H1: 60,  # TIMEFRAME_H1  (1 hour)
    mt5.TIMEFRAME_H2: 120,  # TIMEFRAME_H2  (2 hours)
    mt5.TIMEFRAME_H3: 180,  # TIMEFRAME_H3  (3 hours)
    mt5.TIMEFRAME_H4: 240,  # TIMEFRAME_H4  (4 hours)
    mt5.TIMEFRAME_H6: 360,  # TIMEFRAME_H6  (6 hours)
    mt5.TIMEFRAME_H8: 480,  # TIMEFRAME_H8  (8 hours)
    mt5.TIMEFRAME_H12: 720,  # TIMEFRAME_H12 (12 hours)
    mt5.TIMEFRAME_D1: 1440,  # TIMEFRAME_D1  (24 hours / 1 day)

    # Semana (0x8000 = 32768)
    mt5.TIMEFRAME_W1: 10080,  # TIMEFRAME_W1 (7 days * 24h * 60m)

    # Mês (0xC000 = 49152)
    mt5.TIMEFRAME_MN1: 43200,  # TIMEFRAME_MN1 (30 days * 24h * 60m)
}

NUM_CANDLES_DEFAULT = 3


def normalizar_data(data_entrada, formato):
    if isinstance(data_entrada, str):
        return datetime.strptime(data_entrada, formato).date()
    elif isinstance(data_entrada, datetime):
        return data_entrada.date()
    elif isinstance(data_entrada, date):
        return data_entrada
    else:
        raise ValueError("Formato de data inválido.")


class CandleAnalyzer:
    def __init__(self, login: int, senha: str, servidor: str,
                 symbol: str, timeframe, volume, enableMartingale=True,
                 enableBollinger=True, enableMACD=True, enableCCI=True,
                 enableRSI=True, enableMoveStop=True, stopMinimo=200,
                 enableADX=True,
                 enablePatternAnalysis=True, enableVerifyDivergence=True, maximoPerdaDiaria=0,
                 percentRiskAndProfit=100, magic=10001, step: float = 0.001):
        """
        Inicializa o analisador de candles.

        :param symbol: Símbolo do ativo (ex: "XAUUSD")
        :param timeframe: Timeframe (ex: mt5.TIMEFRAME_M15)
        :param step: Precisão de agrupamento dos preços
        """
        self.login = login
        self.senha = senha
        self.servidor = servidor
        self.symbol = symbol
        self.timeframe = timeframe
        self.volume = volume
        self.magic = magic
        self.step = step
        self.enableMartingale = enableMartingale
        self.maximoPerdaDiaria = maximoPerdaDiaria
        self.enableBollinger = enableBollinger
        self.enableMACD = enableMACD
        self.enableCCI = enableCCI
        self.enableADX = enableADX
        self.enableRSI = enableRSI
        self.enableMoveStop = enableMoveStop
        self.stopMinimo = stopMinimo
        self.percentRiskAndProfit = percentRiskAndProfit
        self.enablePatternAnalysis = enablePatternAnalysis
        self.enableVerifyDivergence = enableVerifyDivergence
        self.account = self.conectar_mt5(self.login, self.senha, self.servidor)

    def conectar_mt5(self, login: int, senha: str, servidor: str, caminho_terminal: str = None):
        if caminho_terminal:
            mt5.initialize(path=caminho_terminal)
        else:
            mt5.initialize()

        if not mt5.initialize():
            raise RuntimeError("Não foi possível inicializar o MetaTrader 5")

        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"Não foi possível selecionar o símbolo {self.symbol}")

        logado = mt5.login(login, password=senha, server=servidor)

        if logado:
            print("✅ Conectado com sucesso!")
            conta = mt5.account_info()
            print(f"Conta: {conta.login} | Saldo: {conta.balance}")
            return conta
        else:
            print(f"❌ Falha ao conectar: {mt5.last_error()}")
            return None

    def get_price_zones(self, df, show_debug: bool = False):
        """
        Retorna uma lista de tuplas com mínimos e máximos agrupados.

        :param df: dataframe com os dados do grafico
        :param show_debug: Se True, exibe detalhes do DataFrame
        :return: Lista de tuplas (min_group, max_group)
        """
        df['open_time'] = pd.to_datetime(df['time'], unit='s')
        df['close_time'] = df['open_time'] + timedelta(minutes=15)  # Ajustável por timeframe

        # Arredondar e remover casas decimais (inteiros)
        df['high_group'] = ((df['high'] / self.step).round() * self.step).astype(int)
        df['low_group'] = ((df['low'] / self.step).round() * self.step).astype(int)

        if show_debug:
            print(df[['open_time', 'low', 'high', 'low_group', 'high_group']])

        # Criar tuplas de low e high
        grupos = list(zip(df['low_group'].tolist(), df['high_group'].tolist()))

        # Remover duplicatas corretamente com set de tuplas inteiras
        grupos_unicos = sorted(set(grupos))
        return grupos_unicos

    def recuperarInfoGrafico(self, num_candles: int, start_date=0, end_date=0):
        """
        Agrupa os candles em blocos de `group_size` e retorna high/low de cada grupo.
             :param end_date: Data fim da verificacao dos graficos
             :param start_date: Data de inicio da verificacao dos graficos
             :param num_candles: Quantidade total de candles a buscar
        """
        if not mt5.initialize():
            raise RuntimeError("Erro ao iniciar MT5")

        if end_date != 0 and start_date != 0:
            rates = mt5.copy_rates_range(self.symbol, self.timeframe, start_date, end_date)
        elif start_date != 0:
            rates = mt5.copy_rates_from(self.symbol, self.timeframe, start_date, num_candles)
        else:
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, num_candles)

        if rates is None or len(rates) == 0:
            raise ValueError("Nenhum dado retornado da API.")

        df = pd.DataFrame(rates)

        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Formata a data para pt-BR
        df['data_formatada'] = self.converterDataPtBr(df['time'].dt)
        return df

    def get_grouped_highs_lows(self, df, group_size: int, show_debug: bool = False):
        """
        Agrupa os candles em blocos de `group_size` e retorna high/low de cada grupo.

        :param group_size: Tamanho de cada grupo de candles
        :param show_debug: Se True, exibe detalhes de cada grupo
        :return: Lista de tuplas (min, max) por grupo
        """
        df['open_time'] = pd.to_datetime(df['time'], unit='s')

        # Divide o DataFrame em grupos de `group_size` candles
        grouped_zones = []

        for i in range(0, len(df), group_size):
            group = df.iloc[i:i + group_size]

            if group.empty:
                continue

            min_low = group['low'].min()
            max_high = group['high'].max()

            grouped_zones.append((int(min_low), int(max_high)))

            if show_debug:
                print(f"\nGrupo {i // group_size + 1}:")
                print(group[['open_time', 'low', 'high']])
                print(f"→ Min: {min_low:.2f} | Max: {max_high:.2f}")

        return grouped_zones

    def contar_low_high_separado(self, tuplas, tolerancia_pontos: int, show_debug: bool = False):
        lows = [low for low, _ in tuplas]
        highs = [high for _, high in tuplas]

        contagem_low = []
        contagem_high = []

        tolerancia = tolerancia_pontos / 100
        for i, val_low in enumerate(lows):
            count_low = sum(1 for x in lows if abs(x - val_low) <= tolerancia) - 1  # menos ele mesmo
            contagem_low.append((val_low, count_low))

        for i, val_high in enumerate(highs):
            count_high = sum(1 for x in highs if abs(x - val_high) <= tolerancia) - 1
            contagem_high.append((val_high, count_high))

        # Para mostrar contagem agregada por valor único, podemos agrupar com Counter
        low_agg = Counter()
        for val, cnt in contagem_low:
            low_agg[val] += cnt + 1  # +1 para incluir a própria ocorrência

        high_agg = Counter()
        for val, cnt in contagem_high:
            high_agg[val] += cnt + 1

        if show_debug:
            print("Contagem agregada de LOWs próximos (±3):")
            for val, count in sorted(low_agg.items()):
                print(f"Low {val}: aparece {count} vezes")

            print("\nContagem agregada de HIGHs próximos (±3):")
            for val, count in sorted(high_agg.items()):
                print(f"High {val}: aparece {count} vezes")

        return low_agg, high_agg

    def detectar_divergencia(self, numCandles=100, periodo_rsi=14):
        """
        Detecta divergências entre preço e RSI no MetaTrader 5.

        Parâmetros:
        - symbol: str (ex: 'EURUSD')
        - timeframe: mt5.TIMEFRAME_* (ex: mt5.TIMEFRAME_H1)
        - numCandles: int, número de candles
        - periodo_rsi: int, período do RSI

        Retorna:
        - DataFrame com divergências detectadas (se houver)
        """
        df = self.recuperarInfoGrafico(numCandles)

        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Calcular RSI
        delta = df['close'].diff()
        ganho = delta.where(delta > 0, 0)
        perda = -delta.where(delta < 0, 0)
        avg_gain = ganho.rolling(window=periodo_rsi).mean()
        avg_loss = perda.rolling(window=periodo_rsi).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df.dropna(inplace=True)

        divergencias = []

        for i in range(periodo_rsi, len(df) - 1):
            # Procurar fundos e topos locais
            preco_anterior = df['close'].iloc[i - 1]
            preco_atual = df['close'].iloc[i]
            preco_proximo = df['close'].iloc[i + 1]

            rsi_anterior = df['rsi'].iloc[i - 1]
            rsi_atual = df['rsi'].iloc[i]
            rsi_proximo = df['rsi'].iloc[i + 1]

            # Fundo local (mínimo entre os três)
            if preco_atual < preco_anterior and preco_atual < preco_proximo:
                # Verificar divergência de alta (fundo no preço com RSI mais alto)
                preco_anterior_fundo = df['close'].iloc[i - 2]
                rsi_anterior_fundo = df['rsi'].iloc[i - 2]
                if preco_atual < preco_anterior_fundo and rsi_atual > rsi_anterior_fundo:
                    divergencias.append({
                        'tipo': 'Divergência de Alta',
                        'data': self.converterDataPtBr(df.index[i]),
                        'preco': preco_atual,
                        'rsi': rsi_atual
                    })

            # Topo local (máximo entre os três)
            if preco_atual > preco_anterior and preco_atual > preco_proximo:
                # Verificar divergência de baixa (topo no preço com RSI mais baixo)
                preco_anterior_topo = df['close'].iloc[i - 2]
                rsi_anterior_topo = df['rsi'].iloc[i - 2]
                if preco_atual > preco_anterior_topo and rsi_atual < rsi_anterior_topo:
                    divergencias.append({
                        'tipo': 'Divergência de Baixa',
                        'data': self.converterDataPtBr(df.index[i]),
                        'preco': preco_atual,
                        'rsi': rsi_atual
                    })

        return pd.DataFrame(divergencias)

    def calcularBollinger(self, df, periodo=20, desvio=2, janela_stop=1):
        """
        Gera sinais de compra e venda com base nas Bandas de Bollinger, com stop e take definidos.

        Retorno:
            list: Lista de tuplas (data, close, sinal, stop, take)
        """
        df['media'] = df['close'].rolling(periodo).mean()
        df['std'] = df['close'].rolling(periodo).std()
        df['banda_superior'] = df['media'] + (desvio * df['std'])
        df['banda_inferior'] = df['media'] - (desvio * df['std'])

        df['sinal'] = np.where(
            df['close'] < df['banda_inferior'], TipoSinal.COMPRA.value,
            np.where(df['close'] > df['banda_superior'], TipoSinal.VENDA.value, TipoSinal.INDEFINIDO.value)
        )

        sinais = []
        for i, row in df.iterrows():
            if row['sinal'] == TipoSinal.INDEFINIDO.value:
                continue

            idx = df.index.get_loc(i)
            if idx < janela_stop:
                continue  # Não há dados suficientes para cálculo do stop

            janela = df.iloc[idx - janela_stop:idx]
            entrada = row['close']

            if row['sinal'] == TipoSinal.COMPRA.value:
                stop = janela['low'].min()
                take = entrada + abs(entrada - stop)
            elif row['sinal'] == TipoSinal.VENDA.value:
                stop = janela['high'].max()
                take = entrada - abs(stop - entrada)
            else:
                stop = None
                take = None

            # Substitui None por 0.0
            stop = 0.0 if stop is None else round(stop, 5)
            take = 0.0 if take is None else round(take, 5)

            if ((row['sinal'] == TipoSinal.COMPRA.value and entrada > stop) or
                    (row['sinal'] == TipoSinal.VENDA.value and entrada < stop)):

                for result in self.verificaResultadoPadraoDivergencia(df, row['time'], row['sinal']):
                    sinais.append((
                        row['time'],
                        entrada,
                        row['sinal'],
                        stop,
                        take,
                        "BOLLINGER"
                    ))

        return sinais

    def executarMartingale(self, sinais, pontos_entre_sinais):
        if self.enableMartingale == 0:
            return sinais

        novos_sinais = sinais
        for sinal in sinais:
            tempo, preco, tipo, stop, take, origem, tempoConvertido = sinal
            if 'MARTINGALE' not in origem:
                diferenca = self.calcularPontosGrafico(abs(stop - preco))
                while diferenca >= pontos_entre_sinais * 2:
                    diferenca = diferenca - pontos_entre_sinais
                    novo_preco = preco
                    if tipo == TipoSinal.COMPRA.value:
                        novo_preco = self.adicionarPontos(preco, -diferenca)
                    elif tipo == TipoSinal.VENDA.value:
                        novo_preco = self.adicionarPontos(preco, diferenca)
                    novo_sinal = (
                        tempo,  # mesmo timestamp
                        round(novo_preco, 2),
                        tipo,
                        round(stop, 2),
                        round(take, 2),
                        f"{origem}_MARTINGALE",
                        tempoConvertido
                    )
                    novos_sinais.append(novo_sinal)

        return novos_sinais

    def identificar_tendencia(self, df, n: int):
        """
        Identifica a tendência atual com base nos últimos n candles.

        Parâmetros:

            df (pd.DataFrame): DataFrame com colunas ['close'] ou ['time', 'close'].
            n (int): Número de candles a considerar (default = 20).

        Retorno:
            str: 'alta', 'baixa' ou 'lateral'.
        """

        if len(df) < n:
            raise ValueError(f"É necessário pelo menos {n} candles para análise.")

        # Seleciona os últimos n candles
        closes = df['close'].iloc[-n:].values
        x = np.arange(n)

        # Regressão linear para detectar inclinação da tendência
        slope, _, r_value, _, _ = linregress(x, closes)

        # Define um limiar de inclinação mínima
        limite = 0.01  # ajustável conforme o ativo/volatilidade

        if slope > limite:
            return TipoSinal.COMPRA.value
        elif slope < -limite:
            return TipoSinal.VENDA.value
        else:
            return TipoSinal.INDEFINIDO.value

    def verificaResultadoPadraoDivergencia(self, df, time, type):
        if not self.enableVerifyDivergence:
            return [(None, None, None, None, None, None)]

        divergencias = []
        resultadosPadroesDetectado = self.detectarPadroes(df, NUM_CANDLES_DEFAULT, time)
        tendencia = self.identificar_tendencia(df, NUM_CANDLES_DEFAULT * 10)
        for resultado in resultadosPadroesDetectado:
            padrao = resultado[5]
            sinal = resultado[2]

            if self.enablePatternAnalysis:
                if type == TipoSinal.COMPRA.value:
                    if padrao == TipoPadrao.DOJI.value or padrao == TipoPadrao.ESTRELA_MANHA.value or padrao == TipoPadrao.MARTELO.value or padrao == TipoPadrao.ENGOLFO_ALTA.value:
                        divergencias.append(resultado)
                elif type == TipoSinal.VENDA.value:
                    if padrao == TipoPadrao.DOJI.value or padrao == TipoPadrao.ESTRELA_NOITE.value or padrao == TipoPadrao.ENFORCADO.value or padrao == TipoPadrao.ENGOLFO_BAIXA.value:
                        divergencias.append(resultado)

            if tendencia == type and type != sinal:
                divergencias.append(resultado)

        return divergencias

    def filtrarSinais(self, sinais, tipo: TipoSinal, n):
        """
        Filtra os últimos 'n' sinais do tipo especificado ('COMPRA' ou 'VENDA').

        Parâmetros:
            sinais (list): Lista de tuplas (data, close, sinal)
            tipo (str): Tipo do sinal ('COMPRA' ou 'VENDA')
            n (int): Número de sinais a retornar (default=5)

        Retorna:
            list: Lista com os últimos 'n' sinais do tipo escolhido
        """
        # Filtra só o tipo
        if TipoSinal.INDEFINIDO.value == tipo.value:
            filtrados = [s for s in sinais if (s[2] == TipoSinal.COMPRA.value or s[2] == TipoSinal.VENDA.value)]
        else:
            filtrados = [s for s in sinais if (s[2] == tipo.value)]

        # Ordena do mais recente para o mais antigo (data no índice 0 da tupla)
        filtrados.sort(key=lambda x: x[0], reverse=False)

        # Formata a data para pt-BR

        if n == 0:
            sinais_formatados = [(self.converterDataPtBr(s[0]), s[1], s[2], s[3], s[4], s[5], s[0]) for s in filtrados]
        else:
            sinais_formatados = [(self.converterDataPtBr(s[0]), s[1], s[2], s[3], s[4], s[5], s[0]) for s in
                                 filtrados[:n]]

        return sinais_formatados

    def converterDataPtBr(self, data):
        # Se for int, converte para datetime (assumindo timestamp em segundos)
        if isinstance(data, int):
            dt = datetime.fromtimestamp(data)
        else:
            dt = data  # já é datetime

        return dt.strftime("%d/%m/%Y %H:%M")

    def calcularMACD(self, df, janela_stop=5):
        """
        Gera sinais de compra e venda com base no cruzamento MACD e retorna uma lista de tuplas:
        (data, fechamento, tipo_sinal, stop, take)

        Parâmetros:
            df (DataFrame): Precisa ter colunas 'time' (datetime) e 'close' (float)
            janela_stop (int): Número de candles anteriores usados para calcular stop

        Retorno:
            list: Lista de tuplas (data, close, TipoSinal, stop, take)
        """
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sinal = macd.ewm(span=9, adjust=False).mean()

        df['macd'] = macd
        df['macd_signal'] = sinal

        df['sinal'] = TipoSinal.INDEFINIDO.value
        df.loc[(df['macd'].shift(1) < df['macd_signal'].shift(1)) &
               (df['macd'] > df['macd_signal']), 'sinal'] = TipoSinal.COMPRA.value
        df.loc[(df['macd'].shift(1) > df['macd_signal'].shift(1)) &
               (df['macd'] < df['macd_signal']), 'sinal'] = TipoSinal.VENDA.value

        sinais = []

        for i, row in df.iterrows():
            if row['sinal'] == TipoSinal.INDEFINIDO.value:
                continue

            idx = df.index.get_loc(i)
            if idx < janela_stop:
                continue  # ignora se não tem dados suficientes para calcular stop

            janela = df.iloc[idx - janela_stop:idx]

            entrada = row['close']
            if row['sinal'] == TipoSinal.COMPRA.value:
                stop = janela['low'].min()
                take = entrada + (entrada - stop)
            elif row['sinal'] == TipoSinal.VENDA.value:
                stop = janela['high'].max()
                take = entrada - (stop - entrada)
            else:
                stop = None
                take = None

            for result in self.verificaResultadoPadraoDivergencia(df, row['time'], row['sinal']):
                sinais.append((
                    row['time'],
                    entrada,
                    row['sinal'],
                    round(stop, 5),
                    round(take, 5),
                    "MACD"
                ))

        return sinais

    def calcularCci(self, df, periodo=20, janela_stop=5):
        """
        Gera sinais de compra e venda com base no indicador CCI, com stop e take calculados.
        Substitui None por 0.0 para evitar erros.

        Retorno:
            list: Lista de tuplas (data, close, sinal, stop, take)
        """
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = df['tp'].rolling(window=periodo).mean()
        mad = df['tp'].rolling(window=periodo).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df['cci'] = (df['tp'] - tp_ma) / (0.015 * mad)

        df['sinal'] = TipoSinal.INDEFINIDO.value
        df.loc[(df['cci'].shift(1) < -150) & (df['cci'] > -100), 'sinal'] = TipoSinal.COMPRA.value
        df.loc[(df['cci'].shift(1) > 150) & (df['cci'] < 100), 'sinal'] = TipoSinal.VENDA.value

        sinais = []

        for i, row in df.iterrows():
            if row['sinal'] == TipoSinal.INDEFINIDO.value:
                continue

            idx = df.index.get_loc(i)
            if idx < janela_stop:
                continue

            janela = df.iloc[idx - janela_stop:idx]
            entrada = row['close']

            if row['sinal'] == TipoSinal.COMPRA.value:
                stop = janela['low'].min()
                take = entrada + (entrada - stop)
            elif row['sinal'] == TipoSinal.VENDA.value:
                stop = janela['high'].max()
                take = entrada - (stop - entrada)
            else:
                stop = None
                take = None

            # Substitui None por zero
            stop = 0.0 if stop is None else round(stop, 5)
            take = 0.0 if take is None else round(take, 5)

            for result in self.verificaResultadoPadraoDivergencia(df, row['time'], row['sinal']):
                sinais.append((
                    row['time'],
                    entrada,
                    row['sinal'],
                    stop,
                    take,
                    "CCI"
                ))

        return sinais

    def calcularRSI(self, df, periodo=14, janela_stop=5, rsi_entrada=30, rsi_saida=70):
        """
        Gera sinais de compra e venda com base no indicador RSI, com stop e take calculados.

        Parâmetros:
            df (DataFrame): Deve conter 'time' (datetime) e 'close' (float)
            periodo (int): Período do RSI
            janela_stop (int): Número de candles anteriores para calcular stop/take

        Retorno:
            list: Lista de tuplas (data, close, sinal, stop, take)
        """

        delta = df['close'].diff()
        ganho = np.where(delta > 0, delta, 0)
        perda = np.where(delta < 0, -delta, 0)

        ganho_ema = pd.Series(ganho).ewm(span=periodo, adjust=False).mean()
        perda_ema = pd.Series(perda).ewm(span=periodo, adjust=False).mean()

        rs = ganho_ema / (perda_ema + 1e-10)  # Evita divisão por zero
        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi

        df['sinal'] = TipoSinal.INDEFINIDO.value

        # Confirmação adicional: só compra se a tendência anterior era de sobrevenda (rsi < 30 por mais de 1 candle)
        df['rsi_below'] = df['rsi'] < rsi_entrada
        df['rsi_above'] = df['rsi'] > rsi_saida

        for i in range(2, len(df)):
            if df.loc[df.index[i - 2], 'rsi_below'] and df.loc[df.index[i - 1], 'rsi'] < rsi_entrada and df.loc[
                df.index[i], 'rsi'] > rsi_entrada:
                df.loc[df.index[i], 'sinal'] = TipoSinal.COMPRA.value

            if df.loc[df.index[i - 2], 'rsi_above'] and df.loc[df.index[i - 1], 'rsi'] > rsi_saida and df.loc[
                df.index[i], 'rsi'] < rsi_saida:
                df.loc[df.index[i], 'sinal'] = TipoSinal.VENDA.value

        sinais = []

        for i, row in df.iterrows():
            if row['sinal'] == TipoSinal.INDEFINIDO.value:
                continue

            idx = df.index.get_loc(i)
            if idx < janela_stop:
                continue

            janela = df.iloc[idx - janela_stop:idx]
            entrada = row['close']

            if row['sinal'] == TipoSinal.COMPRA.value:
                stop = janela['low'].min()
                take = entrada + (entrada - stop)  # Relação risco/retorno melhor
            elif row['sinal'] == TipoSinal.VENDA.value:
                stop = janela['high'].max()
                take = entrada - (stop - entrada)
            else:
                continue

            stop = round(stop, 5)
            take = round(take, 5)

            for result in self.verificaResultadoPadraoDivergencia(df, row['time'], row['sinal']):
                sinais.append((
                    row['time'],
                    entrada,
                    row['sinal'],
                    stop,
                    take,
                    "RSI"
                ))

        return sinais

    def verificarPadroesRepetidos(self, df_padrao, df_comparacao, tamanho_padrao=4):
        """
        Compara os últimos N candles com padrões semelhantes no passado
        e sugere COMPRA ou VENDA baseado no comportamento posterior.

        Parâmetros:
            df (DataFrame): Deve conter colunas 'open', 'high', 'low', 'close'
            tamanho_padrao (int): Quantidade de candles a comparar (padrão: 4)
            tolerancia (float): Tolerância percentual para considerar padrões semelhantes (ex: 0.5%)

        Retorno:
            str: 'COMPRA', 'VENDA' ou 'NEUTRO'
        """
        tolerancia = 0.5
        if len(df_padrao) < tamanho_padrao or len(df_comparacao) < tamanho_padrao + 1:
            return TipoSinal.INDEFINIDO.value

            # Extrai o padrão a partir dos últimos N candles do df_padrao
        padrao_atual = df_padrao.iloc[-tamanho_padrao:][['open', 'high', 'low', 'close']].values

        sinais = {'compra': 0, 'venda': 0}

        # Percorre o df_comparacao buscando padrões semelhantes
        for i in range(len(df_comparacao) - tamanho_padrao - 1):
            grupo = df_comparacao.iloc[i:i + tamanho_padrao][['open', 'high', 'low', 'close']].values

            similar = True
            for j in range(tamanho_padrao):
                ref = padrao_atual[j]
                comp = grupo[j]
                # Calcula diferença percentual entre candles
                diferencas = np.abs((ref - comp) / ref) * 100
                if np.any(diferencas > tolerancia):
                    similar = False
                    break

            if similar:
                fechamento_anterior = df_comparacao.iloc[i + tamanho_padrao - 1]['close']
                fechamento_seguinte = df_comparacao.iloc[i + tamanho_padrao]['close']
                variacao = (fechamento_seguinte - fechamento_anterior) / fechamento_anterior

                if variacao > 0.001:
                    sinais['compra'] += 1
                elif variacao < -0.001:
                    sinais['venda'] += 1

        if sinais['compra'] > sinais['venda']:
            return TipoSinal.COMPRA.value
        elif sinais['venda'] > sinais['compra']:
            return TipoSinal.VENDA.value
        else:
            return TipoSinal.INDEFINIDO.value

    def detectarPadroes(self, df, num_candles=NUM_CANDLES_DEFAULT, data_inicio=None):
        # Converte a coluna 'time' para datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Filtra o DataFrame entre as datas, se fornecidas
        if data_inicio:
            intervalo_minutos = timeframe_to_minutes[self.timeframe] * num_candles  # Retorna 60
            data_fim = pd.to_datetime(data_inicio)
            data_inicial = data_fim - pd.Timedelta(minutes=intervalo_minutos)
            df = df[df['time'] >= data_inicial]
            df = df[df['time'] <= data_fim]

        # Reindexa após o filtro (opcional, mas pode evitar problemas de índice)
        df = df.reset_index(drop=True)

        padroes_detectados = []

        # Para cada posição possível no DataFrame que permite pegar 'tamanho' candles
        for i in range(len(df) - num_candles + 1):
            # Para cada tamanho de grupo de 1 até num_candles
            for tamanho in range(1, num_candles + 1):
                grupo = df.iloc[i:i + tamanho][['open', 'high', 'low', 'close']].to_dict('records')

                # Armazena o candle anterior ao grupo atual (se houver)
                if i > 0:
                    historico_anterior = [df.iloc[i - 1][['open', 'high', 'low', 'close']].to_dict()]
                else:
                    historico_anterior = []

                padrao = self.identificarPadraoCandle(grupo, historico_anterior)
                if padrao['default'] != TipoPadrao.INDEFINIDO:
                    fechamento = df.iloc[i + tamanho - 1]['close']
                    padroes_detectados.append(
                        (df.iloc[i]['time'], fechamento, padrao['signal'], padrao['stop'], padrao['take'],
                         padrao['default'].value)
                    )

        return padroes_detectados

    def identificarPadraoCandle(self, grupo, historico_anterior):
        def corpo(c):
            return abs(c['close'] - c['open'])

        def tamanho(c):
            return c['high'] - c['low']

        def sombra_superior(c):
            return c['high'] - max(c['open'], c['close'])

        def sombra_inferior(c):
            return min(c['open'], c['close']) - c['low']

        def format_result(padrao, stop, take):
            if stop > take:
                return {"signal": TipoSinal.VENDA.value, "default": padrao, "stop": round(stop, 5),
                        "take": round(take, 5)}
            elif stop < take:
                return {"signal": TipoSinal.COMPRA.value, "default": padrao, "stop": round(stop, 5),
                        "take": round(take, 5)}
            else:
                return {"signal": TipoSinal.INDEFINIDO.value, "default": padrao, "stop": round(stop, 5),
                        "take": round(take, 5)}

        if len(grupo) == 1:
            c = grupo[0]
            corpo_c = corpo(c)
            tamanho_c = tamanho(c)
            if tamanho_c == 0:
                return {"default": TipoPadrao.INDEFINIDO, "stop": 0, "take": 0}

            proporcao_corpo = corpo_c / tamanho_c
            sombra_inf = sombra_inferior(c)
            sombra_sup = sombra_superior(c)

            if proporcao_corpo < 0.3:
                if sombra_inf > corpo_c * 2 and sombra_sup < corpo_c * 0.5:
                    stop = c['low']
                    take = c['close'] + corpo_c * 2
                    return format_result(TipoPadrao.MARTELO, stop, take)

                elif sombra_sup > corpo_c * 2 and sombra_inf < corpo_c * 0.5:
                    stop = c['high']
                    take = c['close'] - corpo_c * 2
                    return format_result(TipoPadrao.ENFORCADO, stop, take)

            if proporcao_corpo < 0.1:
                if len(historico_anterior) >= 1:
                    anterior = historico_anterior[-1]
                    corpo_anterior = abs(anterior['close'] - anterior['open'])
                    tamanho_anterior = anterior['high'] - anterior['low']
                    if corpo_anterior / tamanho_anterior > 0.5:
                        # Direção anterior foi clara
                        if anterior['close'] > anterior['open']:
                            # Tendência de alta anterior → possível reversão para baixa
                            stop = c['high']
                            take = c['close'] - (corpo_anterior * 1.5)
                        else:
                            # Tendência de baixa anterior → possível reversão para alta
                            stop = c['low']
                            take = c['close'] + (corpo_anterior * 1.5)
                        return format_result(TipoPadrao.DOJI, stop, take)

        elif len(grupo) == 2:
            c1, c2 = grupo
            corpo2 = corpo(c2)
            total2 = tamanho(c2)

            if total2 == 0 or corpo2 / total2 < 0.7:
                return format_result(TipoPadrao.INDEFINIDO, 0, 0)

            # Engolfo de Alta
            if c2['open'] < c1['close'] < c1['open'] < c2['close'] and c2['close'] > c2['open']:
                stop = c2['low']
                take = c2['close'] + corpo2 * 1.5
                return format_result(TipoPadrao.ENGOLFO_ALTA, stop, take)

            # Engolfo de Baixa
            if c2['open'] > c1['close'] > c1['open'] > c2['close'] and c2['close'] < c2['open']:
                stop = c2['high']
                take = c2['close'] - corpo2 * 1.5
                return format_result(TipoPadrao.ENGOLFO_BAIXA, stop, take)

        elif len(grupo) == 3:
            c1, c2, c3 = grupo
            corpo2 = corpo(c2)
            if tamanho(c2) == 0 or corpo2 / tamanho(c2) > 0.3:
                return format_result(TipoPadrao.INDEFINIDO, 0, 0)

            # Estrela da Manhã
            if c1['close'] < c1['open'] and c3['close'] > c3['open'] and \
                    c3['close'] > ((c1['open'] + c1['close']) / 2):
                stop = c3['low']
                take = c3['close'] + (c3['close'] - c1['close'])
                return format_result(TipoPadrao.ESTRELA_MANHA, stop, take)

            # Estrela da Noite
            if c1['close'] > c1['open'] and c3['close'] < c3['open'] and \
                    c3['close'] < ((c1['open'] + c1['close']) / 2):
                stop = c3['high']
                take = c3['close'] - (c1['close'] - c3['close'])
                return format_result(TipoPadrao.ESTRELA_NOITE, stop, take)

        return format_result(TipoPadrao.INDEFINIDO, 0, 0)

    def agruparSinaisPorDia(self, sinais):
        agrupados = defaultdict(list)

        for sinal in sinais:
            timestamp = pd.to_datetime(sinal[0], dayfirst=True)
            data = timestamp.date()
            agrupados[data].append(sinal)

        return agrupados

    def analisarTodosIndicadores(self, dataframe, minPontosEntrada):
        """
        Consolida sinais de MACD, RSI, CCI, Bollinger e Padrões de Candles.

        Retorno:
            list: Lista de tuplas (data, indicador, tipo_sinal, close, stop, take)
        """
        sinais = []

        def agruparResultados(resultados):
            for timeBR, close, tipo, stop, take, padrao, time in resultados:
                sinais.append((timeBR, close, tipo, stop, take, padrao, time))

        # if self.enableBollinger:
        #     resultadoSinaisBollinger = self.calcularBollinger(dataframe)
        #     agruparResultados(self.filtrarSinais(resultadoSinaisBollinger, TipoSinal.INDEFINIDO, minPontosEntrada))
        # if self.enableMACD:
        #     resultadoSinaisMACD = self.calcularMACD(dataframe)
        #     agruparResultados(self.filtrarSinais(resultadoSinaisMACD, TipoSinal.INDEFINIDO, minPontosEntrada))
        # if self.enableCCI:
        #     resultadoSinaisCCI = self.calcularCci(dataframe)
        #     agruparResultados(self.filtrarSinais(resultadoSinaisCCI, TipoSinal.INDEFINIDO, minPontosEntrada))
        # if self.enableRSI:
        #     resultadoSinaisRSI = self.calcularRSI(dataframe)
        #     agruparResultados(self.filtrarSinais(resultadoSinaisRSI, TipoSinal.INDEFINIDO, minPontosEntrada))
        # if self.enableADX:
        #     resultadoSinaisADX = self.candle_trend(dataframe)
        #     agruparResultados(self.filtrarSinais(resultadoSinaisADX, TipoSinal.INDEFINIDO, minPontosEntrada))

        medias = self.get_moving_averages(dataframe)


        # Ordena os sinais pela data (mais recentes primeiro)
        sinais.sort(key=lambda x: x[0], reverse=True)
        sinais_filtrados = []

        for sinal in sinais:
            data, entrada, tipo, stop, take, metodo, timestamp = sinal
            risco = 0
            lucro = 0

            if tipo == TipoSinal.COMPRA.value:
                risco = entrada - stop
                lucro = take - entrada
            elif tipo == TipoSinal.VENDA.value:
                risco = stop - entrada
                lucro = entrada - take
            else:
                continue  # tipo inválido, pula

            pips = self.calcularPontosGrafico(risco)
            if (risco * self.percentRiskAndProfit / 100) <= lucro and pips > self.stopMinimo:
                sinais_filtrados.append(sinal)

        sinais_filtrados = self.executarMartingale(sinais_filtrados, self.enableMartingale)
        sinais_filtrados = self.limitarSinaisPorMaximoPerdaDiaria(sinais_filtrados)

        return sinais_filtrados

    def limitarSinaisPorMaximoPerdaDiaria(self, sinais):
        if self.maximoPerdaDiaria > 0:
            sinais_filtrados = []
            agrupados = self.agruparSinaisPorDia(sinais)
            for dia, sinais_do_dia in agrupados.items():
                totalRisco = 0
                for sinal in sinais_do_dia:
                    data, entrada, tipo, stop, take, metodo, timestamp = sinal
                    timestamp = pd.to_datetime(data, dayfirst=True)
                    if timestamp.date() == dia:
                        totalRisco = totalRisco + self.calcular_valor_em_dolares(abs(entrada - stop), self.volume)
                        if totalRisco <= self.maximoPerdaDiaria:
                            sinais_filtrados.append(sinal)

            return sinais_filtrados

        return sinais

    def getQtdPosicoes(self):
        """
        Verifica se há uma ordem (posição) aberta para o símbolo especificado.

        Retorna:
            True  -> se existe pelo menos uma posição aberta
            False -> se não houver nenhuma posição aberta
        """
        posicoes = mt5.positions_get(symbol=self.symbol)
        return len(posicoes)

    def existe_ordem_aberta(self):
        """
        Verifica se há uma ordem (posição) aberta para o símbolo especificado.

        Retorna:
            True  -> se existe pelo menos uma posição aberta
            False -> se não houver nenhuma posição aberta
        """
        posicoes = mt5.positions_get(symbol=self.symbol)
        if posicoes is None:
            print("Erro ao acessar posições.")
            return False

        if len(posicoes) > 0:
            return True
        else:
            return False

    def agrupar_por_proximidade_temporal_e_valor(self, sinais, tolerancia_min=5, tolerancia_valor=5.0):
        """
        Agrupa sinais por tipo (COMPRA/VENDA) onde:
        - a diferença de tempo entre sinais ≤ tolerancia_min (minutos)
        - a diferença de valor (close) entre sinais ≤ tolerancia_valor (ex: 5 pontos)

        Retorno:
            dict com agrupamento mais frequente de COMPRA e VENDA
        """
        compras = []
        vendas = []

        # Separa por tipo de sinal
        for s in sinais:
            if s[2] == 'COMPRA':
                compras.append(s)
            elif s[2] == 'VENDA':
                vendas.append(s)

        def agrupar(sinais_filtrados):
            grupos = []

            for sinal in sinais_filtrados:
                _, close, _, _, _, _, dt = sinal
                colocado = False

                for grupo in grupos:
                    for ref_sinal in grupo:
                        _, ref_close, _, _, _, _, ref_dt = ref_sinal
                        if (
                                abs((dt - ref_dt).total_seconds()) <= tolerancia_min * 60 and
                                abs(close - ref_close) <= tolerancia_valor
                        ):
                            grupo.append(sinal)
                            colocado = True
                            break
                    if colocado:
                        break

                if not colocado:
                    grupos.append([sinal])

            # Retorna o grupo com mais elementos
            grupo_mais_frequente = max(grupos, key=len, default=[])

            return {
                'valor_medio': round(sum(s[1] for s in grupo_mais_frequente) / len(grupo_mais_frequente),
                                     2) if grupo_mais_frequente else 0,
                'stop_medio': round(sum(s[3] for s in grupo_mais_frequente) / len(grupo_mais_frequente),
                                    2) if grupo_mais_frequente else 0,
                'take_medio': round(sum(s[4] for s in grupo_mais_frequente) / len(grupo_mais_frequente),
                                    2) if grupo_mais_frequente else 0,
                'ocorrencias': len(grupo_mais_frequente),
                'datas': [s[0] for s in grupo_mais_frequente],
                'padroes': [s[5] for s in grupo_mais_frequente],
                'timestamps': [s[6] for s in grupo_mais_frequente]
            }

        return {
            TipoSinal.COMPRA: agrupar(compras),
            TipoSinal.VENDA: agrupar(vendas)
        }

    def mapearGruposParaTuplas(self, grupos, tipoSinal: TipoSinal):
        obj_venda = grupos[tipoSinal]
        if obj_venda['ocorrencias'] > 1:
            return [(
                data,
                obj_venda['valor_medio'],
                TipoSinal.VENDA.value,
                obj_venda['stop_medio'],
                obj_venda['take_medio'],
                obj_venda['padroes'][i],
                obj_venda['timestamps'][i]
            ) for i, data in enumerate(obj_venda['datas'])]

        return None

    def executarOrdemPendente(self, tipo_ordem: TipoSinal, preco_entrada, stop, take, volume=0, pattern=''):
        """
        Envia uma ordem pendente (buy limit ou sell limit) para o MetaTrader 5.

        Parâmetros:
            symbol (str): Nome do ativo (ex: 'WIN$N')
            tipo_ordem (str): 'buy' ou 'sell'
            preco_entrada (float): Preço de entrada da ordem
            stop (float): Valor do stop loss
            take (float): Valor do take profit
            volume (float): Lote da ordem (default 0.1)
            magic (int): Identificador único da ordem (opcional)

        Retorno:
            dict: Resultado da operação
        """
        if volume == 0:
            volume = self.volume

        tipo_ordem_mt5 = {
            TipoSinal.COMPRA: mt5.ORDER_TYPE_BUY_LIMIT,
            TipoSinal.VENDA: mt5.ORDER_TYPE_SELL_LIMIT
        }.get(tipo_ordem)

        if tipo_ordem_mt5 is None:
            return {"erro": "Tipo de ordem inválido. Use 'buy' ou 'sell'."}

        # Hora de expiração (1 hora a partir de agora)
        data_expiracao = datetime.now() + timedelta(hours=1)

        ordem = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": volume,
            "type": tipo_ordem_mt5,
            "price": preco_entrada,
            "sl": stop,
            "tp": take,
            "deviation": 10,
            "magic": self.magic,
            "comment": "Ordem pendente via Python - " + pattern,
            "type_time": mt5.ORDER_TIME_GTC,  # GTC = Good Till Cancelled
            "type_filling": mt5.ORDER_FILLING_RETURN
        }

        resultado = mt5.order_send(ordem)

        if resultado.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "erro": "Falha ao enviar ordem",
                "codigo": resultado.retcode,
                "mensagem": mt5.last_error()
            }

        return {"sucesso": True, "ordem": resultado}

    def executarOrdemImediata(self, tipo_ordem: TipoSinal, sl: float, tp: float, volume=0, pattern=""):
        """
        Envia uma ordem de mercado (BUY ou SELL) no MetaTrader 5.

        :param symbol: Ex: 'XAUUSD'
        :param lotes: volume da ordem, ex: 0.1
        :param tipo_ordem: 'buy' ou 'sell'
        :param sl: stop loss (preço) opcional
        :param tp: take profit (preço) opcional
        :param magic: número mágico da ordem
        :param comentario: comentário da ordem
        """
        info = mt5.symbol_info_tick(self.symbol)
        if volume == 0:
            volume = self.volume

        # Define preço de entrada com base no tipo de ordem
        price = info.ask if tipo_ordem == TipoSinal.COMPRA else info.bid
        if tipo_ordem == TipoSinal.COMPRA.value:
            tipo = mt5.ORDER_TYPE_BUY
        elif tipo_ordem == TipoSinal.VENDA.value:
            tipo = mt5.ORDER_TYPE_SELL

        comentario = frase = f"Ordem pendente via Python {pattern}"
        # Prepara estrutura de ordem
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": tipo,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 10,
            "magic": self.magic,
            "comment": comentario,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        # Verifica resultado
        if result is None:
            print("Erro grave:", mt5.last_error())
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Erro ao enviar ordem: {result.retcode} - {result.comment}")
        elif result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Ordem enviada com sucesso: Ticket {result.order} Preco {price}")

        return result.retcode == mt5.TRADE_RETCODE_DONE

    def obter_valor_pip(self, lote=1.0):
        # Verifica se símbolo está disponível
        info = mt5.symbol_info(self.symbol)
        tick = mt5.symbol_info_tick(self.symbol)

        # Detecta tamanho do pip automaticamente
        digitos = info.digits
        pip_size = 0.01 if digitos == 3 else 0.0001

        # Valor do pip com fórmula padrão
        preco = tick.bid
        contrato = info.trade_contract_size  # normalmente 100.000 para Forex
        valor_pip_manual = (pip_size / preco) * contrato * lote

        # Valor do pip fornecido pela corretora
        valor_pip_broker = info.trade_tick_value * lote

        return {
            "preco_atual": preco,
            "pip_size": pip_size,
            "valor_pip_manual": round(valor_pip_manual, 2),
            "valor_pip_broker": round(valor_pip_broker, 2),
            "contrato": contrato,
            "pontos": info.point,
            "digitos": digitos
        }

    def filtrarPorMinutosEPreco(self, sinais, n_minutos):
        # Filtrar pelos últimos N minutos
        agora = datetime.now()
        limite = agora - timedelta(minutes=n_minutos)

        # Filtrar
        tick = mt5.symbol_info_tick(self.symbol)
        sinais_filtrados = []
        for s in sinais:
            horarioBrasilia = s[6] - timedelta(hours=6)
            if horarioBrasilia >= limite:
                if (s[2] == TipoSinal.COMPRA.value and s[1] <= tick.bid) or (
                        s[2] == TipoSinal.VENDA.value and s[1] >= tick.bid):
                    sinais_filtrados.append(s)

        return sinais_filtrados

    def get_posicao_ativa(self):
        # Pega as posições abertas para o símbolo
        posicoes = mt5.positions_get(symbol=self.symbol)
        if posicoes is None or len(posicoes) == 0:
            print("Nenhuma posição aberta para", self.symbol)
            return None

        # Exemplo: pegar a primeira posição aberta (pode adaptar para várias)
        pos = posicoes[0]
        tipo = "COMPRA" if pos.type == mt5.POSITION_TYPE_BUY else "VENDA"
        entrada = pos.price_open
        volume = pos.volume
        # O stop loss atual está em pos.sl
        stop_inicial = pos.sl
        take = pos.tp

        if pos.tp == 0:
            take = self.calcular_take_proporcional(entrada, stop_inicial, tipo)  # take profit

        return {
            "tipo": tipo,
            "entrada": entrada,
            "stop": stop_inicial,
            "take": take,
            "volume": volume,
            "ticket": pos.ticket
        }

    def move_stop(self, ganhoPercent, protecaoPercent):
        """
        Move o stop quando o preço atingir uma certa porcentagem do alvo.

        - entrada: preço de entrada
        - stop_inicial: valor original do SL
        - take: valor do take profit
        - tipo: 'COMPRA' ou 'VENDA'
        - ganho_trigger: % de progresso até o take para começar a mover o stop (0.5 = 50%)
        - protecao: % do caminho lucro que será protegido ao mover o stop (0.3 = 30%)

        Retorna: novo_stop ou None (se ainda não atingiu o trigger)
        """
        posicao = self.get_posicao_ativa()
        if posicao is None:
            return None

        tick = mt5.symbol_info_tick(self.symbol)
        preco = tick.bid
        entrada = posicao['entrada']
        take = posicao['take']
        stop = posicao['stop']
        tipo = posicao['tipo']
        ticket = posicao['ticket']

        ganho_trigger = ganhoPercent / 100
        protecao = protecaoPercent / 100

        if self.enableMoveStop:
            # Condição para mover stop
            if tipo == TipoSinal.COMPRA.value:
                trigger_price = entrada + (take - entrada) * ganho_trigger
                novo_stop = entrada + (take - entrada) * protecao
                if preco >= trigger_price and (stop is None or novo_stop > stop):
                    sucesso = self.modificar_stop(ticket, novo_stop, take, tipo)
                    if sucesso:
                        print(f"Stop movido para {novo_stop} no preço {preco}")
            elif tipo == TipoSinal.VENDA.value:
                novo_stop = entrada - (entrada - take) * protecao
                trigger_price = entrada - (entrada - take) * ganho_trigger
                if preco <= trigger_price and (stop is None or novo_stop < stop):
                    sucesso = self.modificar_stop(ticket, novo_stop, take, tipo)
                    if sucesso:
                        print(f"Stop movido para {novo_stop} no preço {preco}")

    def modificar_stop(self, ticket, novo_stop, take, tipo):
        tipoOrdem = TipoSinal.INDEFINIDO.value
        if tipo == TipoSinal.COMPRA.value:
            tipoOrdem = mt5.ORDER_TYPE_BUY
        elif tipo == TipoSinal.VENDA.value:
            tipoOrdem = mt5.ORDER_TYPE_SELL

        if tipoOrdem != TipoSinal.INDEFINIDO.value:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "position": ticket,
                "sl": novo_stop,
                "tp": take,  # ou o take que quiser manter
                "deviation": 10,
                "type": tipoOrdem,  # não altera tipo da ordem
                "magic": 0,
                "comment": "Movendo stop",
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }

            resultado = mt5.order_send(request)
            if resultado is None:
                print("Erro grave:", mt5.last_error())
                return False

            if resultado.retcode == mt5.TRADE_RETCODE_DONE:
                print("✅ Stop Loss successfully moved.")
                return True
            else:
                print(f"❌ Failed to move SL. Retcode: {resultado.retcode} - Comment: {resultado.comment}")
                return False
        return False

    def monitorar_ticks(self, callback, intervalo=1):
        """
        Função que roda em loop, e chama callback(price) a cada novo tick.
        """
        ultimo_tick = None
        while True:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is not None:
                if ultimo_tick != tick.last:
                    ultimo_tick = tick.last
                    callback(tick.last)  # chama a função callback com o preço atual
            time.sleep(intervalo)

    def minha_funcao_callback(self, preco_atual):
        print(f"Callback chamado com preço: {preco_atual}")
        # Aqui pode chamar move_stop ou outras lógicas

    def adicionarPontos(self, valor, pontos):
        valorPip = self.obter_valor_pip()
        return valor + round(pontos * valorPip['pontos'], 2)

    def calcularPontosGrafico(self, pips):
        valorPip = self.obter_valor_pip()
        return round(pips / valorPip['pontos'], 2)

    def calcular_valor_em_dolares(self, pips, volume_lote):
        valorPip = self.calcularPontosGrafico(pips)
        return round(valorPip * volume_lote, 2)

    def calcular_take_proporcional(self, entrada, stop, tipo, risco_retorno=1.0):
        """
        Calcula o take profit baseado na distância do stop para a entrada.

        Parâmetros:
            entrada (float): Preço de entrada da operação
            stop (float): Stop loss definido
            tipo (str): 'COMPRA' ou 'VENDA'
            risco_retorno (float): Quantas vezes o risco será usado como alvo (ex: 2.0 = 2:1)

        Retorna:
            float: preço alvo (take profit)
        """
        if tipo.upper() == "COMPRA":
            distancia = entrada - stop
            take = entrada + distancia * risco_retorno
        elif tipo.upper() == "VENDA":
            distancia = stop - entrada
            take = entrada - distancia * risco_retorno
        else:
            raise ValueError("Tipo deve ser 'COMPRA' ou 'VENDA'.")

        return round(take, 2)

    def calculate_adx(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calcula o ADX (Average Directional Index).
        Espera colunas: 'high', 'low', 'close'
        Retorna o DataFrame com colunas ['plus_di', 'minus_di', 'adx'].
        """
        df = df.copy()

        # True Range (TR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Directional Movement
        df['+dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                             np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['-dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                             np.maximum(df['low'].shift(1) - df['low'], 0), 0)

        # Suavização simples (pode trocar por EMA se preferir)
        df['tr_rma'] = df['tr'].rolling(period).mean()
        df['+dm_rma'] = df['+dm'].rolling(period).mean()
        df['-dm_rma'] = df['-dm'].rolling(period).mean()

        # DI+
        df['plus_di'] = 100 * (df['+dm_rma'] / df['tr_rma'])
        # DI-
        df['minus_di'] = 100 * (df['-dm_rma'] / df['tr_rma'])

        # DX e ADX
        df['dx'] = (100 * abs(df['plus_di'] - df['minus_di']) /
                    (df['plus_di'] + df['minus_di']))
        df['adx'] = df['dx'].rolling(period).mean()

        return df[['plus_di', 'minus_di', 'adx']]

    def trade_signal(self, df: pd.DataFrame, threshold: float = 25.0):
        """
        Retorna o sinal de trade com ponto de entrada, stoploss e takeprofit.
        """
        sinais = []
        ind = self.calculate_adx(df).dropna()
        if ind.empty:
            return sinais

        last = ind.iloc[-1]
        price = df['close'].iloc[-1]
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        # Verifica se existe coluna 'time', senão usa o index
        if 'time' in df.columns:
            trade_time = df['time'].iloc[-1]
        else:
            trade_time = df.index[-1]

        if last['adx'] > threshold:
            if last['plus_di'] > last['minus_di']:
                # Sinal de compra
                sl = low
                entry = price
                tp = entry + (entry - sl)

                sinais.append((
                    trade_time,
                    entry,
                    TipoSinal.COMPRA.value,
                    sl,
                    tp,
                    "DXI"
                ))
            else:
                # Sinal de venda
                sl = high
                entry = price
                tp = entry - (sl - entry)
                sinais.append((
                    trade_time,
                    entry,
                    TipoSinal.VENDA.value,
                    sl,
                    tp,
                    "DXI"
                ))
        else:
            sinais.append((
                trade_time,
                None,
                TipoSinal.INDEFINIDO.value,
                None,
                None,
                "DXI"
            ))

        return sinais

    def candle_trend(self, df: pd.DataFrame, n: int = 5, body_ratio: float = 0.5):
        """
        Valida os últimos N candles para identificar tendência.
        Regras:
        - Corpo deve ser maior que body_ratio (ex.: 0.5 = 50%) da vela.
        - Se maioria bullish -> tendência de compra.
        - Se maioria bearish -> tendência de venda.
        - Caso contrário -> indefinido.

        Espera colunas: ['open', 'high', 'low', 'close'].
        """
        sinais = []
        sinaisCount = []

        if len(df) == 0:
            return sinais

        # Percorre do final para o início de N em N
        for i in range(len(df), n - 1, -n):
            window = df.iloc[i - n:i]  # pega n linhas da posição atual para trás
            for _, row in window.iterrows():
                candle_range = row['high'] - row['low']
                body = abs(row['close'] - row['open'])

                if candle_range == 0:  # evita divisão por zero
                    continue

                body_strength = body / candle_range

                if body_strength >= body_ratio:
                    if row['close'] > row['open']:
                        sinaisCount.append(TipoSinal.COMPRA.value)
                    else:
                        sinaisCount.append(TipoSinal.VENDA.value)
                else:
                    sinaisCount.append(TipoSinal.INDEFINIDO.value)

        # Contagem dos sinais
        buy_count = sinaisCount.count(TipoSinal.COMPRA.value)
        sell_count = sinaisCount.count(TipoSinal.VENDA.value)

        last_row = df.iloc[-1]
        entry = last_row['close']
        sl = last_row['low']
        tp = last_row['high']

        if buy_count > sell_count and buy_count >= n // 2:
            sinais.append((
                last_row['time'],
                entry,
                TipoSinal.COMPRA.value,
                sl,
                tp,
                "TENDENCY"
            ))
        elif sell_count > buy_count and sell_count >= n // 2:
            sinais.append((
                last_row['time'],
                entry,
                TipoSinal.VENDA.value,
                sl,
                tp,
                "TENDENCY"
            ))
        else:
            sinais.append((
                last_row['time'],
                None,
                TipoSinal.INDEFINIDO.value,
                None,
                None,
                "TENDENCY"
            ))
        return sinais

    # --- Função para calcular médias móveis ---
    def get_moving_averages(self, df: pd.DataFrame, periods=[20, 50, 80, 200, 400], tolerance=0.0005,
                            price_tolerance=0.0010):
        """
        Recupera médias móveis (SMA e EMA) em todos os timeframes do MT5.

        :param symbol: Ativo a ser analisado (ex: "EURUSD").
        :param periods: Lista de períodos das médias móveis.
        :param n_bars: Quantidade de candles a carregar.
        :return: dicionário com DataFrames contendo as médias móveis por timeframe.
        """
        # Lista de timeframes suportados
        timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            # "W1": mt5.TIMEFRAME_W1,
            # "MN1": mt5.TIMEFRAME_MN1,
        }
        # preço atual do símbolo
        current_price = self.get_current_price()

        results = {}
        all_mas = []  # todas as médias móveis

        for name, tf in timeframes.items():
            df['time'] = pd.to_datetime(df['time'], unit='s')

            last_ma = {}
            for p in periods:
                sma = df['close'].rolling(window=p).mean().iloc[-1]
                ema = df['close'].ewm(span=p, adjust=False).mean().iloc[-1]
                last_ma[f"SMA_{p}"] = sma
                last_ma[f"EMA_{p}"] = ema

                # só guarda se estiver perto do preço atual
                if abs(sma - current_price) <= price_tolerance:
                    all_mas.append((name, f"SMA_{p}", sma))
                if abs(ema - current_price) <= price_tolerance:
                    all_mas.append((name, f"EMA_{p}", ema))

            results[name] = last_ma

        # --- Agrupar apenas as médias próximas entre si ---
        if not all_mas:
            return results, [], current_price

        all_mas.sort(key=lambda x: x[2])
        clusters = []
        current_cluster = [all_mas[0]]

        for item in all_mas[1:]:
            if abs(item[2] - current_cluster[-1][2]) <= tolerance:
                current_cluster.append(item)
            else:
                clusters.append(current_cluster)
                current_cluster = [item]

        clusters.append(current_cluster)

        return results, clusters, current_price

    def get_last_closed_candles(self, n=10):
        """
        Recupera os últimos N candles fechados para um símbolo e timeframe.

        :param symbol: Ativo (ex: "EURUSD").
        :param timeframe: Timeframe do MT5 (ex: mt5.TIMEFRAME_M1, mt5.TIMEFRAME_H1).
        :param n: Quantidade de candles fechados a retornar.
        :return: DataFrame com N últimos candles fechados.
        """
        rates = mt5.copy_rates_from(self.symbol, self.timeframe, datetime.now(), n + 1)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Remove o último candle se ele ainda não estiver fechado
        now = datetime.now()
        last_candle_time = df.iloc[-1]['time']
        if last_candle_time > now:
            df = df.iloc[:-1]

        # Garante que só retornamos N candles fechados
        return df.tail(n)

    def get_current_price(self):
        tick = mt5.symbol_info_tick(self.symbol)
        current_price = tick.last

        if current_price is None or current_price <= 0:
            current_price = self.get_last_closed_candles(1)['close'].iloc[0]

        return current_price
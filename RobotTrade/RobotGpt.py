import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import MetaTrader5 as mt5
from openai import OpenAI
from win32ctypes.core import ctypes

from RobotTrade.PerplexityClient import PerplexityClient


class TipoSinal(Enum):
    COMPRA = 'Buy'
    VENDA = 'Sell'


@dataclass
class Candle:
    datetime_str: str
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class CandleResponse:
    price: float
    stop: float
    take: float
    type: str
    tendency: str
    time: datetime
    decision: str
    confidence: str
    justification: str


def impedir_reposo():
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )


class RobotGpt:
    def __init__(self, login: int, senha: str, servidor: str, symbol: str, timeframe: mt5.TIMEFRAME_M30, volume: float):
        self.login = login
        self.senha = senha
        self.servidor = servidor
        self.symbol = symbol
        self.timeframe = timeframe
        self.volume = volume

        self.account = self.conectar_mt5(login, senha, servidor)
        self.client = OpenAI(
            api_key="Teste")
        self.perplexityClient = PerplexityClient()

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

    def generatePrompt(self, contador: int, indices: Any, num_candles: int):
        candles = self.candles_to_json_string(indices)
        lines = [
            "Você é um trader profissional especialista em Day Trade e Scalp, com foco em Price Action, estrutura de mercado e indicadores técnicos, operando ouro (XAU/USD).",
            "",
            "Vou te fornecer:",
            f"- O ativo {self.symbol}",
            f"- O timeframe {self.mt5_timeframe_to_string(self.timeframe)}",
            f"- Com o preço atual: {indices[num_candles - 1].close}",
            f"- Os Últimos {num_candles} candles (ordem cronológica, do mais antigo para o mais recente):",
            f" Historico de Candles: {candles}",
            "",
            "OBJETIVO PRINCIPAL (ANÁLISE RETROATIVA):",
            "- Identificar TODAS as negociações PROVÁVEIS que TERIAM OCORRIDO nos ÚLTIMOS 50 MINUTOS,",
            "- Baseando-se EXCLUSIVAMENTE nos candles fornecidos",
            "- Reconstruir trades REALISTAS como se estivessem sendo executados em tempo real",
            "- NÃO prever o futuro",
            "- Avaliar candle a candle dentro da janela dos últimos 50 minutos",
            "- Ignorar completamente candles fora dessa janela",
            "- Caso nenhuma negociação válida exista, retornar lista vazia",
            "",
            "Inclua obrigatoriamente na análise, como critério NÃO BLOQUEANTE:",
            f"- Baseie-se no Perplexity Finance, ",
            "- Verificação de notícias e eventos recentes relevantes para o ouro e o mercado macro,",
            " considerando impacto (high / medium / low), com base em fontes internacionais confiáveis:",
            " Federal Reserve (Fed), U.S. Treasury, BIS, World Gold Council (WGC), LBMA, Bloomberg e Reuters.",
            "",
            f"Validações técnicas obrigatórias (calcular internamente a partir dos {contador} candles fornecidos):",
            "- CCI (identificar sobrecompra, sobrevenda e direção predominante)",
            "- MACD (cruzamentos, direção e força do histograma)",
            "- Médias Móveis:",
            " - Curta: EMA 9",
            " - Média: EMA 21",
            " - Longa: EMA 50",
            "- Bandas de Bollinger (compressão, expansão, toque PARCIAL e retorno à média)",
            "- Padrões de candle (rejeição, continuação ou reversão)",
            "",
            "Regras obrigatórias de análise:",
            "- O foco é Day Trade e Scalp",
            "- Avaliar oportunidades com horizonte de 1 a 90 minutos",
            "- Todos os indicadores devem ser calculados exclusivamente a partir dos candles fornecidos",
            "- Identificar a tendência imediata com base em MICRO estrutura (micro topos e fundos)",
            "- Avaliar força, enfraquecimento ou exaustão do movimento recente",
            "- Considerar cenários de pullback curto, continuação imediata ou reversão técnica curta",
            "- Utilizar confluência entre:",
            " - Microestrutura de mercado",
            " - Retorno à média curta (EMA 9 / EMA 21)",
            " - Comportamento dos candles (rompimento ou rejeição clara)",
            " - Confirmações de CCI, MACD, Médias Móveis e Bandas de Bollinger",
            "- Evitar trades contra tendência forte, salvo confirmação técnica ROBUSTA em microestrutura",
            "",
            "Regras específicas por tipo de operação:",
            "- Scalp:",
            " - Aceitar probabilidade MÉDIA se houver forte confluência técnica",
            " - Stops curtos baseados em microestrutura",
            " - Execução rápida (1 a 3 candles)",
            "- Day Trade curto:",
            " - Exigir probabilidade ALTA",
            " - Operações de 30 a 90 minutos",
            " - Preferir trades a favor da tendência imediata",
            "",
            "REGRAS PARA LISTA DE NEGOCIAÇÕES:",
            "- É permitido retornar MAIS DE UMA negociação",
            "- Cada trade deve ser INDEPENDENTE",
            "- Máximo de 1 trade por candle",
            "- Não sobrepor trades no mesmo candle",
            "- Ignorar sinais fracos ou ambíguos",
            "- Priorizar QUALIDADE sobre QUANTIDADE",
            "",
            "Gerenciamento de risco (AJUSTADO PARA SCALP):",
            "- Scalp:",
            " - Risco-retorno mínimo: 1:1 (take OBRIGATORIAMENTE >= stop)",
            " - Risco-retorno ideal: 1:2 (take = 2x o stop)",
            "- Day Trade curto:",
            " - Risco-retorno mínimo: 1:1 (take OBRIGATORIAMENTE >= stop)",
            " - Risco-retorno ideal: 1:2 (take = 2x o stop)",
            "",
            "- REGRA ABSOLUTA DE VALIDAÇÃO (NÃO NEGOCIÁVEL):",
            " - Se a distância do STOP for MAIOR que a distância do TAKE → INVALIDAR TRADE",
            " - Se o TAKE não for pelo menos IGUAL ao STOP → NÃO incluir o trade",
            " - Priorizar SEMPRE reduzir o STOP antes de tentar estender o TAKE",
            "",
            "- Stop Loss:",
            " - Baseado no último micro topo/fundo relevante",
            " - Preferencialmente posicionado PRÓXIMO às EMAs ou Bandas de Bollinger",
            " - Distância típica entre 1.0 e 3.0 dólares (priorizar o MENOR valor possível)",
            " - Nunca maior que a distância do take",
            " - Nunca posicionado fora da banda completa de Bollinger",
            "",
            "- Take Profit:",
            " - EMA 9, EMA 21, meio da Banda de Bollinger ou extremo do micro range",
            " - Priorizar Bandas de Bollinger como ALVO PRIMÁRIO",
            " - Take deve ficar alguns pontos ANTES ou LOGO APÓS a banda",
            " - Exemplo SELL: Banda inferior = 4578.90 → Take entre 4579.0 e 4580.5",
            " - Exemplo BUY: Banda superior - buffer técnico curto",
            " - Take DEVE respeitar a relação matemática com o stop (1:1 mínimo, 1:2 ideal)",
            "",
            "- NÃO utilizar resistências ou suportes macro (evitar lógica de swing trade)",
            "- NÃO posicionar stop ou take distante das médias ou bandas relevantes",
            "- Normalizar preços de entrada, stop e alvo para valores próximos e divisíveis por 0.5, 1 ou 2.5",
            "",
            "VALIDAÇÃO FINAL OBRIGATÓRIA PARA CADA TRADE:",
            "- Recalcular matematicamente stop e take antes de retornar",
            "- Se stop > take → NÃO incluir o trade",
            "- Se take < stop → NÃO incluir o trade",
            "- Se risco-retorno < 1:1 → NÃO incluir o trade",
            "",
            "Formato de saída:",
            "Retorne EXCLUSIVAMENTE um JSON válido no seguinte formato:",
            "",
            "{",
            ' "trades": [',
            "   {",
            '     "decision": "TRADE",',
            '     "type": "Buy" | "Sell",',
            '     "price": number,',
            '     "stop": number,',
            '     "take": number,',
            '     "datetime": "%Y-%m-%d %H:%M:%S",',
            '     "duration_minutes": number,',
            '     "tendency": "A favor" | "Contra",',
            '     "confidence": "Alta" | "Média" | "Baixa",',
            '     "justification": "Justificativa técnica curta e objetiva baseada nos candles e indicadores"',
            "   }",
            " ]",
            "}",
            "",
            "- Se nenhuma negociação válida existir nos últimos 50 minutos:",
            '{ "trades": [] }'
        ]

        prompt = "\n".join(lines)
        return prompt

    def analisar_grafico_day_trade(self, contador: int) -> CandleResponse:
        """
        Analisa um gráfico para Day Trade utilizando:
        - CCI
        - MACD
        - Médias móveis
        - Tendência
        - Price Action
        Retorna entrada, stop, take, tipo e tendência.
        """
        num_candles = 150
        indices = self.recuperarInfoGrafico(num_candles)
        if len(indices) > 0:
            prompt = self.generatePrompt(contador, indices, num_candles)
            response = self.perplexityClient.perguntar(prompt)
            # response = self.client.chat.completions.create(
            #     model="gpt-5.2",
            #     messages=[
            #         {
            #             "role": "system",
            #             "content": (
            #                 "Você é um trader profissional especializado em commodities, com foco em ouro (XAU/USD). "
            #                 "Responda de forma objetiva, direta e técnica, com baixa verbosidade. "
            #                 "Baseie análises exclusivamente em dados de mercado e conceitos consolidados. "
            #                 "Priorize fontes internacionais confiáveis: "
            #                 "LBMA, World Gold Council (WGC), CME Group, Bloomberg, Reuters, ICE, "
            #                 "Federal Reserve (Fed), U.S. Treasury, BIS e relatórios macroeconômicos oficiais. "
            #                 f"Utilize análise técnica aplicada ao gráfico {self.symbol} "
            #                 "Evite especulação. Diferencie fato, hipótese e simulação. "
            #                 "Sempre indique níveis-chave, viés, risco e ponto de invalidação."
            #             )
            #         },
            #         {
            #             "role": "user",
            #             "content": prompt
            #         }
            #     ],
            #     temperature=0.1
            # )

            try:
                content = response.choices[0].message.content.strip()
                candle = json.loads(content)
                return CandleResponse(
                    candle['price'],
                    candle['stop'],
                    candle['take'],
                    candle['type'],
                    candle['tendency'],
                    candle['datetime'],
                    candle['decision'],
                    candle['confidence'],
                    candle['justification']
                )
            except Exception as e:
                content = response['choices'][0]['message']['content'].strip()
                candle = json.loads(content)
                return CandleResponse(
                    candle['price'],
                    candle['stop'],
                    candle['take'],
                    candle['type'],
                    candle['tendency'],
                    candle['datetime'],
                    candle['decision'],
                    candle['confidence'],
                    candle['justification']
                )

        return None

    def getQtdPosicoes(self):
        """
        Verifica se há uma ordem (posição) aberta para o símbolo especificado.

        Retorna:
            True  -> se existe pelo menos uma posição aberta
            False -> se não houver nenhuma posição aberta
        """
        posicoes = mt5.positions_get(symbol=self.symbol)
        return len(posicoes)

    def recuperarInfoGrafico(self, numCandle: int):
        """
        Retorna:
        - último candle FECHADO
        - candle anterior
        """
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, numCandle)
        candles = []
        if rates is None or len(rates) < numCandle:
            return candles

        for i in range(numCandle):
            # Último candle fechado
            last_closed = rates[i]
            # previous = rates[-3]

            # dtPrev = datetime.fromtimestamp(previous['time']) - timedelta(hours=2)
            dtLast = datetime.fromtimestamp(last_closed['time']) - timedelta(hours=2)
            candles.append(
                Candle(
                    time=dtLast,
                    datetime_str=dtLast.strftime("%Y-%m-%d %H:%M:%S"),
                    open=last_closed['open'],
                    high=last_closed['high'],
                    low=last_closed['low'],
                    close=last_closed['close']
                )
            )

        return candles

    def candles_to_json_string(self, candles):
        data = [
            {
                "datetime": c.datetime_str,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close)
            }
            for c in candles
        ]
        return json.dumps(data, ensure_ascii=False)

    def executarOrdemImediata(self, tipo_ordem: str, sl: float, tp: float, volume=0, pattern=""):
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
        price = info.ask if tipo_ordem == 'BUY' else info.bid
        tipo = mt5.ORDER_TYPE_BUY if tipo_ordem == 'BUY' else mt5.ORDER_TYPE_SELL
        comentario = frase = f"Ordem pendente via Python {pattern}"
        # Prepara estrutura de ordem
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": tipo,
            "price": price,
            "sl": self.normalize_price(sl, 10),
            "tp": self.normalize_price(tp, 10),
            "deviation": 10,
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

    def normalize_price(self, price: float, digits: int) -> float:
        """
        Normaliza o preço para a quantidade correta de casas decimais do ativo.

        :param price: preço bruto calculado (ex: 4580.12783)
        :param digits: número de casas decimais do ativo (XAUUSD geralmente = 2)
        :return: preço normalizado
        """
        factor = 10 ** digits
        return round(price * factor) / factor

    def normalizar_data(self, data_entrada, formato='%Y-%m-%d %H:%M:%S'):
        if isinstance(data_entrada, str):
            return datetime.strptime(data_entrada, formato)
        elif isinstance(data_entrada, datetime):
            return data_entrada
        else:
            raise ValueError("Formato de data inválido.")

    def impedir_reposo(self):
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002

        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )

    def mt5_timeframe_to_string(self, timeframe: str) -> str:
        timeframe_map = {
            1: "M1",
            2: "M2",
            3: "M3",
            4: "M4",
            5: "M5",
            6: "M6",
            10: "M10",
            12: "M12",
            15: "M15",
            20: "M20",
            30: "M30",
            60: "H1",
            120: "H2",
            180: "H3",
            240: "H4",
            360: "H6",
            480: "H8",
            720: "H12",
            1440: "D1",
            10080: "W1",
            43200: "MN1"
        }

        return timeframe_map.get(timeframe, "UNKNOWN")

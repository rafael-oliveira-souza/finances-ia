import ctypes
import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from datetime import datetime, timedelta
from typing import Any

import MetaTrader5 as mt5

import DailyRiskManager
import RobotGpt
import SecureIdentifier


@dataclass
class DailyResult:
    allowed: bool
    daily_result: float
    type: str  # GAIN | LOSS
    message: str
    datetime: str


@dataclass
class RobotInfo:
    nome: str
    data_final: date
    data_inicial: date


SECRET_KEY = b'kF4xZJm3Qm9V0J6Z4Yy0c1l6F1N5dYkN0bZxP8Qk3oE='


class RobotGptExecutor:
    def __init__(self, hash: str, robotName: str, login: int, senha: str,
                 servidor: str, symbol: str = "XAUUSD", timeframe: Any = mt5.TIMEFRAME_M30,
                 volume: float = 0.01, numberRobots: int = 10, enableMoveStop: bool = False,
                 minimosNegociacao: float = None, ganhoMaximo: float = 100, perdaMaxima: float = 100):
        self.robotName = robotName
        self.symbol = symbol
        self.numberRobots = numberRobots
        self.enableMoveStop = enableMoveStop
        self.minimosNegociacao = minimosNegociacao
        self.ganhoMaximo = ganhoMaximo
        self.perdaMaxima = perdaMaxima
        self.robotGPT = RobotGpt.RobotGpt(login=login, senha=senha, servidor=servidor, symbol=symbol,
                                          timeframe=timeframe, volume=volume)
        self.risk = DailyRiskManager.DailyRiskManager()
        self.contador = 0
        self.contadorTempo = 0

        if hash.strip() != "jornada7913":
            service = SecureIdentifier.SecureIdentifier(SECRET_KEY)
            valor = service.recover_original(hash.strip())
            resultData = self.parse_robot_identifier(valor)
            if resultData.data_inicial >= date.today() >= resultData.data_final:
                print("Usuario Valido")
            else:
                print("Usuario invalido")
                raise RuntimeError("Usuario invalido. Solicite outro acesso ao Administrador")

    def _impedir_reposo(self):
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002

        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )

    def execute(self, segundos_espera=500):
        try:
            print(
                f"############################################ {self.robotName} ATIVO - Robos {self.robotGPT.getQtdPosicoes()} ################################################################################")
            while True:
                if self.contadorTempo > segundos_espera:
                    self.contadorTempo = 0

                if self.mercado_aberto_mt5(self.symbol):
                    self._impedir_reposo()
                    status = self.risk.pode_operar(
                        ganho_maximo=self.ganhoMaximo,
                        perda_maxima=self.perdaMaxima
                    )
                    if self.enableMoveStop:
                        self.move_stop()

                    if status["allowed"]:
                        if self.robotGPT.getQtdPosicoes() < self.numberRobots:
                            print()
                            print(
                                f"############################################ INICIO DE ANALISE ################################################################################")
                            results = self.robotGPT.analisar_grafico_day_trade(self.contador)
                            for result in results:
                                print(
                                    f"Decisao: {datetime.now()} - {result.decision} - {result.type} - {result.confidence} - Entrada:{result.price} - Stop: {result.stop} - Take: {result.take} - {result.justification}")
                                if result.decision != "NO_TRADE" and result.confidence != "Baixa":
                                    # Filtrar pelos últimos N minutos
                                    agora = datetime.now()
                                    limite = agora - timedelta(minutes=30)
                                    ptStop = abs(result.stop - result.price) / 0.01
                                    ptTake = abs(result.take - result.price) / 0.01
                                    dataNormalizada = self.robotGPT.normalizar_data(result.time)
                                    if (limite <= dataNormalizada <= agora):
                                        # and ptStop > self.minimosNegociacao and ptTake > self.minimosNegociacao)
                                        print()
                                        print(
                                            f"############################################ ENVIAR ORDEM ################################################################################")
                                        print(
                                            f"Ordem: {result.time} - {result.type} - {result.price} - {result.tendency} - {result.stop} - {result.take}")
                                        print(f"Decisao: {datetime.now()} - {result.decision} - {result.justification}")
                                        if result.type and result.type.upper():
                                            result = self.robotGPT.executarOrdemImediata(result.type.upper(),
                                                                                         result.stop,
                                                                                         result.take)
                                            self.contador = 0
                                        else:
                                            self.contador += 1
                                        print(
                                            f"############################################ ENVIAR ORDEM ################################################################################")
                            print(
                                f"############################################ FIM DE ANALISE ################################################################################")
                    else:
                        print(f"Limite {status["type"]} de trade diário atingido!")
                else:
                    print("Mercado fechado!")
                time.sleep(300)
                self.contadorTempo += 1
        except Exception as e:
            logging.error(e)

    def move_stop(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return

        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return

        if len(positions) <= 0:
            return

        print(
            f"############################################ INICIO MOVE STOP ################################################################################")
        for pos in positions:
            entry = pos.price_open
            stop = pos.sl
            take = pos.tp
            ticket = pos.ticket
            volume = pos.volume

            direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            price = tick.bid if direction == "BUY" else tick.ask

            if take == 0 or stop == 0:
                continue

            target_distance = abs(take - entry)
            protection_trigger = 0.40 * target_distance
            buffer = 0.5
            step = 3.0  # 300 pontos

            # ===============================
            # FASE 1 — PROTEÇÃO
            # ===============================
            if direction == "BUY":
                if price >= entry + protection_trigger and stop < entry:
                    new_sl = entry + buffer

                elif price >= stop + step:
                    new_sl = stop + step
                else:
                    continue

            else:  # SELL
                if price <= entry - protection_trigger and stop > entry:
                    new_sl = entry - buffer

                elif price <= stop - step:
                    new_sl = stop - step
                else:
                    continue

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": round(new_sl, 2),
                "tp": take,
            }

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Erro ao mover stop {ticket}: {result.comment}")
            else:
                print(f"Stop atualizado {ticket} → Antigo Stop: {stop} -> Novo Stop: {new_sl}")
        print(
            f"############################################ FIM MOVE STOP ################################################################################")

    def parse_robot_identifier(self, value: str) -> RobotInfo:
        pattern = (
            r"^(?P<nome>.+?)_data_inicial_(?P<df>\d{4}_\d{2}_\d{2})"
            r"_data_final_(?P<di>\d{4}_\d{2}_\d{2})$"
        )

        match = re.match(pattern, value)

        if not match:
            raise ValueError("Formato inválido do identificador")

        nome = match.group("nome")

        data_final = date.fromisoformat(match.group("df").replace("_", "-"))
        data_inicial = date.fromisoformat(match.group("di").replace("_", "-"))

        return RobotInfo(
            nome=nome,
            data_final=data_final,
            data_inicial=data_inicial
        )

    def mercado_aberto_mt5(self, symbol: str) -> bool:
        if datetime.now().weekday() >= 5:
            return False

        if not mt5.initialize():
            return False

        info = mt5.symbol_info(symbol)
        if info is None:
            return False

        return info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL

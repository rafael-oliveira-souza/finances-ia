from dataclasses import dataclass
from datetime import date

import MetaTrader5 as mt5

from RobotTrade import RobotGptExecutor


# import RobotGpt
# import SecureIdentifier

@dataclass
class RobotInfo:
    nome: str
    data_final: date
    data_inicial: date


# analisadorAccount2 = CandleAnalyzer(login=541072629, senha='LH5GG4jA*6?@i', servidor='FTMO-Server4',
#                                     symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, volume=0.05,
#                                     enableMACD=True, enableCCI=True, enableRSI=True,
#                                     enableADX=True,
#                                     enableBollinger=True, enablePatternAnalysis=True, enableVerifyDivergence=True,
#                                     enableMartingale=300, stopMinimo=300, maximoPerdaDiaria=300,
#                                     percentRiskAndProfit=100)
executor = RobotGptExecutor.RobotGptExecutor(hash='jornada7913', robotName='JourneyBot', login=531126955,
                                             senha='*Mc*@n9S!!JJ5v',
                                             servidor='FTMO-Server3', symbol="XAUUSD",
                                             timeframe=mt5.TIMEFRAME_M30, volume=0.2,
                                             numberRobots=10, enableMoveStop=True,
                                             minimosNegociacao=300, perdaMaxima=0,
                                             ganhoMaximo=0)
executor.execute()

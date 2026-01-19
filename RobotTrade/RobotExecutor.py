import ctypes
import time
from datetime import timedelta, datetime
from typing import List

from CandleAnalyzer import CandleAnalyzer, normalizar_data
from DayTracker import DayTracker
from SignalManagement import SignalManagement
from Simulador import Simulador

num_candles = 30
qtd_minutes = 20

class RobotExecutor:
    def __init__(self, analyzers: List[CandleAnalyzer],
                 enableMoveStop=True, enableSimulation=False,
                 dataInicio=None, dataFim=None, maxOrders=5):

        if not isinstance(analyzers, list) or not all(isinstance(a, CandleAnalyzer) for a in analyzers):
            raise TypeError("O parâmetro 'analyzers' deve ser uma lista de objetos CandleAnalyzer.")

        self.analyzers = analyzers
        self.enableMoveStop = enableMoveStop
        self.enableSimulation = enableSimulation
        self.dataInicio = dataInicio
        self.dataFim = dataFim
        self.maxOrders = maxOrders

    def _impedir_reposo(self):
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002

        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )

    def execute(self):
        print(f"Iniciando Journey Bot")
        day_tracker = DayTracker()
        for analyzer in self.analyzers:
            countValue = 0
            historico = []
            if self.enableSimulation:
                sim = Simulador(analyzer, start_date=self.dataInicio, end_date=self.dataFim)
                sim.salvar_estatisticas_excel()
            else:
                self._impedir_reposo()
                signalManagement = SignalManagement()
                while True:
                    if countValue > 50:
                        countValue = 0

                    if countValue == 0:
                        print("Robo em execucao")

                    dataframe = analyzer.recuperarInfoGrafico(num_candles=num_candles, start_date=0, end_date=0)
                    # zonas = self.analyzer.get_grouped_highs_lows(dataframe, group_size, show_debug=False)
                    resultados = analyzer.analisarTodosIndicadores(dataframe, 0)
                    if analyzer.existe_ordem_aberta() and self.enableMoveStop:
                        analyzer.move_stop(70, 30)

                    if len(resultados) > 0:
                        for resultado in analyzer.filtrarPorMinutosEPreco(resultados, qtd_minutes):
                            data, entrada, tipo, stop, take, metodo, timestamp = resultado
                            print()
                            print(
                                f"############################################ ORDEM DEFINIDA #####################################################################")
                            horarioBrasilia = normalizar_data(data, "%d/%m/%Y %H:%M") - timedelta(hours=6)
                            print(
                                f"Resultados nos ultimos {qtd_minutes} min: {(horarioBrasilia)} - {tipo} - {entrada} - {metodo} - {stop} - {take} - Robot {analyzer.login}")
                            print(
                                f"############################################ ORDEM DEFINIDA #####################################################################")

                    resultadoFiltrado = analyzer.filtrarPorMinutosEPreco(resultados, qtd_minutes)
                    if len(resultadoFiltrado) > 0:
                        sinaisGerenciados = signalManagement.adicionar_sinais(resultadoFiltrado)
                        for res in sinaisGerenciados:
                            data, entrada, tipo, stop, take, metodo, timestamp, sinalId = res

                            if self.maxOrders == 0 or analyzer.getQtdPosicoes() <= self.maxOrders:
                                if not signalManagement.ja_executado(sinal_id=sinalId):
                                    print()
                                    print(
                                        f"############################################ ENVIAR ORDEM ################################################################################")
                                    horarioBrasilia = normalizar_data(data, "%d/%m/%Y %H:%M") - timedelta(hours=6)
                                    print(
                                        f"Ordem: {(horarioBrasilia)} - {tipo} - {entrada} - {metodo} - {stop} - {take} - Robot {analyzer.login}")

                                    result = analyzer.executarOrdemImediata(tipo, stop, take)
                                    if result:
                                        signalManagement.marcar_como_executado(sinal_id=sinalId)
                                        historico.append((data, entrada, tipo, stop, take, metodo, timestamp))
                                    print(
                                        f"############################################ ENVIAR ORDEM ################################################################################")
                        # else:
                        # print(f"Sem resultados - Robot - {analyzer.login}")
                    countValue += 1
                    actualDay = datetime.now()
                    if len(historico) > 0 and day_tracker.eh_novo_dia(actualDay):
                        print(
                            f"############################################ GERANDO HISTORICO ################################################################################")
                        dataInicio, dataFim = day_tracker.inicio_e_fim_do_dia()
                        sim = Simulador(analyzer, start_date=dataInicio, end_date=dataFim, sinaisExecutados=historico)
                        sim.salvar_estatisticas_excel()
                        historico = []

                    time.sleep(30)

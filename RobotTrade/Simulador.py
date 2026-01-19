import os
import winreg
from datetime import datetime

import pandas as pd

import CandleAnalyzer


class Simulador:
    def __init__(self, candle_analyzer: CandleAnalyzer, start_date: datetime, end_date: datetime, sinaisExecutados=None):
        df = candle_analyzer.recuperarInfoGrafico(num_candles=0, start_date=start_date, end_date=end_date)
        self.df = df.copy()
        self.volume = candle_analyzer.volume
        self.candle_analyzer = candle_analyzer
        self.resultados = []

        if sinaisExecutados is None or len(sinaisExecutados) == 0:
            self.sinais = candle_analyzer.analisarTodosIndicadores(self.df, 0)
        else:
            self.sinais = sinaisExecutados

    def get_downloads_folder(self):
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            return winreg.QueryValueEx(key, downloads_guid)[0]

    def _simular(self):
        if len(self.sinais) > 0:
            for sinal in self.sinais:
                timeBR, close, tipo, stop, take, padrao, time = sinal
                resultado = self._avaliar_ordem(time, tipo, close, stop, take, padrao)
                self.resultados.append(resultado)

    def _avaliar_ordem(self, data_sinal, tipo, entrada, stop, take, padrao):
        df = self.df
        if isinstance(data_sinal, str):
            data_sinal = CandleAnalyzer.normalizar_data(data_sinal, "%d/%m/%Y %H:%M")

        idx_inicio = df[df['time'] >= data_sinal].index
        if len(idx_inicio) == 0:
            return {"data": data_sinal, "tipo": tipo, "metodo": padrao, "resultado": "ignorado", "lucro": 0}

        for i in idx_inicio:
            for _, candle in df.iloc[i:].iterrows():
                high = candle['high']
                low = candle['low']
                if tipo == 'COMPRA':
                    if low <= stop:
                        lucro = self.candle_analyzer.calcular_valor_em_dolares((stop - entrada), self.volume)
                        return {"data": data_sinal, "tipo": tipo, "metodo": padrao, "resultado": "STOP", "lucro": lucro}
                    elif high >= take:
                        lucro = self.candle_analyzer.calcular_valor_em_dolares((take - entrada), self.volume)
                        return {"data": data_sinal, "tipo": tipo, "metodo": padrao, "resultado": "TAKE", "lucro": lucro}
                elif tipo == 'VENDA':
                    if high >= stop:
                        lucro = self.candle_analyzer.calcular_valor_em_dolares((entrada - stop), self.volume)
                        return {"data": data_sinal, "tipo": tipo, "metodo": padrao, "resultado": "STOP", "lucro": lucro}
                    elif low <= take:
                        lucro = self.candle_analyzer.calcular_valor_em_dolares((entrada - take), self.volume)
                        return {"data": data_sinal, "tipo": tipo, "metodo": padrao, "resultado": "TAKE", "lucro": lucro}

        return {"data": data_sinal, "tipo": tipo, "metodo": padrao, "resultado": "ABERTO", "lucro": 0}

    def gerarEstatisticas(self):
        self._simular()
        df_result = pd.DataFrame(self.resultados)
        total = len(df_result)

        if total == 0:
            print("Nenhuma estatistica foi gerada, pois nenhum ordem foi acionada.")
            return

        ganhos = (df_result['lucro'] > 0).sum()
        perdas = (df_result['lucro'] < 0).sum()
        acuracia_geral = round(ganhos / total * 100, 2) if total > 0 else 0
        lucro_total = round(df_result['lucro'].sum(), 2)

        stats = df_result.groupby('metodo')['lucro'].agg([
            ('total_ops', 'count'),
            ('ganhos', lambda x: (x > 0).sum()),
            ('perdas', lambda x: (x < 0).sum()),
            ('lucro', 'sum')
        ])
        stats['acuracia_%'] = round(stats['ganhos'] / stats['total_ops'] * 100, 2)

        return {
            "total": total,
            "ganhos": ganhos,
            "perdas": perdas,
            "acuracia_total_%": acuracia_geral,
            "lucro_total": lucro_total,
            "acuracia_por_metodo": stats,
            "resultado_detalhado": df_result
        }

    def salvar_estatisticas_excel(self):
        stats = self.gerarEstatisticas()

        df = stats.get('resultado_detalhado')
        if df is None:
            raise KeyError(f"'resultado_detalhado' não encontrado em stats: {list(stats.keys())}")

        resumo = pd.DataFrame({
            'Métrica': ['Total Ops', 'Ganhos', 'Perdas', 'Acurácia Geral (%)', 'Lucro Total'],
            'Valor': [stats['total'], stats['ganhos'], stats['perdas'],
                      stats['acuracia_total_%'], stats['lucro_total']]
        })

        df_metodo = stats['acuracia_por_metodo'].reset_index()

        downloads = self.get_downloads_folder()
        nome_arquivo = f"JourneyBot_estatisticas_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        caminho = os.path.join(downloads, nome_arquivo)

        with pd.ExcelWriter(caminho, engine="xlsxwriter") as writer:
            resumo.to_excel(writer, sheet_name="Resumo", index=False)
            df_metodo.to_excel(writer, sheet_name="Por_Metodo", index=False)
            df.to_excel(writer, sheet_name="Detalhado", index=False)

        print("Arquivo salvo em:", caminho)
        return caminho

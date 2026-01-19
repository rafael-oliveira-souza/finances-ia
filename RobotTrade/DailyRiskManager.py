from datetime import datetime, time

import MetaTrader5 as mt5


class DailyRiskManager:

    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("Não foi possível inicializar o MetaTrader 5")

    def _inicio_fim_dia(self):
        hoje = datetime.now().date()
        inicio = datetime.combine(hoje, time.min)
        fim = datetime.combine(hoje, time.max)
        return inicio, fim

    def _resultado_diario(self) -> float:
        inicio, fim = self._inicio_fim_dia()
        deals = mt5.history_deals_get(inicio, fim)

        if deals is None:
            return 0.0

        return round(sum(d.profit for d in deals), 2)

    def pode_operar(self, ganho_maximo: float, perda_maxima: float) -> dict:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if perda_maxima == 0 or ganho_maximo == 0:
            return {
                "allowed": True,
                "type": "GAIN",
                "daily_result": round(0, 2),
                "message": "Dentro do limite diário",
                "datetime": now
            }

        resultado = self._resultado_diario()
        if resultado >= abs(ganho_maximo):
            return {
                "allowed": False,
                "type": "GAIN",
                "daily_result": resultado,
                "message": "STOP DE GANHO diário atingido",
                "datetime": now
            }

        if resultado <= -abs(perda_maxima):
            return {
                "allowed": False,
                "type": "LOSS",
                "daily_result": resultado,
                "message": "STOP DE PERDA diário atingido",
                "datetime": now
            }

        return {
            "allowed": True,
            "type": "GAIN" if resultado >= 0 else "LOSS",
            "daily_result": resultado,
            "message": "Dentro do limite diário",
            "datetime": now
        }

from datetime import datetime, time

from CandleAnalyzer import normalizar_data


class DayTracker:
    def __init__(self):
        self.ultimo_dia = None

    def eh_novo_dia(self, data_str):
        """
        Verifica se a data informada representa um novo dia.

        :param data_str: Data no formato 'YYYY-MM-DD'
        :return: True se for um novo dia, False caso contrário
        """
        data = normalizar_data(data_str, "%Y-%m-%d")

        if self.ultimo_dia != data:
            self.ultimo_dia = data
            return True
        return False

    def inicio_e_fim_do_dia(self):
        """
        Retorna o início e fim do dia atual como datetimes.

        :return: (inicio_do_dia, fim_do_dia) ou None se nenhum dia foi definido ainda
        """
        if self.ultimo_dia is None:
            return None

        inicio = datetime.combine(self.ultimo_dia, time.min)  # 00:00:00
        fim = datetime.combine(self.ultimo_dia, time.max)  # 23:59:59.999999
        return inicio, fim
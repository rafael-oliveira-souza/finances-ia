import uuid


class SignalManagement:
    def __init__(self):
        self.sinais_status = {}  # mapa: id_sinal -> executado (True/False)

    def _gerar_id_unico(self, texto_base):
        """
        Gera um UUID baseado em uma string (determinístico),
        e garante que não foi usado ainda.
        """
        base = str(texto_base)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

    def adicionar_sinais(self, lista_de_sinais):
        """
        Adiciona novos sinais ao mapa, atribuindo ID aleatório se não houver.
        Cada sinal deve conter um campo opcional 'id'.
        """
        sinaisFiltrados = []
        for sinal in lista_de_sinais:
            data, entrada, tipo, stop, take, metodo, timestamp = sinal
            if len(sinal) == 7:
                sinal_id = self._gerar_id_unico(f'{str(data)}_{str(entrada)}_{tipo}_{metodo}')
                novo_sinal = sinal + (sinal_id,)  # cria nova tupla com 8 elementos
                sinaisFiltrados.append(novo_sinal)
                if sinal_id not in self.sinais_status:
                    self.sinais_status[sinal_id] = False
            else:
                sinaisFiltrados.append(sinal)

        return  sinaisFiltrados

    def marcar_como_executado(self, sinal_id):
        if sinal_id in self.sinais_status:
            self.sinais_status[sinal_id] = True

    def ja_executado(self, sinal_id):
        for chave, valor in self.sinais_status.items():
            if chave == sinal_id:
                return valor
        return False

    def obter_status(self):
        return self.sinais_status

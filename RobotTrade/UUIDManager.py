import base64
import json
from datetime import datetime


class UUIDManager:
    def __init__(self):
        self.uuid_map = [
            'db9cb86b-d545-41f4-8fec-c32badd50d7d',
            '776176c9-4a6b-544c-b9d2-6c57005f4e72'
        ]  # opcional: salva os dados se quiser decodificar depois

    def codificar(self, nome: str, data_inicio: datetime, data_fim: datetime) -> str:
        obj = {"nome": nome, "dataInicio": data_inicio.isoformat(), "dataFim": data_fim.isoformat()}
        raw = json.dumps(obj, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        b64 = base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')
        return f"{b64}"

    def decodificar(self, u: str) -> dict:
        pad = '=' * (-len(u) % 4)  # adiciona padding correto
        raw = base64.urlsafe_b64decode(u + pad)
        return json.loads(raw.decode('utf-8'))

    def validarUUID(self, uuid: str):
        if uuid not in self.uuid_map:
            try:
                result = self.decodificar(uuid)
                if 'JORNADA_ROBOT_TRADE' in result['nome']:
                    if not datetime.fromisoformat(result['dataInicio']) <= datetime.now() <= datetime.fromisoformat(
                            result['dataFim']):
                        raise ValueError("Credencial Invalida!!")
                else:
                    raise ValueError("Credencial Invalida!!")
            except ValueError:
                raise ValueError("Credencial Invalida!!")

        print('Credencial Autenticada!')

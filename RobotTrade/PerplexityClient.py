import requests
from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()

# Endpoint da Sonar API (confira a doc atual)
PPLX_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityClient:
    def __init__(self):
        self.api_key = "TesteAPisdaiji"
        if not self.api_key:
            raise ValueError("PPLX_API_KEY não encontrada no .env")

    def perguntar(self, pergunta: str, modelo: str = "sonar-pro") -> dict:
        """Faz uma pergunta ao Perplexity e retorna a resposta."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": modelo,
            "messages": [
                {
                    "role": "user",
                    "content": pergunta
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "trade_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "enum": ["TRADE", "NO_TRADE"]
                            },
                            "type": {
                                "oneOf": [
                                    {"type": "string", "enum": ["Buy", "Sell"]},
                                    {"type": "null"}
                                ]
                            },
                            "price": {
                                "oneOf": [
                                    {"type": "number"},
                                    {"type": "null"}
                                ]
                            },
                            "stop": {
                                "oneOf": [
                                    {"type": "number"},
                                    {"type": "null"}
                                ]
                            },
                            "take": {
                                "oneOf": [
                                    {"type": "number"},
                                    {"type": "null"}
                                ]
                            },
                            "datetime": {
                                "oneOf": [
                                    {
                                        "type": "string",
                                        "pattern": "^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$"
                                    },
                                    {
                                        "type": "null"
                                    }
                                ]
                            },
                            "tendency": {
                                "oneOf": [
                                    {"type": "string", "enum": ["A favor", "Contra"]},
                                    {"type": "null"}
                                ]
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["ALTA", "MEDIA", "BAIXA"]
                            },
                            "justification": {
                                "type": "string"
                            }
                        },
                        "required": ["decision", "confidence", "justification"],
                        "additionalProperties": False
                    }
                }
            }
        }

        response = requests.post(PPLX_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Erro API: {response.status_code} - {response.text}")


# Exemplo de uso (sua API endpoint)
def endpoint_pergunta(pergunta: str):
    """Simula um endpoint da sua API."""
    client = PerplexityClient()
    resultado = client.perguntar(pergunta)

    # Extrai a resposta
    resposta = resultado["choices"][0]["message"]["content"]
    return resposta


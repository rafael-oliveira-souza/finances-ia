import requests
from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()

# Endpoint da Sonar API (confira a doc atual)
PPLX_URL = "https://api.perplexity.ai/chat/completions"


class PerplexityClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
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
                    "name": "trades_list",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "trades": {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        # =====================
                                        # CASO: TRADE VÁLIDO
                                        # =====================
                                        {
                                            "type": "object",
                                            "properties": {
                                                "decision": {
                                                    "type": "string",
                                                    "enum": ["TRADE", "NO_TRADE"]
                                                },
                                                "type": {
                                                    "type": "string",
                                                    "enum": ["Buy", "Sell"]
                                                },
                                                "price": {
                                                    "type": "number"
                                                },
                                                "stop": {
                                                    "type": "number"
                                                },
                                                "take": {
                                                    "type": "number"
                                                },
                                                "datetime": {
                                                    "type": "string",
                                                    "pattern": "^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$"
                                                },
                                                "duration_minutes": {
                                                    "type": "number",
                                                    "minimum": 1,
                                                    "maximum": 90
                                                },
                                                "tendency": {
                                                    "type": "string",
                                                    "enum": ["A favor", "Contra"]
                                                },
                                                "confidence": {
                                                    "type": "string",
                                                    "enum": ["ALTA", "MEDIA", "BAIXA"]
                                                },
                                                "justification": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": [
                                                "decision",
                                                "type",
                                                "price",
                                                "stop",
                                                "take",
                                                "datetime",
                                                "duration_minutes",
                                                "tendency",
                                                "confidence",
                                                "justification"
                                            ],
                                            "additionalProperties": False
                                        },

                                        # =====================
                                        # CASO: NO_TRADE
                                        # =====================
                                        {
                                            "type": "object",
                                            "properties": {
                                                "decision": {
                                                    "type": "string",
                                                    "enum": ["TRADE", "NO_TRADE"]
                                                },
                                                "type": {
                                                    "type": ["string", "null"]
                                                },
                                                "price": {
                                                    "type": ["number", "null"]
                                                },
                                                "stop": {
                                                    "type": ["number", "null"]
                                                },
                                                "take": {
                                                    "type": ["number", "null"]
                                                },
                                                "datetime": {
                                                    "type": "string",
                                                    "pattern": "^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$"
                                                },
                                                "duration_minutes": {
                                                    "type": ["number", "null"]
                                                },
                                                "tendency": {
                                                    "type": ["string", "null"]
                                                },
                                                "confidence": {
                                                    "type": "string",
                                                    "enum": ["ALTA", "MEDIA", "BAIXA"]
                                                },
                                                "justification": {
                                                    "type": "string"
                                                }
                                            },
                                            "required": [
                                                "decision",
                                                "datetime",
                                                "confidence",
                                                "justification"
                                            ],
                                            "additionalProperties": False
                                        }
                                    ]
                                }
                            }
                        },
                        "required": ["trades"],
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

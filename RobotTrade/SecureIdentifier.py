import uuid

from cryptography.fernet import Fernet


class SecureIdentifier:

    def __init__(self, secret_key: bytes):
        self.fernet = Fernet(secret_key)

    def encrypt(self, raw_value: str) -> str:
        return self.fernet.encrypt(raw_value.encode()).decode()

    def decrypt(self, encrypted_value: str) -> str:
        return self.fernet.decrypt(encrypted_value.encode()).decode()

    def generate_uuid_from_value(self, raw_value: str) -> str:
        """
        UUID determinístico baseado no valor original
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_value))

    def generate_secure_id(self, raw_value: str) -> dict:
        try:
            encrypted = self.encrypt(raw_value)
            uuid_value = self.generate_uuid_from_value(encrypted)

            return {
                "uuid": uuid_value,
                "encrypted": encrypted
            }
        except ValueError:
            raise ValueError("Não foi possível autenticar o usuario")

    def recover_original(self, encrypted_value: str) -> str:
        try:
            return self.decrypt(encrypted_value)
        except ValueError:
            raise ValueError("Não foi possível autenticar o usuario")
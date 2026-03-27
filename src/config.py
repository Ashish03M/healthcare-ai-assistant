import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    groq_api_key: str
    model_name: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    sender_email: str

    @classmethod
    def from_env(cls) -> "Settings":
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required. Get one free at https://console.groq.com")

        return cls(
            groq_api_key=groq_api_key,
            model_name=os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile"),
            smtp_host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ.get("SMTP_USER", ""),
            smtp_password=os.environ.get("SMTP_PASSWORD", ""),
            sender_email=os.environ.get("SENDER_EMAIL", ""),
        )

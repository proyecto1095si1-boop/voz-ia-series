"""
Configuración centralizada de VoxSync.
"""

from pathlib import Path
from typing import Optional
import os

from pydantic import BaseModel, Field


class VoxSyncConfig(BaseModel):
    """Configuración principal del producto."""

    # Modelos
    whisper_model: str = Field(default="large-v3", description="Modelo de Whisper a usar")
    device: str = Field(default="auto", description="auto | cpu | cuda")
    compute_type: str = Field(default="auto", description="float16 | int8 | auto")

    # Idiomas
    source_lang: str = Field(default="en")
    target_lang: str = Field(default="es")

    # Rutas
    output_dir: Path = Field(default=Path("output"))
    models_dir: Path = Field(default=Path("models"))
    temp_dir: Path = Field(default=Path("temp"))

    # Comportamiento
    keep_audio: bool = Field(default=False)
    batch_size: int = Field(default=16)
    max_speakers: Optional[int] = Field(default=None, description="Límite de hablantes (None = automático)")

    # Hugging Face
    hf_token: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def resolve_compute_type(self) -> str:
        if self.compute_type != "auto":
            return self.compute_type
        return "float16" if self.resolve_device() == "cuda" else "int8"

    @classmethod
    def from_env(cls) -> "VoxSyncConfig":
        """Carga configuración desde variables de entorno."""
        return cls(
            whisper_model=os.getenv("VOXSYNC_MODEL", "large-v3"),
            device=os.getenv("VOXSYNC_DEVICE", "auto"),
            source_lang=os.getenv("VOXSYNC_SOURCE_LANG", "en"),
            target_lang=os.getenv("VOXSYNC_TARGET_LANG", "es"),
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        )


def get_default_config() -> VoxSyncConfig:
    return VoxSyncConfig.from_env()

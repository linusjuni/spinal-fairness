from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # General settings
    RANDOM_SEED: int = Field(default=42)

    # Paths — CSpineSeg source data
    DATA_DIR: Path = Path("/work3/s225224/data/cspineseg")
    OUTPUT_DIR: Path = Path("outputs")

    # Paths — nnU-Net (read from env vars set by the user)
    nnUNet_raw: Path = Field(default=Path("/work3/s225224/nnunet/raw"))
    nnUNet_preprocessed: Path = Field(default=Path("/work3/s225224/nnunet/preprocessed"))
    nnUNet_results: Path = Field(default=Path("/work3/s225224/nnunet/results"))

    # Paths — pretrained encoder weights (MedicalNet, RadImageNet, etc.)
    MODELS_DIR: Path = Path("/work3/s225224/models")

    @property
    def raw_dir(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def extracted_dir(self) -> Path:
        return self.DATA_DIR / "extracted"

    @property
    def annotation_dir(self) -> Path:
        return self.extracted_dir / "annotation"

    @property
    def segmentation_dir(self) -> Path:
        return self.extracted_dir / "segmentation"

    @property
    def structured_dir(self) -> Path:
        return self.extracted_dir / "structured"

    @property
    def processed_dir(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.DATA_DIR / "splits"


settings = Settings()

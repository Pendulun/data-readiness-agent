from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    n_rows: int = Field(description="Quantidade de linhas na base")
    n_columns: int = Field(description="Quantidade de colunas na base")
    columns_types: dict[str, str] = Field(description="Tipos das colunas")
    null_counts: dict[str, int] = Field(
        description="Quantidade de nulos por coluna")
    unique_counts: dict[str, int] = Field(
        description="Quantidade de valores únicos por coluna")
    samples: dict[str, list] = Field(description="Amostra de dados por coluna")

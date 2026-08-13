from pathlib import Path

import pytest

from latesignal.data.config import load_data_config
from latesignal.errors import ConfigurationError


def test_default_schema_contract_has_exactly_23_unique_fields() -> None:
    config = load_data_config(Path("configs/data.yaml"))

    assert len(config.schema.fields) == 23
    assert len(set(config.schema.fields)) == 23
    assert config.dataset.expected_bytes == 2_002_864_638
    assert config.dataset.expected_sha256 is None


def test_unknown_configuration_key_fails_closed(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "version: 1\ndataset: {}\narchive_limits: {}\nschema: {}\nsurprise: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="invalid keys"):
        load_data_config(config_path)

"""The .env file is read without any extra package."""
import os
from config import load_env_file


def test_env_file_is_loaded_and_never_overrides(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\n\nECX_T_A=hello\nECX_T_B = "quoted value"\n  ECX_T_C=pa#ss word\n#ECX_T_D=off\nECX_T_E=fromfile\nnot a line\n', encoding="utf-8")
    for k in ("ECX_T_A", "ECX_T_B", "ECX_T_C", "ECX_T_D"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ECX_T_E", "already-set")
    load_env_file(f)
    assert os.environ["ECX_T_A"] == "hello" and os.environ["ECX_T_B"] == "quoted value" and os.environ["ECX_T_C"] == "pa#ss word"
    assert "ECX_T_D" not in os.environ and os.environ["ECX_T_E"] == "already-set"
    for k in ("ECX_T_A", "ECX_T_B", "ECX_T_C"):
        monkeypatch.delenv(k, raising=False)


def test_missing_env_file_is_fine(tmp_path):
    load_env_file(tmp_path / "nope.env")

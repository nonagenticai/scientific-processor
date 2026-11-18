"""Tests for config.py — env parsing helpers and Settings dataclass."""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _get_int
# ---------------------------------------------------------------------------


class TestGetInt:
    def test_returns_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_INT_VAR", raising=False)
        from config import _get_int

        assert _get_int("TEST_INT_VAR", 42) == 42

    def test_returns_parsed_int(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "7")
        from config import _get_int

        assert _get_int("TEST_INT_VAR", 42) == 7

    def test_returns_default_on_non_numeric(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "not_a_number")
        from config import _get_int

        assert _get_int("TEST_INT_VAR", 99) == 99

    def test_returns_default_on_float_string(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "3.14")
        from config import _get_int

        assert _get_int("TEST_INT_VAR", 10) == 10

    def test_returns_default_on_empty_string(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "")
        from config import _get_int

        assert _get_int("TEST_INT_VAR", 5) == 5


# ---------------------------------------------------------------------------
# _get_float
# ---------------------------------------------------------------------------


class TestGetFloat:
    def test_returns_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_FLOAT_VAR", raising=False)
        from config import _get_float

        assert _get_float("TEST_FLOAT_VAR", 1.5) == 1.5

    def test_returns_parsed_float(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "2.718")
        from config import _get_float

        assert _get_float("TEST_FLOAT_VAR", 0.0) == pytest.approx(2.718)

    def test_returns_default_on_non_numeric(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "abc")
        from config import _get_float

        assert _get_float("TEST_FLOAT_VAR", 9.9) == 9.9

    def test_returns_default_on_empty_string(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "")
        from config import _get_float

        assert _get_float("TEST_FLOAT_VAR", 3.0) == 3.0

    def test_parses_integer_string_as_float(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "42")
        from config import _get_float

        assert _get_float("TEST_FLOAT_VAR", 0.0) == 42.0


# ---------------------------------------------------------------------------
# _get_bool
# ---------------------------------------------------------------------------


class TestGetBool:
    def test_returns_default_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_BOOL_VAR", raising=False)
        from config import _get_bool

        assert _get_bool("TEST_BOOL_VAR", True) is True
        assert _get_bool("TEST_BOOL_VAR", False) is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "True", "YES", "ON", " true "])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("TEST_BOOL_VAR", value)
        from config import _get_bool

        assert _get_bool("TEST_BOOL_VAR", False) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "False", "NO", "OFF", " false "])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("TEST_BOOL_VAR", value)
        from config import _get_bool

        assert _get_bool("TEST_BOOL_VAR", True) is False

    def test_invalid_string_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL_VAR", "maybe")
        from config import _get_bool

        assert _get_bool("TEST_BOOL_VAR", True) is True
        assert _get_bool("TEST_BOOL_VAR", False) is False

    def test_empty_string_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL_VAR", "")
        from config import _get_bool

        # "" stripped and lowered is "" — not in truthy or falsy sets → returns default
        assert _get_bool("TEST_BOOL_VAR", True) is True


# ---------------------------------------------------------------------------
# Settings defaults and env overrides
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults_no_env(self, monkeypatch):
        """Settings uses sane defaults when no env vars are set."""
        # Clear the lru_cache so a fresh Settings is built
        from config import Settings, get_settings

        get_settings.cache_clear()

        # Remove env vars that would override defaults. Keep ENVIRONMENT
        # set to "test" so the production fail-fast validator in
        # Settings.__post_init__ does not trip — see
        # gitops-production/docs/config-externalization-policy.md.
        for key in [
            "SCIENTIFIC_PROCESSOR_VERSION",
            "SCIENTIFIC_PROCESSOR_SERVICE_NAME",
            "ENABLE_DOCS",
            "DEFAULT_TENANT_ID",
            "SCIENTIFIC_PROCESSOR_BIND_PORT",
            "SCIENTIFIC_PROCESSOR_REQUEST_TIMEOUT",
        ]:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("ENVIRONMENT", "test")

        s = Settings()
        assert s.version == "0.1.0"
        assert s.service_name == "scientific-processor"
        assert s.bind_port == 8808
        assert s.request_timeout == 300.0
        assert s.default_tenant_id == "0195f48b-d000-7000-8000-000000000001"

    def test_env_override(self, monkeypatch):
        from config import Settings, get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_VERSION", "2.0.0")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_BIND_PORT", "9999")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_REQUEST_TIMEOUT", "60.5")
        monkeypatch.setenv("ENABLE_DOCS", "true")

        s = Settings()
        assert s.version == "2.0.0"
        assert s.bind_port == 9999
        assert s.request_timeout == pytest.approx(60.5)
        assert s.enable_docs is True

    def test_get_settings_cached(self, monkeypatch):
        from config import get_settings

        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


class TestProductionValidator:
    """`Settings.__post_init__` fails fast when required env is missing in production."""

    def _clear_required_env(self, monkeypatch):
        for key in [
            "SCIENTIFIC_PROCESSOR_GRAPH_ADAPTER_URL",
            "SCIENTIFIC_PROCESSOR_PYTORCH_BACKEND_URL",
            "SCIENTIFIC_PROCESSOR_INFERENCE_ROUTER_URL",
            "SCIENTIFIC_PROCESSOR_DISCOVERY_ENGINE_URL",
            "SCIENTIFIC_PROCESSOR_LEAN4_PROVER_URL",
            "SCIENTIFIC_PROCESSOR_MINIO_ENDPOINT",
            "SCIENTIFIC_PROCESSOR_MINIO_ACCESS_KEY",
            "SCIENTIFIC_PROCESSOR_SANDBOX_IMAGE",
            "SCIENTIFIC_PROCESSOR_DOWNSTREAM_LICENSE_TOKEN",
            "OTLP_TRACES_ENDPOINT",
        ]:
            monkeypatch.delenv(key, raising=False)

    def test_production_missing_required_raises(self, monkeypatch):
        from config import Settings, get_settings

        get_settings.cache_clear()
        self._clear_required_env(monkeypatch)
        monkeypatch.setenv("ENVIRONMENT", "production")

        with pytest.raises(RuntimeError, match="missing required fields in production"):
            Settings()

    def test_production_with_all_required_succeeds(self, monkeypatch):
        from config import Settings, get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_GRAPH_ADAPTER_URL", "http://gx:1")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_PYTORCH_BACKEND_URL", "http://px:1")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_INFERENCE_ROUTER_URL", "http://ix:1")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_DISCOVERY_ENGINE_URL", "http://dx:1")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_LEAN4_PROVER_URL", "http://lx:1")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_MINIO_ENDPOINT", "mx:1")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_MINIO_ACCESS_KEY", "k")
        monkeypatch.setenv("OTLP_TRACES_ENDPOINT", "http://ox:1")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_DOWNSTREAM_LICENSE_TOKEN", "svc-token")

        s = Settings()
        assert s.environment == "production"
        assert s.graph_adapter_url == "http://gx:1"

    def test_production_without_service_licence_fails_boot(self, monkeypatch):
        """lean4-prover (GPU backend) rejects tokenless calls, so a production
        boot with no service licence must fail loud rather than 401 later."""
        from config import Settings, get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("ENVIRONMENT", "production")
        for k, v in [
            ("SCIENTIFIC_PROCESSOR_GRAPH_ADAPTER_URL", "http://gx:1"),
            ("SCIENTIFIC_PROCESSOR_PYTORCH_BACKEND_URL", "http://px:1"),
            ("SCIENTIFIC_PROCESSOR_INFERENCE_ROUTER_URL", "http://ix:1"),
            ("SCIENTIFIC_PROCESSOR_DISCOVERY_ENGINE_URL", "http://dx:1"),
            ("SCIENTIFIC_PROCESSOR_LEAN4_PROVER_URL", "http://lx:1"),
            ("SCIENTIFIC_PROCESSOR_MINIO_ENDPOINT", "mx:1"),
            ("SCIENTIFIC_PROCESSOR_MINIO_ACCESS_KEY", "k"),
            ("OTLP_TRACES_ENDPOINT", "http://ox:1"),
        ]:
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_DOWNSTREAM_LICENSE_TOKEN", "")

        with pytest.raises(RuntimeError, match="downstream_license_token"):
            Settings()

    def test_production_sandbox_enabled_requires_image(self, monkeypatch):
        from config import Settings, get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("SCIENTIFIC_PROCESSOR_SANDBOX_ENABLED", "true")
        for k, v in [
            ("SCIENTIFIC_PROCESSOR_GRAPH_ADAPTER_URL", "http://gx:1"),
            ("SCIENTIFIC_PROCESSOR_PYTORCH_BACKEND_URL", "http://px:1"),
            ("SCIENTIFIC_PROCESSOR_INFERENCE_ROUTER_URL", "http://ix:1"),
            ("SCIENTIFIC_PROCESSOR_DISCOVERY_ENGINE_URL", "http://dx:1"),
            ("SCIENTIFIC_PROCESSOR_LEAN4_PROVER_URL", "http://lx:1"),
            ("SCIENTIFIC_PROCESSOR_MINIO_ENDPOINT", "mx:1"),
            ("SCIENTIFIC_PROCESSOR_MINIO_ACCESS_KEY", "k"),
            ("OTLP_TRACES_ENDPOINT", "http://ox:1"),
        ]:
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("SCIENTIFIC_PROCESSOR_SANDBOX_IMAGE", raising=False)

        with pytest.raises(RuntimeError, match="sandbox_image"):
            Settings()

    def test_non_production_skips_validator(self, monkeypatch):
        from config import Settings, get_settings

        get_settings.cache_clear()
        self._clear_required_env(monkeypatch)
        monkeypatch.setenv("ENVIRONMENT", "test")

        s = Settings()
        assert s.environment == "test"
        assert s.graph_adapter_url == ""


# ---------------------------------------------------------------------------
# parse_auth_flag — auth-gate booleans refuse boot on an unknown spelling
# ---------------------------------------------------------------------------


class TestAuthFlags:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, True),
            ("", True),
            ("  ", True),
            ("1", True),
            (" TRUE ", True),
            ("on", True),
            ("0", False),
            ("Off", False),
        ],
    )
    def test_known_values_and_blank_default(self, raw, expected):
        from config import parse_auth_flag

        assert parse_auth_flag("X_FLAG", raw, True) is expected

    @pytest.mark.parametrize("raw", ["maybe", "flase", "enforce", "enabled", "2", "tru"])
    def test_unknown_value_raises_naming_the_flag(self, raw):
        from config import parse_auth_flag

        with pytest.raises(ValueError, match="X_FLAG"):
            parse_auth_flag("X_FLAG", raw, True)

    @pytest.mark.parametrize("raw", ["maybe", "enabled", "tru"])
    def test_unknown_enable_auth_refuses_boot(self, raw, monkeypatch):
        from config import Settings

        monkeypatch.setenv("ENABLE_AUTH", raw)
        with pytest.raises(ValueError, match="ENABLE_AUTH"):
            Settings()

    @pytest.mark.parametrize("raw", ["maybe", "enabled"])
    def test_unknown_enable_auth_refuses_import_of_auth_dependencies(self, raw):
        """``auth.dependencies`` reads ENABLE_AUTH at import; it used to be
        ``== "true"``, so ``ENABLE_AUTH=1`` served the anonymous user there while
        ``Settings`` had installed the middleware. Now it uses the same parser
        and an unknown value stops the process from importing it."""
        import os
        import pathlib
        import subprocess
        import sys

        src = pathlib.Path(__file__).resolve().parents[2] / "src"
        env = {**os.environ, "ENABLE_AUTH": raw, "PYTHONPATH": str(src)}
        proc = subprocess.run(
            [sys.executable, "-c", "import auth.dependencies"], env=env, capture_output=True, text=True, check=False
        )
        assert proc.returncode != 0
        assert "ENABLE_AUTH" in proc.stderr

    @pytest.mark.parametrize(("raw", "expected"), [("1", True), ("yes", True), ("0", False), ("false", False)])
    def test_dependencies_and_settings_agree(self, raw, expected, monkeypatch):
        import importlib

        import auth.dependencies as deps
        from config import Settings

        monkeypatch.setenv("ENABLE_AUTH", raw)
        try:
            assert importlib.reload(deps).ENABLE_AUTH is expected
            assert Settings().enable_auth is expected
        finally:
            monkeypatch.delenv("ENABLE_AUTH", raising=False)
            importlib.reload(deps)

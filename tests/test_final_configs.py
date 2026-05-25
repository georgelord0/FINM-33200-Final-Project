import yaml

from bench.report_notebook import BENCHMARKS, configured_result_models, load_config


def test_default_benchmarks_configure_chronos_and_timesfm():
    for spec in BENCHMARKS.values():
        cfg = load_config(spec)
        models = configured_result_models(cfg)
        assert any(m.startswith("chronos-bolt") for m in models), spec.key
        assert any(m.startswith("chronos-2") for m in models), spec.key
        assert any(m.startswith("timesfm-2.5") for m in models), spec.key


def test_intraday_latency_config_includes_timesfm():
    cfg = yaml.safe_load(BENCHMARKS["intraday_1m_wrds"].config_path.read_text())
    models = configured_result_models(cfg)
    assert cfg["latency"] is True
    assert any(m.startswith("timesfm-2.5") for m in models)

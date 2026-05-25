from bench.run import build_models


def test_tsfm_model_names_build_without_loading_weights():
    models = build_models([
        "chronos-bolt-252-cpu",
        "chronos-2-252-cpu",
        "timesfm-2.5-252",
        "timesfm-2.5-xreg-252",
        "moirai-1.1-small-252-cpu",
    ])
    assert [m.name for m in models] == [
        "chronos-bolt-252",
        "chronos-2-252",
        "timesfm-2.5-252",
        "timesfm-2.5-xreg-252",
        "moirai-1.1-small-252",
    ]

"""Lightweight checks to prevent documentation/config regressions.

Verifies that config files referenced in documentation actually exist and
that versioned regression blocks keep their required keys and coherent
reference counts.
"""

from __future__ import annotations

import pathlib

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _load_yaml(name: str) -> dict:
    path = CONFIGS_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestConfigPathsExist:
    def test_data_rules_yaml_exists(self):
        assert (CONFIGS_DIR / "data_rules.yaml").exists()

    def test_recommender_features_yaml_exists(self):
        assert (CONFIGS_DIR / "recommender_features.yaml").exists()

    def test_classifier_features_yaml_exists(self):
        assert (CONFIGS_DIR / "classifier_features.yaml").exists()

    def test_model_parameters_yaml_exists(self):
        assert (CONFIGS_DIR / "model_parameters.yaml").exists()

    def test_presets_yaml_exists(self):
        assert (CONFIGS_DIR / "presets.yaml").exists()

    def test_app_yaml_exists(self):
        assert (CONFIGS_DIR / "app.yaml").exists()


class TestRegressionConsistency:
    def test_identity_matches_regression_block(self):
        config = _load_yaml("data_rules.yaml")
        regression = config.get("regression", {})
        identity = config.get("identity", {})
        assert (
            regression["exact_recording_groups_after_quarantine"]
            == identity["expected_exact_recording_groups"]
        )

    def test_regression_block_has_required_keys(self):
        config = _load_yaml("data_rules.yaml")
        regression = config.get("regression", {})
        required = [
            "raw_dataset_sha256",
            "raw_rows",
            "quarantined_identity_rows",
            "valid_track_ids",
            "exact_recording_groups_after_quarantine",
        ]
        for key in required:
            assert key in regression, f"Missing regression key: {key}"

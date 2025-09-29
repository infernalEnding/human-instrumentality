from __future__ import annotations

import argparse

import pytest
from persona.models import (
    DEFAULT_PROFILE_NAME,
    ModelProfile,
    apply_profile_defaults,
    describe_model_profiles,
    get_model_profile,
    list_model_profiles,
)


def test_list_model_profiles_contains_default() -> None:
    profiles = list_model_profiles()
    assert DEFAULT_PROFILE_NAME in profiles


def test_get_model_profile_unknown() -> None:
    with pytest.raises(ValueError):
        get_model_profile("does-not-exist")


def test_apply_profile_defaults_populates_missing_fields() -> None:
    profile = get_model_profile(DEFAULT_PROFILE_NAME)
    namespace = argparse.Namespace()
    apply_profile_defaults(
        namespace,
        profile,
        (
            "asr_model",
            "llm_model",
            "tts_model",
            "tts_vocoder",
            "tts_speaker_dataset",
            "tts_speaker_sample",
        ),
    )
    assert namespace.asr_model == profile.asr_model
    assert namespace.llm_model == profile.llm_model
    assert namespace.tts_model == profile.tts_model
    assert namespace.tts_vocoder == profile.tts_vocoder
    assert namespace.tts_speaker_dataset == profile.tts_speaker_dataset
    assert namespace.tts_speaker_sample == profile.tts_speaker_sample


def test_apply_profile_defaults_respects_existing_values() -> None:
    profile = ModelProfile(
        name="custom",
        description="",
        asr_model="a",
        llm_model="b",
        tts_model="c",
    )
    namespace = argparse.Namespace(asr_model="override", tts_model="existing")
    apply_profile_defaults(namespace, profile, ("asr_model", "llm_model", "tts_model"))
    assert namespace.asr_model == "override"
    assert namespace.llm_model == "b"
    assert namespace.tts_model == "existing"


def test_describe_model_profiles_includes_descriptions() -> None:
    lines = describe_model_profiles()
    assert any("ASR:" in line for line in lines)


def test_cli_profile_argument_populates_defaults() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--list-model-profiles", action="store_true")
    parser.add_argument("--asr-model", default=argparse.SUPPRESS)
    parser.add_argument("--llm-model", default=argparse.SUPPRESS)
    parser.add_argument("--tts-model", default=argparse.SUPPRESS)
    args = parser.parse_args([])
    profile = get_model_profile(args.profile)
    apply_profile_defaults(args, profile, ("asr_model", "llm_model", "tts_model"))
    assert args.asr_model == profile.asr_model
    assert args.llm_model == profile.llm_model
    assert args.tts_model == profile.tts_model


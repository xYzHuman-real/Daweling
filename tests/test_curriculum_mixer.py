from training.curriculum import CurriculumExample, CurriculumStage, CurriculumScheduler
from training.curriculum_mixer import CurriculumMixer


def test_mixer_is_deterministic_and_filters_invalid_examples():
    examples = [
        CurriculumExample("language", CurriculumStage.LANGUAGE),
        CurriculumExample("reasoning", CurriculumStage.REASONING),
        CurriculumExample("empty", CurriculumStage.REASONING, 0),
        CurriculumExample("   ", CurriculumStage.LANGUAGE),
    ]
    mixer = CurriculumMixer(CurriculumScheduler(warmup_epochs=0, stage_epochs=1))
    first = mixer.mix(examples, 2)
    second = mixer.mix(examples, 2)
    assert first == second
    assert {item.text for item in first.examples} == {"language", "reasoning"}
    assert first.total_weight == 2.0


def test_weights_by_stage_is_aggregated():
    examples = [
        CurriculumExample("a", CurriculumStage.LANGUAGE, 2.0),
        CurriculumExample("b", CurriculumStage.INSTRUCTION, 3.0),
    ]
    mix = CurriculumMixer(CurriculumScheduler(warmup_epochs=0, stage_epochs=1)).mix(examples, 1)
    assert CurriculumMixer.weights_by_stage(mix) == {"language": 2.0, "instruction": 3.0}


def test_empty_text_mixture_fails_closed():
    mixer = CurriculumMixer(CurriculumScheduler(warmup_epochs=0))
    try:
        mixer.mix([CurriculumExample("  ", CurriculumStage.LANGUAGE)], 0)
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("expected empty curriculum mixture to fail")

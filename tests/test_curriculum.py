from training.curriculum import CurriculumExample, CurriculumScheduler, CurriculumStage


def test_curriculum_progresses_deterministically():
    scheduler = CurriculumScheduler(warmup_epochs=1, stage_epochs=2)
    assert scheduler.stage_for_epoch(0) is CurriculumStage.LANGUAGE
    assert scheduler.stage_for_epoch(1) is CurriculumStage.INSTRUCTION
    assert scheduler.stage_for_epoch(2) is CurriculumStage.INSTRUCTION
    assert scheduler.stage_for_epoch(3) is CurriculumStage.REASONING
    assert scheduler.stage_for_epoch(99) is CurriculumStage.END_TO_END


def test_batch_contains_only_reached_stages():
    examples = [
        CurriculumExample("language", CurriculumStage.LANGUAGE),
        CurriculumExample("reason", CurriculumStage.REASONING),
        CurriculumExample("e2e", CurriculumStage.END_TO_END),
    ]
    batch = CurriculumScheduler(warmup_epochs=0).batch(examples, epoch=1)
    assert batch.stage is CurriculumStage.INSTRUCTION
    assert [item.text for item in batch.examples] == ["language"]


def test_invalid_epoch_and_empty_batch_fail_closed():
    scheduler = CurriculumScheduler()
    try:
        scheduler.stage_for_epoch(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative epoch should fail")

    try:
        scheduler.batch([CurriculumExample("e2e", CurriculumStage.END_TO_END)], epoch=0)
    except ValueError:
        pass
    else:
        raise AssertionError("unavailable curriculum stage should fail")

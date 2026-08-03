from architecture.toxacute_tasks import (
    ANIMAL_SOURCE_TASKS,
    HUMAN_TARGET_TASKS,
    TOXACUTE_TASKS,
    parse_toxacute_task_name,
)


def test_toxacute_registry_has_59_unique_endpoints():
    assert len(TOXACUTE_TASKS) == 59
    assert len(set(TOXACUTE_TASKS)) == 59
    assert len(ANIMAL_SOURCE_TASKS) == 56
    assert len(HUMAN_TARGET_TASKS) == 3


def test_toxacute_metadata_parsing():
    women = parse_toxacute_task_name("women_oral_TDLo")
    guinea = parse_toxacute_task_name("guinea pig_intravenous_LD50")
    assert (women.organism, women.population, women.route, women.measurement) == (
        "human", "women", "oral", "TDLo"
    )
    assert guinea.organism == "guinea pig"

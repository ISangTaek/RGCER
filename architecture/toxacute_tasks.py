"""Canonical ToxAcute endpoint registry and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


TOXACUTE_TASKS: List[str] = [
    "mouse_intraperitoneal_LD50",
    "mouse_intraperitoneal_LDLo",
    "mouse_intravenous_LD50",
    "mouse_intravenous_LDLo",
    "mouse_oral_LD50",
    "mouse_oral_LDLo",
    "mouse_unreported_LD50",
    "mouse_skin_LD50",
    "mouse_subcutaneous_LD50",
    "mouse_subcutaneous_LDLo",
    "mouse_intramuscular_LD50",
    "mouse_parenteral_LD50",
    "rat_intraperitoneal_LD50",
    "rat_intraperitoneal_LDLo",
    "rat_intravenous_LD50",
    "rat_intravenous_LDLo",
    "rat_oral_LD50",
    "rat_oral_LDLo",
    "rat_unreported_LD50",
    "rat_skin_LD50",
    "rat_subcutaneous_LD50",
    "rat_subcutaneous_LDLo",
    "rat_intramuscular_LD50",
    "mammal (species unspecified)_intraperitoneal_LD50",
    "mammal (species unspecified)_oral_LD50",
    "mammal (species unspecified)_unreported_LD50",
    "mammal (species unspecified)_subcutaneous_LD50",
    "guinea pig_intraperitoneal_LD50",
    "guinea pig_intravenous_LD50",
    "guinea pig_intravenous_LDLo",
    "guinea pig_oral_LD50",
    "guinea pig_skin_LD50",
    "guinea pig_subcutaneous_LD50",
    "guinea pig_subcutaneous_LDLo",
    "rabbit_intraperitoneal_LD50",
    "rabbit_intravenous_LD50",
    "rabbit_intravenous_LDLo",
    "rabbit_oral_LD50",
    "rabbit_oral_LDLo",
    "rabbit_skin_LD50",
    "rabbit_skin_LDLo",
    "rabbit_subcutaneous_LD50",
    "rabbit_subcutaneous_LDLo",
    "dog_intravenous_LD50",
    "dog_intravenous_LDLo",
    "dog_oral_LD50",
    "dog_oral_LDLo",
    "cat_intravenous_LD50",
    "cat_intravenous_LDLo",
    "cat_oral_LD50",
    "cat_oral_LDLo",
    "bird-wild_oral_LD50",
    "quail_oral_LD50",
    "duck_oral_LD50",
    "chicken_oral_LD50",
    "frog_subcutaneous_LDLo",
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
]

HUMAN_TARGET_TASKS: List[str] = [
    "man_oral_TDLo",
    "women_oral_TDLo",
    "human_oral_TDLo",
]

ANIMAL_SOURCE_TASKS: List[str] = [
    task for task in TOXACUTE_TASKS if task not in HUMAN_TARGET_TASKS
]


@dataclass(frozen=True)
class TaskMetadata:
    task_name: str
    organism: str
    route: str
    measurement: str
    population: str


def parse_toxacute_task_name(task_name: str) -> TaskMetadata:
    """Parse ``<organism>_<route>_<measurement>`` endpoint names.

    ``rsplit`` is intentional: organisms such as ``guinea pig`` and
    ``mammal (species unspecified)`` may contain spaces but not the final
    route/measurement separators.
    """

    if not isinstance(task_name, str) or not task_name:
        raise ValueError(f"Invalid ToxAcute task name: {task_name!r}")
    parts = task_name.rsplit("_", maxsplit=2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Invalid ToxAcute task name: {task_name!r}. "
            "Expected '<organism_or_population>_<route>_<measurement>'."
        )

    subject, route, measurement = parts
    if subject == "man":
        organism, population = "human", "man"
    elif subject == "women":
        organism, population = "human", "women"
    elif subject == "human":
        organism, population = "human", "general"
    else:
        organism, population = subject, "general"

    return TaskMetadata(
        task_name=task_name,
        organism=organism,
        route=route,
        measurement=measurement,
        population=population,
    )


def build_task_metadata(task_names: Iterable[str]) -> List[TaskMetadata]:
    return [parse_toxacute_task_name(task_name) for task_name in task_names]


def metadata_as_dict(task_names: Iterable[str]) -> Dict[str, TaskMetadata]:
    metadata = build_task_metadata(task_names)
    return {item.task_name: item for item in metadata}


__all__ = [
    "ANIMAL_SOURCE_TASKS",
    "HUMAN_TARGET_TASKS",
    "TOXACUTE_TASKS",
    "TaskMetadata",
    "build_task_metadata",
    "metadata_as_dict",
    "parse_toxacute_task_name",
]

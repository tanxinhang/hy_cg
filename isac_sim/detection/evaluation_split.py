"""Immutable train/calibration/test partitions for scientific evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

SplitRole = Literal["train", "calibration", "test"]


@dataclass(frozen=True)
class SplitDataset:
    """Records carrying one enforced role and unique scene identifiers."""

    role: SplitRole
    rows: tuple[Mapping, ...]

    def __post_init__(self) -> None:
        if self.role not in ("train", "calibration", "test"):
            raise ValueError(f"unknown evaluation split {self.role!r}")
        scene_ids = []
        for row in self.rows:
            if row.get("split") != self.role:
                raise ValueError(
                    f"{self.role} dataset contains {row.get('split')!r} record"
                )
            if "scene_id" not in row:
                raise ValueError("every split record requires scene_id")
            scene_ids.append(int(row["scene_id"]))
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError(f"duplicate scene_id inside {self.role} split")

    @property
    def scene_ids(self) -> frozenset[int]:
        return frozenset(int(row["scene_id"]) for row in self.rows)

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class EvaluationPartition:
    """Three mutually exclusive scene-level datasets."""

    train: SplitDataset
    calibration: SplitDataset
    test: SplitDataset

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping], *, require_train: bool = False
    ) -> "EvaluationPartition":
        buckets = {role: [] for role in ("train", "calibration", "test")}
        for row in records:
            role = row.get("split")
            if role not in buckets:
                raise ValueError(f"record has invalid split {role!r}")
            buckets[role].append(row)
        result = cls(*(
            SplitDataset(role, tuple(buckets[role]))
            for role in ("train", "calibration", "test")
        ))
        if require_train and not result.train.rows:
            raise ValueError("protocol requires independent training scenes")
        if not result.calibration.rows or not result.test.rows:
            raise ValueError("calibration and test splits must both be non-empty")
        result.assert_disjoint()
        return result

    def assert_disjoint(self) -> None:
        groups = (self.train, self.calibration, self.test)
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                overlap = left.scene_ids & right.scene_ids
                if overlap:
                    raise ValueError(
                        f"scene leakage between {left.role} and {right.role}: "
                        f"{sorted(overlap)}"
                    )


__all__ = ["EvaluationPartition", "SplitDataset", "SplitRole"]

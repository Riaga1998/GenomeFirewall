"""
The deterministic target gate
==============================
Runs BEFORE the ML model. Encodes hard biological rules:

  * If a required molecular target is ABSENT  -> intrinsic resistance -> FAIL.
  * If a species-intrinsic resistance marker is PRESENT -> FAIL.
  * Otherwise the gate is "open": the ML model is allowed to speak, and a
    "likely to work" verdict is only reachable through an open gate.

This is what stops the system from reporting "likely to work" merely because
no acquired resistance gene was found.
"""
from __future__ import annotations
from dataclasses import dataclass
import yaml


@dataclass
class GateResult:
    forced_label: str | None   # "R" (fail) if the gate forces resistance, else None
    reason: str                # human-readable explanation
    gate_open: bool            # True if ML is permitted to conclude susceptible


def load_gate(path: str = "config/target_gate.yaml") -> dict:
    return yaml.safe_load(open(path))


def apply_gate(antibiotic: str, present_features: set[str], gate: dict) -> GateResult:
    """present_features: set of detected determinant columns for this genome,
    e.g. {"GENE:blaKPC-2", "MUT:gyrA_S83L"} plus any target-presence flags you add.
    """
    rules = gate.get(antibiotic)
    if rules is None:
        return GateResult(None, f"No gate rule for {antibiotic}; ML unrestricted.", True)

    # 1) Required target absent -> intrinsic resistance.
    for target in rules.get("target_required") or []:
        if target not in present_features:
            return GateResult(
                "R",
                f"Molecular target '{target}' not detected — {antibiotic} has no "
                f"target to act on (intrinsic resistance).",
                False,
            )

    # 2) Intrinsic resistance marker present -> fail.
    for marker in rules.get("intrinsic_fail_if_present") or []:
        if marker in present_features:
            return GateResult(
                "R",
                f"Intrinsic resistance determinant '{marker}' detected for "
                f"{antibiotic}.",
                False,
            )

    return GateResult(None, "Target present; ML prediction permitted.", True)

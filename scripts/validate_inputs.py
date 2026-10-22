"""Validate configurations and raw inputs without modifying them."""

import json
from siniestros_riesgo.config import TABLES, setup_logging
from siniestros_riesgo.integrity import validate_inputs

logger = setup_logging("validate_inputs")
result = validate_inputs()
TABLES.mkdir(parents=True, exist_ok=True)
(TABLES / "input_validation.json").write_text(json.dumps({"ok": result.ok, "blockers": result.blockers, "warnings": result.warnings}, indent=2, ensure_ascii=False), encoding="utf-8")
for warning in result.warnings:
    logger.warning(warning)
for blocker in result.blockers:
    logger.error(blocker)
if not result.ok:
    raise SystemExit(2)

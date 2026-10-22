"""Generate the Spanish project PDF from available reproducible outputs."""

from siniestros_riesgo.config import setup_logging
from siniestros_riesgo.report_cdmx import build_cdmx_report

logger = setup_logging("build_report")
logger.info("PDF CDMX escrito en %s", build_cdmx_report())

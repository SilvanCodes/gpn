from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForMaskedLM

from gpn.pipeline.gpn_pipeline import GPNPipeline

PIPELINE_REGISTRY.register_pipeline(
    "gpn",
    pipeline_class=GPNPipeline,
    pt_model=AutoModelForMaskedLM,
    default={"pt": ("songlab/gpn-brassicales", "main")},
    type="text",
)

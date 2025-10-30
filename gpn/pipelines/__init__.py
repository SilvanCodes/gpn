from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForMaskedLM

from gpn.pipelines.simple_gpn_pipeline import SimpleGPNPipeline
from gpn.pipelines.sliding_window_gpn_pipeline import SlidingWindowGPNPipeline

DEFAULT_GPN_MODEL = ("songlab/gpn-brassicales", "main")

PIPELINE_REGISTRY.register_pipeline(
    "gpn-fast",
    pipeline_class=SimpleGPNPipeline,
    pt_model=AutoModelForMaskedLM,
    default={"pt": DEFAULT_GPN_MODEL},
    type="text",
)

PIPELINE_REGISTRY.register_pipeline(
    "gpn",
    pipeline_class=SlidingWindowGPNPipeline,
    pt_model=AutoModelForMaskedLM,
    default={"pt": DEFAULT_GPN_MODEL},
    type="text",
)

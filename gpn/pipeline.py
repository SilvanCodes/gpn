import torch
from transformers import Pipeline


class GPNPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, seq):
        tokens = self.tokenizer(
            seq,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {"tokens": tokens, "seq": seq}

    def _forward(self, model_inputs):
        output = self.model(input_ids=model_inputs["tokens"]["input_ids"])
        return {"output": output, "seq": model_inputs["seq"]}

    def postprocess(self, model_outputs):
        acgt_idxs = [self.tokenizer.get_vocab()[nuc] for nuc in list("acgt")]
        nucleotide_logits = model_outputs["output"].logits[:, :, acgt_idxs]
        output_probs = torch.nn.functional.softmax(nucleotide_logits, dim=-1)

        print(model_outputs["seq"])

        return output_probs
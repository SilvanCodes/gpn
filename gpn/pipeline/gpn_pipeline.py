import torch
from transformers import Pipeline


class GPNPipeline(Pipeline):
    def preprocess(self, seq):
        return self.tokenizer(
            seq,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    def _forward(self, model_inputs):
        return self.model(input_ids=model_inputs["input_ids"])

    def postprocess(self, model_outputs):
        acgt_idxs = [self.tokenizer.get_vocab()[nuc] for nuc in list("acgt")]
        nucleotide_logits = model_outputs.logits[:, :, acgt_idxs]
        output_probs = torch.nn.functional.softmax(nucleotide_logits, dim=-1)
        return output_probs

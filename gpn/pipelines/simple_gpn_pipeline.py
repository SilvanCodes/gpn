import torch
import pandas as pd
import numpy as np
from transformers import Pipeline


class SimpleGPNPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, sequence):
        tokens = self.tokenizer(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {"tokens": tokens, "seq": sequence}

    def _forward(self, model_inputs):
        output = self.model(**model_inputs["tokens"])
        return {"output": output, "seq": model_inputs["seq"]}

    def postprocess(self, model_outputs):
        seq = model_outputs["seq"]
        logits = model_outputs["output"].logits
        vocab = self.tokenizer.get_vocab()
        acgt = np.array(list("acgt"))

        # get logits only for A/C/G/T and convert to probabilities
        acgt_idxs = [vocab[n] for n in acgt]
        probs = torch.softmax(logits[:, :, acgt_idxs], dim=-1).squeeze().cpu().numpy()

        df = pd.DataFrame(probs, columns=[f"p_{n}" for n in acgt])
        df["ref"] = list(seq)

        # Vectorized selection of reference probabilities
        ref_idx = pd.Categorical(df["ref"].str.lower(), categories=acgt).codes
        df["p_ref"] = probs[np.arange(len(df)), ref_idx]

        # Compute GPN log2 ratios in one go (broadcasting)
        gpn = np.log2(probs / df["p_ref"].to_numpy()[:, None])
        df[[f"gpn_{n}" for n in acgt]] = gpn

        cols = ["ref", "p_ref"] + [f"p_{n}" for n in acgt] + [f"gpn_{n}" for n in acgt]
        return df[cols]

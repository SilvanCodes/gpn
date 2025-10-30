import torch
import pandas as pd
import numpy as np
from transformers.pipelines.base import ChunkPipeline


class SlidingWindowGPNPipeline(ChunkPipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "window_size" in kwargs:
            preprocess_params["window_size"] = int(kwargs["window_size"])
        if "use_mask" in kwargs:
            preprocess_params["use_mask"] = bool(kwargs["use_mask"])
        return preprocess_params, {}, {}

    def preprocess(self, sequence, window_size=513, use_mask="True"):
        """
        Slides a masking window across the sequence and yields tokenized inputs
        with the center position masked.
        """
        tokens = list(sequence)
        seq_len = len(tokens)
        half = window_size // 2

        pad_tok = self.tokenizer.pad_token
        mask_tok = self.tokenizer.mask_token

        padded = [pad_tok] * half + tokens + [pad_tok] * half

        for pos, ref in enumerate(tokens):
            # Create a masked window
            window_tokens = padded[pos : pos + window_size]
            window_tokens = list(window_tokens)
            if use_mask:
                window_tokens[half] = mask_tok

            model_input = self.tokenizer(
                window_tokens,
                is_split_into_words=True,
                return_tensors="pt",
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            yield {
                **model_input,
                "reference": ref.lower() if isinstance(ref, str) else ref,
                "is_last": pos == seq_len - 1,
            }

    def _forward(self, model_inputs):
        reference = model_inputs.pop("reference")
        is_last = model_inputs.pop("is_last")

        output = self.model(**model_inputs)
        return {"output": output, "reference": reference, "is_last": is_last}

    def postprocess(self, all_model_outputs):
        acgt = np.array(list("acgt"))
        vocab = self.tokenizer.get_vocab()
        acgt_idxs = [vocab[n] for n in acgt]

        # Collect per-position results
        refs, probs = [], []
        for out in all_model_outputs:
            logits = out["output"].logits[:, :, acgt_idxs]
            p = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            center_p = p[len(p) // 2]  # only the central masked position
            refs.append(out["reference"])
            probs.append(center_p)

        probs = np.vstack(probs)
        df = pd.DataFrame(probs, columns=[f"p_{n}" for n in acgt])
        df["ref"] = refs

        # Vectorized p_ref and log-ratio computation
        ref_idx = pd.Categorical(df["ref"], categories=acgt).codes
        df["p_ref"] = probs[np.arange(len(df)), ref_idx]
        gpn = np.log2(probs / df["p_ref"].to_numpy()[:, None])
        df[[f"gpn_{n}" for n in acgt]] = gpn

        cols = ["ref", "p_ref"] + [f"p_{n}" for n in acgt] + [f"gpn_{n}" for n in acgt]
        return df[cols]

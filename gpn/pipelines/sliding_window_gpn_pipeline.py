import torch
import pandas as pd
import numpy as np
from transformers.pipelines.base import ChunkPipeline
from tqdm.auto import tqdm


class SlidingWindowGPNPipeline(ChunkPipeline):
    """
    Input: A DNA sequence as a string

    Allowed keyword arguments:
    - start (relative to sequence length, zero indexed, default 0)
    - end (relative to sequence length, end-exclusive, default sequence length)
    - window_size (sequence context centered on predicted position, default 513)
    - use_mask (mask the predicted position, default True)
    - batch_size (how many windows predicted in parallel, default 1)

    Output: A dataframe with the following columns:
    - ref (the reference nucleotide)
    - p_ref (the probability assigned to the reference nucleotide)
    - p_{a,c,g,t} (the probability distribution of the predicted position)
    - gpn_{a,c,g,t} (the GPN score with respect to each alternative)
    """

    @staticmethod
    def verify_range(start, end, seq_len):
        """
        Normalize and validate [start, end) for a sequence of length seq_len.

        Returns
        -------
        (start, end) : tuple[int, int]
            Validated and clamped indices.

        Raises
        ------
        ValueError
            If the resulting range is empty.
        """
        if start is None:
            start = 0
        if end is None:
            end = seq_len

        start = int(start)
        end = int(end)

        # Clamp to valid range
        if start < 0:
            start = 0
        if end > seq_len:
            end = seq_len

        if start >= end:
            raise ValueError("Range is empty")

        return start, end

    def __call__(self, inputs, *args, **kwargs):
        # Determine region length for the progress bar
        seq_len = len(inputs)

        start = kwargs.get("start", 0)
        end = kwargs.get("end", None)

        start, end = self.verify_range(start, end, seq_len)

        total_windows = end - start

        self._progress_bar = tqdm(total=total_windows, desc="Processing windows")
        self._progress_bar_step = kwargs.get("batch_size", 1)

        result = super().__call__(inputs, *args, **kwargs)

        self._progress_bar.close()
        return result

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "window_size" in kwargs:
            preprocess_params["window_size"] = int(kwargs["window_size"])
        if "use_mask" in kwargs:
            preprocess_params["use_mask"] = bool(kwargs["use_mask"])
        if "start" in kwargs:
            preprocess_params["start"] = int(kwargs["start"])
        if "end" in kwargs:
            preprocess_params["end"] = int(kwargs["end"])
        return preprocess_params, {}, {}

    def preprocess(self, sequence, window_size=513, use_mask=True, start=0, end=None):
        """
        Slides a masking window across the sequence and yields tokenized inputs
        with the center position masked, restricted to [start, end).
        """
        tokens = list(sequence)
        seq_len = len(tokens)

        start, end = self.verify_range(start, end, seq_len)

        half = window_size // 2

        pad_tok = self.tokenizer.pad_token
        mask_tok = self.tokenizer.mask_token

        padded = [pad_tok] * half + tokens + [pad_tok] * half

        # Iterate only over the requested region
        for pos in range(start, end):
            ref = tokens[pos]

            # Create a masked window
            window_tokens = padded[pos : pos + window_size]
            window_tokens = list(window_tokens)
            if use_mask:
                window_tokens[half] = mask_tok

            model_input = self.tokenizer(
                window_tokens,
                return_tensors="pt",
                is_split_into_words=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )

            yield {
                **model_input,
                "reference": ref.lower() if isinstance(ref, str) else ref,
                # "is_last" refers to last processed position, not end of sequence
                "is_last": pos == end - 1,
            }

    def _forward(self, model_inputs):
        reference = model_inputs.pop("reference")
        is_last = model_inputs.pop("is_last")

        output = self.model(**model_inputs)
        self._progress_bar.update(self._progress_bar_step)
        # Return the logits directly for batching and further processing
        return {"logits": output.logits, "reference": reference, "is_last": is_last}

    def postprocess(self, all_model_outputs):
        acgt = np.array(list("acgt"))
        vocab = self.tokenizer.get_vocab()
        acgt_idxs = [vocab[n] for n in acgt]

        # Collect per-position results
        refs, probs = [], []
        for out in all_model_outputs:
            logits = out["logits"][:, :, acgt_idxs]
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

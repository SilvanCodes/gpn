import torch
import numpy as np
from transformers.pipelines.base import ChunkPipeline
from tqdm.auto import tqdm

from gpn.pipelines.utils import (
    get_acgt_tokens_and_indices,
    normalize_sequence_to_vocab_case,
)


class NDMPipeline(ChunkPipeline):
    def __call__(self, inputs, *args, **kwargs):
        if isinstance(inputs, str):
            seq_len = len(inputs)
            total_windows = seq_len * 3 + 1
        else:
            # if you want multi-sequence support later, you'll need to sum per-sequence
            raise ValueError(
                "NDMPipeline currently expects a single sequence string as input."
            )

        self._progress_bar = tqdm(total=total_windows, desc="Processing permutations")
        self._progress_bar_step = kwargs.get("batch_size", 1)

        result = super().__call__(inputs, *args, **kwargs)

        self._progress_bar.close()
        return result

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, sequence):
        sequence = normalize_sequence_to_vocab_case(sequence, self.tokenizer)
        _, vocab_tokens, _ = get_acgt_tokens_and_indices(self.tokenizer)

        buf = bytearray(sequence, "ascii")
        L = len(buf)

        for pos in range(L):
            ref = chr(buf[pos])
            original_byte = buf[pos]

            for idx, alt in enumerate(vocab_tokens):
                alt = str(alt)
                if alt.lower() == ref.lower():
                    continue

                # mutate in-place
                buf[pos] = ord(alt)
                model_input = self.tokenizer(
                    buf.decode("ascii"),
                    return_tensors="pt",
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )

                yield {
                    **model_input,
                    "mutated_position": pos,
                    "mutated_nucleotide_index": idx,
                    "is_last": False,
                }

            # restore original base before moving on
            buf[pos] = original_byte

        model_input = self.tokenizer(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        yield {
            **model_input,
            "mutated_position": -1,
            "mutated_nucleotide_index": -1,
            "is_last": True,
        }

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        mutated_position = model_inputs.pop("mutated_position")
        mutated_nucleotide_index = model_inputs.pop("mutated_nucleotide_index")

        output = self.model(**model_inputs)
        self._progress_bar.update(self._progress_bar_step)
        # Return the logits directly for batching and further processing
        return {
            "logits": output.logits,
            "mutated_position": mutated_position,
            "mutated_nucleotide_index": mutated_nucleotide_index,
            "is_last": is_last,
        }

    def postprocess(self, all_model_outputs, epsilon=1e-10):
        _, _, acgt_idxs = get_acgt_tokens_and_indices(self.tokenizer)

        # Collect per-mutation results
        meta, probs = [], []
        for out in all_model_outputs:
            logits = out["logits"][:, :, acgt_idxs]
            p = torch.softmax(logits, dim=-1)
            probs.append(p)
            meta.append(
                (
                    out["mutated_position"],
                    out["mutated_nucleotide_index"],
                )
            )

        meta = np.array(meta[:-1], dtype=np.int64)

        # rebuild to (seq_len * 3 + 1, seq_len, 4)
        snp_reconstruct = torch.concat(probs, axis=0).numpy()

        snp_reconstruct = (snp_reconstruct + epsilon) / snp_reconstruct.sum(
            axis=-1, keepdims=True
        )

        seq_len = snp_reconstruct.shape[1]
        snp_effect = np.zeros((seq_len, seq_len, 4, 4))

        # separate out reference probabilities
        reference_probs = snp_reconstruct[-1]
        snp_reconstruct = snp_reconstruct[:-1]

        snp_effect[
            meta[:, 0],
            :,
            meta[:, 1],
            :,
        ] = (
            np.log2(snp_reconstruct)
            - np.log2(1 - snp_reconstruct)
            - np.log2(reference_probs)
            + np.log2(1 - reference_probs)
        )

        dep_map = np.max(np.abs(snp_effect), axis=(2, 3))
        np.fill_diagonal(dep_map, 0.0)

        return dep_map

    def vis(dep_map):
        vis = dep_map.sum(axis=1) / (dep_map.shape[1] - 1)
        vis.name = "VIS"
        return vis

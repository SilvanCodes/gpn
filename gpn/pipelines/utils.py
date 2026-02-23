import numpy as np


def get_acgt_tokens_and_indices(tokenizer):
    """
    Return canonical lowercase A/C/G/T labels plus tokenizer-specific tokens/indices.
    """
    vocab = tokenizer.get_vocab()
    acgt = np.array(list("acgt"))

    if all(base in vocab for base in acgt):
        vocab_tokens = acgt
    elif all(base.upper() in vocab for base in acgt):
        vocab_tokens = np.char.upper(acgt)
    else:
        raise KeyError(
            "Tokenizer vocabulary must contain either all lowercase or all uppercase A/C/G/T tokens."
        )

    acgt_idxs = [vocab[token] for token in vocab_tokens]
    return acgt, vocab_tokens, acgt_idxs


def normalize_sequence_to_vocab_case(sequence, tokenizer):
    _, vocab_tokens, _ = get_acgt_tokens_and_indices(tokenizer)
    if str(vocab_tokens[0]).islower():
        return sequence.lower()
    return sequence.upper()

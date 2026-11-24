# Feasibility: vocabulary-byte prediction for lossless prefixes

**Decision: do not launch a new model study or develop this as a claimed research advance.** The transform is portable and lossless, but established compression methods are strong alternatives and the current real-message artifacts do not support the proposed vocabulary-anchored comparison. At most, consider a small screen using already retained arrays after the frozen alignment study finishes.

This is a read-only feasibility assessment of the pinned repository snapshot below. No compression benchmark, model inference, numerical study, or outcome measurement was performed for this note.

## What would be exact

Suppose both endpoints already hold identical vocabulary row bytes and the same ordered token IDs. Concatenate the selected rows into predictor bytes P, using exactly the target dtype, byte order, shape and layout. For original final-prefix bytes X, encode a losslessly compressed residual:

    E = compress(X XOR P)
    X = decompress(E) XOR P

Decoding needs byte lookup, decompression and XOR; it does not reproduce an eigensolver, floating-point normalization or matrix multiplication. Signed zeros and other floating-point bit patterns can be preserved as opaque bytes.

The protocol must bind the vocabulary fingerprint, ID sequence, representation and output length. If IDs or the vocabulary are not already shared, charge their transmission and storage. A dtype-converted vocabulary must be materialized and fingerprinted as exact bytes; independently recomputing floating-point predictor values would reintroduce the portability problem.

Selecting the smaller **complete framed** raw or compressed packet prevents expansion relative to that same framed raw baseline. It does not guarantee savings over ordinary compression or compensate for extra encoding work. XOR with a known predictor is reversible; its benefit depends on exposing redundancy to the compressor, not reducing the underlying information requirement.

## Closest primary precedents and required baselines

- [FPC](https://userweb.cs.txstate.edu/~burtscher/papers/tc09.pdf), section 2, already combines value prediction, reversible XOR residuals and compact encoding. [Gorilla](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf), section 4.1.2, uses the previous value as a cheap floating-point bit predictor. [Chimp](https://vldb.org/pvldb/vol15/p3058-liakos.pdf) also considers earlier reference values.
- [Bitshuffle's author implementation](https://github.com/kiyo-masui/bitshuffle) and [original paper](https://arxiv.org/abs/1503.00638) establish reversible bit-plane rearrangement followed by ordinary compression. Its lossless transform must be distinguished from the paper's separate lossy rounding stage.
- The [TREBE conference abstract](https://epapers2.org/aicas2025/ESR/paper_details.php?paper_id=6060) describes lossless KV-cache compression using token redundancy, XOR and bit-plane encoding.
- [Lossless Compression of Neural Network Components](https://arxiv.org/html/2508.19263v1) separates exponent/mantissa streams for neural tensors and KV caches, and uses XOR deltas for checkpoints.
- [VCDIFF, RFC 3284](https://www.rfc-editor.org/rfc/rfc3284), provides longstanding precedent for portable compression against shared reference bytes.

A useful comparison must include raw framing; ordinary Zstandard compression; byte-shuffle and bit-shuffle plus the same compressor; and previous-word/previous-row XOR followed by the same shuffle/compressor. A shared-reference dictionary compressor should receive the same vocabulary-derived reference bytes as the proposed method. Native-precision specialized float codecs are additional comparators where supported.

Use complete per-message packets, including codec framing and selection metadata. Compressing a whole NPZ or batching many related examples answers a different storage question. Any portfolio that tries several encodings must charge all attempts. Float closeness alone does not establish compressible mantissa bits.

## What Research3 actually retains

The inspected repository snapshot was [e7c467f2b097666dc474a3dbb6e273bf253e785f](https://github.com/ConfusedDuckCollectibles/Research3/tree/e7c467f2b097666dc474a3dbb6e273bf253e785f).

**Readout study:** [message_features.npz](https://github.com/ConfusedDuckCollectibles/Research3/blob/e7c467f2b097666dc474a3dbb6e273bf253e785f/results/readout_study/exports/message_features.npz) contains actual 16-coordinate float32 messages; [feature_manifest.json](https://github.com/ConfusedDuckCollectibles/Research3/blob/e7c467f2b097666dc474a3dbb6e273bf253e785f/results/readout_study/feature_manifest.json) records their shapes, order and hashes. [bridge_weights.npz](https://github.com/ConfusedDuckCollectibles/Research3/blob/e7c467f2b097666dc474a3dbb6e273bf253e785f/results/readout_study/exports/bridge_weights.npz) retains the decoder weights.

The [decoder source](https://github.com/ConfusedDuckCollectibles/Research3/blob/9eb4478fae1ad373234eb2a725345b66c8c63c57/latent_bridge/query_conditioned.py) is one linear layer. The [export source](https://github.com/ConfusedDuckCollectibles/Research3/blob/9eb4478fae1ad373234eb2a725345b66c8c63c57/latent_bridge/readout_study.py) saves its complete state. Thus `message.reshape(N, 2, 8) @ expander.weight.T + expander.bias` produces new expanded-prefix arrays without pretrained-model inference.

However, original expanded-prefix bytes/hashes were not retained, so reconstruction on another arithmetic backend cannot establish historical bit identity. Neither matching receiver vocabulary bytes nor one-token-per-slot reference IDs are exported. The learned slots are not token-aligned StateBridge rows.

**The decisive baseline is already 64 bytes per message plus a shared linear decoder.** Compressing its expanded prefix mainly attempts to rediscover this existing factorization. It cannot repair the measured task failures.

**Binding study:** retained parameters and prediction records do not include per-example message or final-prefix tensors; see its [results and retention description](https://github.com/ConfusedDuckCollectibles/Research3/blob/e7c467f2b097666dc474a3dbb6e273bf253e785f/docs/binding_results.md).

**Planned alignment exports:** the [frozen design](alignment_numerical_design.md) retains synthetic source/reference arrays and pre-calibration outputs. The reviewed runner also retains synthetic vocabularies and IDs. These permit a later model-free codec screen, including derived final prefixes, but provide no representative real-model prefix corpus. Collapse fixtures are favorable constructed controls, not evidence of general compression.

## Reopening condition

Reopen only with representative, exactly retained final-prefix bytes and their legitimate shared reference information, or a clear incremental win over the strongest ordinary codecs on an explicitly limited saved-array screen. Stop if gains depend on synthetic collapse, uncharged dictionaries, large batches, or comparing expanded prefixes only against raw expanded prefixes. The present evidence does not justify another model run.

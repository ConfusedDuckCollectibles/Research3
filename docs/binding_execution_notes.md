# Binding study execution record

## Frozen scope and implementation

The scientific design was published at `dfce06b5dfb2860414af166ef4441eefd9548afc`.
The reviewed implementation and a separate offline validation workflow were
published at `cc843cee45bce8525de7603c1bddc7a6915344ba`.
The empirical study workflow was deliberately absent from that commit, so
neither its push nor its validation failures could start pretrained fitting.

The frozen data, model pair, six arms, three seeds, auxiliary loss, training
updates, checkpoint selection, competence gate and advancement criteria remain
unchanged.

## First offline validation

[Validation run 45353866979](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45353866979),
job 138306705157, executed the pinned runtime and **88 tests: 87 passed and one
errored**. The ordinary repository test workflow
[45353866781](https://github.com/ConfusedDuckCollectibles/Research3/actions/runs/45353866781)
reported the same error.

The only error was the synthetic double-precision GPT-2 test comparing selected
logits and their gradients with the complete output head. Transformers 4.48.2's
SDPA mask helper supplied a float64 minimum to a tensor factory using the
default float32 dtype, causing an overflow before the equivalence assertions.
The correction scopes the default dtype to float64 inside that one test and
restores the original dtype in a `finally` block. SDPA and the original
`1e-10` value/gradient tolerances are retained. The empirical study continues
to use float32; no model, dataset, loss or selection rule was changed.

The successful tests included the complete independent data fixtures, all
40,320 valid rank codes, all 25,216 unused 16-bit ranks, the loss and routing
checks, and a real harmless subprocess test of the timeout and partial-output
retention path.

## Compression provenance correction before outcomes

Independent retention review found a potential stale-hash error: the runner
records its gzip file's digest, while the original wrapper then recompressed
that same file using a different gzip API. The
[Python 3.11 documentation](https://docs.python.org/3.11/library/gzip.html)
describes why `gzip.compress(..., mtime=0)` can emit a different header from
`GzipFile`, even when both streams decompress to identical raw bytes.

The wrapper now preserves an existing valid gzip byte-for-byte, verifies that
its decompressed content matches the raw predictions, and creates a gzip only
when one is absent. A mismatched existing file is rejected and preserved for
inspection. Two additional offline regressions check exact compressed-byte/hash
preservation and rejection without rewriting evidence. This changes retention,
not empirical computation or scientific scope.

Both corrections were independently reviewed before another offline validation
and before any pretrained binding-study outcome. The original failed validation
and its retained logs remain part of the execution history.

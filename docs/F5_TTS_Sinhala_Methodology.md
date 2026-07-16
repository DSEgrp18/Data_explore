# Testing the F5-TTS methodology for Sinhala

## Research question

Does F5-TTS's inference-time Sway Sampling improve Sinhala synthesis
intelligibility and naturalness at the same inference cost, and does a
Sinhala-specific text frontend provide an additional improvement?

This is a practical adaptation of Chen et al. (2024), not an attempt to
reproduce their 95,000-hour multilingual training run. Their base model used
335.8M parameters and eight A100 80 GB GPUs for more than a week; training that
model from scratch is outside this project's compute and data budget.

## Paper components we will retain

- A flow-matching F5-TTS checkpoint and character-level conditioning.
- Cross-sentence synthesis from a 4–10 second reference prompt and its verified
  transcript.
- Classifier-free guidance strength 2, Euler ODE solver, and three fixed random
  seeds.
- Sway Sampling coefficient `s = -1`, the paper's default. This changes only
  inference flow steps and requires no retraining.
- Paired comparison at 16 and 32 neural function evaluations (NFE).
- Automatic intelligibility, speaker similarity, naturalness ratings, and
  real-time factor (RTF).

## Sinhala-specific adaptation

The main comparison uses the same Sinhala checkpoint, prompt, sentence, seed,
hardware, and solver for every condition. Only NFE, Sway Sampling, or text
normalization changes. The five registered conditions are in
`experiments/f5_sinhala_ablation.json`.

The evaluation set should contain at least 100 unseen sentences, balanced over:

1. plain Sinhala;
2. numerals, currency, dates, times, and abbreviations;
3. English/Sinhala code-switching and loanwords;
4. rare conjuncts and Sanskrit/Pali-derived words;
5. long or repetition-prone sentences.

Use held-out text rather than FLEURS audio as the target for synthesis. Prompt
audio must not contain the target sentence. Keep speakers separated from any
fine-tuning split to prevent leakage.

## Hypotheses and comparisons

- **H1:** `sway_16` has lower CER than `uniform_16` without a higher RTF.
- **H2:** `sway_16` is non-inferior to `uniform_32` in CER while being faster.
- **H3:** `sway_16` has higher native-speaker MOS than `uniform_16`.
- **H4:** `sway_16` outperforms `sway_16_raw_text` on normalization-heavy text.

CER is primary because Sinhala word spacing and ASR tokenization can make WER
unstable. Report WER as secondary. Compute micro-averaged CER/WER from a Sinhala
ASR model, RTF as synthesis seconds divided by generated-audio seconds, and
speaker similarity using one fixed embedding model. Report metrics overall and
by sentence category.

For significance, use paired bootstrap confidence intervals over sentences for
CER/RTF and a paired ordinal or Wilcoxon analysis for listener scores. Keep all
three seed outputs; do not select the best seed. MOS audio should be randomized
and condition labels hidden from listeners.

## Workflow that needs no large download

1. Replace the example rows in `evaluation/sample_manifest.csv` with the final
   balanced text set and locally recorded/available prompt paths.
2. Validate and expand the run matrix:

   ```bash
   python3 scripts/prepare_f5_runs.py \
     --manifest evaluation/sample_manifest.csv \
     --config experiments/f5_sinhala_ablation.json \
     --output runs/f5_sinhala/run_sheet.json
   ```

3. When the checkpoint is available, connect its inference CLI to each run-sheet
   row and record synthesis time and audio duration. The upstream F5-TTS CLI uses
   `--nfe_step` and `--sway_sampling_coef`; confirm names against the exact
   installed revision before running.
4. Transcribe every generated file with one fixed Sinhala ASR checkpoint. Save a
   CSV containing `reference,hypothesis`, then run:

   ```bash
   python3 scripts/score_transcripts.py results/transcripts.csv
   ```

5. Analyze paired results and conduct the blinded MOS study only after all
   conditions finish. Log checkpoint hashes, code revision, GPU, package
   versions, prompts, seeds, and failed generations.

## Decision rule

Adopt Sway Sampling in the application if it improves paired CER or MOS at the
same NFE without a material RTF regression. Use 16 NFE if it is non-inferior to
32 NFE within a predeclared CER margin; otherwise use 32 NFE. Treat the text
frontend as a separate contribution and report both raw and normalized results.

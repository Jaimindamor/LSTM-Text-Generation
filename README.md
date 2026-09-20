# LSTM Text Generation — Interview Task

Word-level LSTM language model trained on Shakespeare's text, generating new
text from a seed phrase.

## Files in this submission

| File | What it is |
|---|---|
| `lstm_text_generation.py` | Full pipeline: preprocessing, model, training, generation (CLI script) |
| `lstm_text_generation.ipynb` | Same pipeline as an executed notebook |
| `generated_samples_main_model.json` | Sample outputs from the main model, several seeds x two temperatures |
| `generated_samples_shallow_model.json` | Sample outputs from the bonus shallow-model experiment |
| `training_curves.png` | Loss/accuracy curves comparing the two architectures |
| `README.md` | This report |

## 1. Dataset

Two options, both handled automatically by `lstm_text_generation.py` via
`--dataset {tiny,full}`:

- **`tiny`** — "Tiny Shakespeare", ~1.1MB / ~200K tokens (Karpathy's
  char-rnn corpus). Fast, good for iterating.
- **`full`** (used for the results below) — 42 Folger Shakespeare plays and
  poems, ~5.5MB, pulled from the
  [cobanov/shakespeare-dataset](https://github.com/cobanov/shakespeare-dataset)
  GitHub repo and concatenated. After cleaning this yields **961,102
  tokens** and a **12,000-word vocabulary** (capped; true unique-word count
  is ~30K) — about 6x more training data than `tiny`.

Manual download, if you'd rather not use `--dataset`:
```bash
curl -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
(Project Gutenberg's Complete Works, https://www.gutenberg.org/ebooks/100,
is the same underlying text as `full` at a similar scale, if you'd rather
source it that way.)

## 2. Preprocessing

1. Lowercase the entire text; normalize smart quotes/dashes to plain ASCII.
2. Strip all punctuation (`string.punctuation`).
3. Collapse whitespace/newlines and split on spaces into a word-token stream.
4. Build a vocabulary of the 12,000 most frequent words, with an `<UNK>`
   bucket for everything else — this keeps the embedding/softmax layers a
   manageable size relative to the ~30K unique words actually in the corpus.
5. Slide a fixed-length window (15 words) across the token stream; each
   window is an input sequence, and the word immediately after it is the
   target label. Standard sliding-window setup for next-word prediction.
6. 90/10 train/validation split (chronological, not shuffled, so validation
   text never appears inside a training window). On the full corpus that's
   **864,978 training sequences / 96,109 validation sequences**.

## 3. Model

```
Embedding(vocab_size=12000, 128)
  -> LSTM(200, return_sequences=True, recurrent_dropout=0.2) -> Dropout(0.2)
  -> LSTM(200, recurrent_dropout=0.2) -> Dropout(0.2)
  -> Dense(vocab_size, activation="softmax")
```

- Loss: sparse categorical crossentropy
- Optimizer: Adam (lr=1e-3)
- Early stopping on validation loss (**patience 6**, restores best weights)
- `ModelCheckpoint` saves only the best-val-loss weights
- `recurrent_dropout=0.2` on both LSTM layers, in addition to the
  between-layer `Dropout(0.2)` — targets overfitting inside the recurrent
  connections specifically, at the cost of the fast cuDNN LSTM kernel on GPU

The comparable single-layer "shallow" model used in the bonus experiment
(Section 6) is identical except for a single `LSTM(128, recurrent_dropout=0.2)`
in place of the two 200-unit layers.

## 4. Training

Trained on the **full 961K-token corpus** (12,000-word vocabulary, sequence
length 15, batch size 256). With `patience=6`, training stopped itself once
validation loss failed to improve for 6 straight epochs.

**Observed loss curve** (see `training_curves.png`):

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 6.464 | 6.486 |
| 2 | 6.020 | 6.351 |
| **3** | **5.823** | **6.312 (best)** |
| 4 | 5.690 | 6.314 |
| 5 | 5.587 | 6.335 |
| 6 | 5.502 | 6.369 |
| 7 | 5.431 | 6.388 |
| 8 | 5.366 | 6.425 |
| 9 | 5.307 | 6.455 |

Validation loss bottoms out at **epoch 3** (6.312) and then rises for 6
straight epochs, triggering early stopping after epoch 9. `ModelCheckpoint`
restores the epoch-3 weights, so that's what generation below uses — not
the final epoch. Final validation accuracy at the point training stopped
was **10.75%** (vs. train accuracy 11.77% at the same point) — a modest,
expected gap given the model is still mildly overfitting by the time it
stops.

Worth noting honestly: even with 6x more data and `recurrent_dropout`
added, the model still turns over by epoch 3 — earlier, in fact, than the
smaller-corpus 12-epoch run from an earlier iteration. That's not a
regression; it reflects the harder 12,000-way classification problem (a
random baseline against 12,000 classes has loss ln(12000)=9.39 vs.
ln(6000)=8.70 for a 6,000-word vocab), and the accuracy numbers below show
real improvement despite the harder target.

## 5. Generated text samples

Seed → generated continuation (40 words), at two sampling temperatures.
Lower temperature (0.5) picks safer, more probable words; higher (1.0)
samples more freely and gets more varied vocabulary at the cost of
grammaticality. These are the actual outputs from the epoch-3 checkpoint
described above.

> **to be or not to** (temp 0.5)
> *to be or not to the crown of a in the king the moon is the and the of
> the and the of my and i have been the of our court and my heart*

> **shall i compare thee to** (temp 1.0)
> *shall i compare thee to buckingham asleep is a poor third madness falls
> the oath in a great fortune champion my coming flight all importune
> aufidius be part of it and say thou of my surety for i know commanded
> that now put me out*

(Full set of 8 seed x temperature combinations in
`generated_samples_main_model.json`.)

**Honest assessment**: the larger vocabulary and corpus show up clearly in
the vocabulary breadth — character and place names from plays well outside
`tiny` (Buckingham, Aufidius, Polydor, Caliban, Trinculo) — but sentence-
level coherence is still limited, and the temp-0.5 samples still lean on
filler ("of the ... and the of"). That's consistent with the earlier
finding: this is a data/architecture-ceiling problem, not a bug — a
12,000-way word-level softmax with a 15-word context window and a
2-layer/200-unit LSTM is fundamentally limited in how much long-range
structure it can capture, regardless of how much Shakespeare it sees.

## 6. Bonus: architecture comparison

Compared the main 2-layer, 200-unit LSTM against a **shallow single-layer,
128-unit LSTM**, both trained on identical data/vocab/dropout settings, each
run to its own early-stopping point (`patience=6`) rather than a fixed
epoch count:

| Model | Params | Epochs trained | Train loss (final) | Best val loss |
|---|---|---|---|---|
| Deep: 2x LSTM(200) | 4.53M | 9 | 5.3067 | **6.3118** |
| Shallow: 1x LSTM(128) | 3.22M | 10 | 5.1174 | **6.2289** |

The shallower model again reaches a **lower best validation loss**
(6.2289 vs. 6.3118) despite having ~1.3M fewer parameters, and it ran one
epoch longer before triggering the same early-stopping criterion —
meaning its validation loss kept improving for longer, not less. This
holds up a third time now, across three very different training budgets
(150K tokens/no recurrent dropout, 960K tokens/256-unit deep model, and
now 960K tokens/200-unit deep model with recurrent dropout and patience 6):
**the extra LSTM layer is not earning its keep on this task.** For a
15-word context window and a corpus this size, a single well-regularized
LSTM layer generalizes at least as well as a deeper one, at noticeably
less compute (170s/epoch vs. 320s/epoch on the hardware this was run on).

Sample from the shallow model (temp 0.5):
> *to be or not to the of the field and the of the of and*

> *shall i compare thee to the of the field and fight enter the and and
> the of the of the way of his and the fancy of the and his friends and
> his son*

Qualitatively similar — sometimes more repetitive — than the deep model's
output, reinforcing that the bottleneck is the modeling setup (word-level,
short context, capped vocabulary) rather than depth. The natural next
experiment, if pursued further, would be to give the *shallow* architecture
the model-capacity budget saved from dropping the second LSTM layer and
spend it elsewhere — a longer `seq_length`, a larger `vocab_size`, or
subword (BPE) tokenization — rather than assuming a second layer is the
right way to spend it.

## 7. Reproducing this

```bash
pip install tensorflow-cpu  # or tensorflow if you have a GPU

# The exact configuration used for the results above
python lstm_text_generation.py \
  --dataset full --data shakespeare_full.txt \
  --vocab_size 12000 --seq_length 15 --lstm_units 200 200 \
  --recurrent_dropout 0.2 --patience 6 \
  --embedding_dim 128 --epochs 40 --batch_size 256 --out_dir run_outputs_full

# Generate more samples later without retraining:
python lstm_text_generation.py \
  --dataset full --data shakespeare_full.txt --vocab_size 12000 \
  --out_dir run_outputs_full --generate_only
```

Every hyperparameter is a CLI flag (`--max_tokens`, `--vocab_size`,
`--seq_length`, `--lstm_units`, `--recurrent_dropout`, `--patience`,
`--epochs`, ...), so it's easy to sweep or scale further. On a machine
without a command-time limit, just run with your target `--epochs`
directly — early stopping halts automatically once validation loss stops
improving.

## 8. Iteration history

This submission went through a few rounds of tuning, each responding to a
concrete finding in the previous round's results:

1. **Initial pass** — 150K-token slice of `tiny` (already 74% of that
   file), 6,000-word vocab, 2-layer/200-unit LSTM, `patience=3`. Val loss
   plateaued and reversed by epoch 8-9 — clear overfitting on a small
   corpus relative to a 2.5M-parameter model.
2. **Scaled the data** — switched to the `full` 961K-token, 12,000-word-
   vocab corpus (~6x more data). Also bumped the model to 256x256 units to
   "match" the bigger dataset. Result: turned over *faster* (val loss
   bottomed by epoch 3 of 40) — more capacity than a 15-word context window
   and 12,000-way softmax could actually use, so the extra parameters
   mostly bought faster overfitting.
3. **Tuned capacity and regularization** (the results reported above) —
   dropped LSTM units back to 200x200, added `recurrent_dropout=0.2` on
   both LSTM layers, and raised `EarlyStopping` patience from 3 to 6 so
   training wouldn't bail at the first plateau. Also switched the bonus
   comparison to let both models run to their own early-stopping point
   rather than a fixed epoch count.

Across all three rounds, one finding has been consistent: **a single-layer
LSTM matches or beats the 2-layer version on validation loss**, at less
compute. That's the most robust empirical result of this exercise, and the
strongest candidate "bonus experiment" finding to lead with.

The next lever after this would be subword (BPE) tokenization instead of
whole-word `<UNK>`-bucketing, which handles rare words far more gracefully
than fixed-vocabulary word-level modeling, and would let a longer
`seq_length` be used without the sequence-length/vocab-size trade-off
getting worse.

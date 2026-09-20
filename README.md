# LSTM Text Generation — Interview Task

Word-level LSTM language model trained on Shakespeare's text, generating new
text from a seed phrase.

## Files in this submission

| File | What it is |
|---|---|
| `lstm_text_generation.py` | Full pipeline: preprocessing, model, training, generation (CLI script) |
| `generated_samples_main_model.json` | Sample outputs from the main model, several seeds x two temperatures |
| `generated_samples_shallow_model.json` | Sample outputs from the bonus shallow-model experiment |
| `training_curves.png` | Loss/accuracy curves comparing the two architectures |
| `README.md` | This report |

## 1. Dataset

Two options, both handled automatically by `lstm_text_generation.py` via
`--dataset {tiny,full}`:

- **`tiny`** — "Tiny Shakespeare", ~1.1MB / ~200K tokens (Karpathy's
  char-rnn corpus). Fast, good for iterating.
- **`full`** (recommended) — 42 Folger Shakespeare plays and poems, ~5.5MB /
  ~960K tokens / ~30K unique words, pulled from the
  [cobanov/shakespeare-dataset](https://github.com/cobanov/shakespeare-dataset)
  GitHub repo and concatenated. ~5x more training data and a much richer
  vocabulary than `tiny` — this is what closes most of the gap between
  "proof of concept" and a model that actually holds together for a few
  words at a time.

Manual download, if you'd rather not use `--dataset`:
```bash
curl -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
(Project Gutenberg's Complete Works, https://www.gutenberg.org/ebooks/100,
is the same underlying text as `full` at a similar scale, if you'd rather
source it that way.)

## 2. Preprocessing

1. Lowercase the entire text.
2. Strip all punctuation (`string.punctuation`).
3. Collapse whitespace/newlines and split on spaces into a word-token stream.
4. Build a vocabulary of the N most frequent words (default 6,000–8,000),
   with an `<UNK>` bucket for everything else — this keeps the
   embedding/softmax layers a manageable size.
5. Slide a fixed-length window (`seq_length` words) across the token stream;
   each window is an input sequence, and the word immediately after it is
   the target label. This is a standard sliding-window setup for next-word
   prediction.
6. 90/10 train/validation split (chronological, not shuffled, so validation
   text never appears inside a training window).

## 3. Model

```
Embedding(vocab_size, 128)
  -> LSTM(200, return_sequences=True) -> Dropout(0.2)
  -> LSTM(200) -> Dropout(0.2)
  -> Dense(vocab_size, activation="softmax")
```

- Loss: sparse categorical crossentropy
- Optimizer: Adam (lr=1e-3)
- Early stopping on validation loss (patience 3, restores best weights)
- `ModelCheckpoint` saves only the best-val-loss weights

## 4. Training

Trained on a 150,000-token slice of the corpus (6,000-word vocabulary,
sequence length 15, batch size 256) for 12 epochs on CPU. Training was done
in 2-epoch checkpointed chunks (`--resume`) because of the sandbox's 1-core
CPU and 5-minute-per-command limit — the script supports this natively via
`--resume` / `--generate_only`, so it also works as a normal single
`--epochs 30` run on a machine with more time or a GPU.

**Observed loss curve** (see `training_curves.png`):

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 6.62 | 6.47 |
| 6 | 5.83 | 6.39 (best) |
| 12 | 5.34 | 6.50 |

Validation loss bottoms out around epoch 6-8 and then creeps back up while
training loss keeps falling — classic overfitting on a fairly small
150K-token corpus with a 2.5M-parameter model. `ModelCheckpoint` keeps the
epoch-8 weights (lowest val loss), so that's what generation uses, not the
final epoch.

## 5. Generated text samples

Seed → generated continuation (40 words), at two sampling temperatures.
Lower temperature (0.5) picks safer, more probable words; higher (1.0)
samples more freely and gets more varied vocabulary at the cost of
grammaticality.

> **to be or not to** (temp 0.5)
> *to be or not to the hundred leave of the with the prince and many king
> richard iii and i will not i am the of the world and i shall i shall not
> the queens babe of the of the*

> **shall i compare thee to** (temp 1.0)
> *shall i compare thee to none alas you it not now once along as thou we
> be did revolt a foe tis can suspect keeps you if you will the volsces go
> the gracious windows have you may obdurate not met go that romeo*

(Full set of 8 seed x temperature combinations in
`generated_samples_main_model.json`.)

**Honest assessment**: at this training budget the model has clearly picked
up Shakespearean vocabulary, character names (Richard III, Margaret,
Coriolanus, Warwick), and some local grammar (articles before nouns, "i
shall", "thy lord"), but it hasn't learned long-range coherence — sentences
drift and the low-temperature samples fall into loops on filler words
("of the ... of the"). That's expected for a word-level LSTM on ~150K
training tokens over 12 epochs on CPU; a char-level model or a much larger
corpus / more epochs / a GPU would be the next steps toward materially more
coherent output.

## 6. Bonus: architecture comparison

Compared the main 2-layer, 200-unit LSTM against a **shallow single-layer,
128-unit LSTM**, both trained on identical data/vocab for the same 6 epochs.

| Model | Params | Train loss @ epoch 6 | Val loss @ epoch 6 |
|---|---|---|---|
| Deep: 2x LSTM(200) | 2.56M | 5.83 | 6.39 |
| Shallow: 1x LSTM(128) | 1.67M | 5.68 | **6.18** |

The shallower model actually reaches a **lower validation loss** at the same
epoch budget, and its train/val gap opens more slowly (see
`training_curves.png`). With ~150K training tokens, the deeper/wider model's
extra capacity mostly buys faster overfitting rather than better
generalization — a useful, if slightly counterintuitive, result: for a
corpus this size, the smaller architecture is the better starting point, and
the extra LSTM layer would likely pay off only with more data, dropout/
regularization tuning, or a shorter sequence length to reduce
effective parameters relative to the data.

Sample from the shallow model (temp 0.5):
> *to be or not to the tender and the and with me to their noble the of
> the and the of the of and on her and my lord*

Qualitatively similar quality to the deep model — reinforcing that the
bottleneck here is data/epochs, not architecture depth.

## 7. Reproducing this

```bash
pip install tensorflow-cpu  # or tensorflow if you have a GPU

# Recommended: full corpus, tuned capacity/regularization (current defaults)
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
`--seq_length`, `--lstm_units`, `--epochs`, ...), so it's easy to sweep or
scale further. On a machine without a 5-minute command limit, just run with
your target `--epochs` directly — early stopping halts automatically once
validation loss stops improving, no chunking needed.

## 8. Scaling up: why the full corpus matters

The original submission trained on a 150K-token slice of `tiny` (already
74% of everything `tiny` has). The `full` corpus is ~6x bigger and ~2.3x
richer in vocabulary (30K vs 13K unique words), which directly targets the
main limitation identified during evaluation: with only 150K tokens for a
2.5M-parameter model, validation loss plateaued and reversed by epoch 8-9
(visible overfitting).

## 9. Round two: capacity and regularization tuning

Running the full-corpus model at 256x256 LSTM units turned over even
*faster* than before (val_loss bottomed by epoch 3 of 40), despite 6x more
data. Comparing loss values directly across runs is also misleading here,
worth flagging explicitly: the vocabulary grew from 6,000 to 12,000 words,
and a random baseline against 12,000 classes has a higher loss floor
(ln(12000)=9.39) than against 6,000 (ln(6000)=8.70) - so the *accuracy*
numbers (which did improve, ~8.6%→~10.5% val accuracy) are the fairer
before/after comparison, not the raw loss.

Diagnosis: 256x256 units is more capacity than a 15-word context window and
12,000-way softmax can actually make use of - more data raises the ceiling
on what a model *could* learn, but doesn't help if the architecture
converges to its optimum before exploiting it. Changes made in response:

- **Dropped LSTM units back to 200x200** — the smaller shallow (128-unit)
  model was already matching or beating the 256-unit model's validation
  accuracy at every epoch checked, at ~40% less time per epoch.
- **Added `recurrent_dropout=0.2`** on both LSTM layers, on top of the
  existing between-layer dropout — this specifically targets overfitting
  in the recurrent connections, where between-layer dropout alone doesn't
  reach. (Trade-off: this disables the fast cuDNN LSTM kernel, so it's
  slower on GPU; no effect on CPU-only training.)
- **Raised `EarlyStopping` patience from 3 to 6** — patience 3 was cutting
  training right at the first plateau, before it was clear whether that
  plateau was durable or just a bump.
- **Bonus comparison now runs both models to their own early-stopping
  point** (same `EPOCHS`/`PATIENCE` budget) rather than a fixed epoch
  count, and compares best-val-loss + epochs-to-converge rather than a
  same-epoch snapshot — a fairer comparison once both runs can stop at
  different times.

CLI flags for all of this: `--lstm_units 200 200`, `--recurrent_dropout
0.2`, `--patience 6` (all shown as the new defaults in `--help`).

The next lever after this would be subword (BPE) tokenization instead of
whole-word `<UNK>`-bucketing, which handles rare words far more gracefully
than either vocabulary size used here.

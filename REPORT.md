# Music Generation — Project Report

A from-scratch deep-learning project: seven different model families, each trained to compose
ragtime piano music as a sequence of MIDI events. The point is learning — how each architecture
works, how it trains, and where it breaks — rather than chasing a single best model.

For installing, running the app and deployment, see [README.md](README.md).

---

## 1. Goal

Train and compare sequence models that **generate new piano music**, and serve them behind an API
and a web demo. Each model reads a sequence of musical events and learns to predict what comes next;
at generation time it writes a piece one event at a time.

## 2. Data

| | |
|---|---|
| Source | Public ragtime MIDI files scraped from `ragtimemusic.com` (notebook cell `download_dataset`) |
| Size | **446 pieces**, 67–879 s each (average 3.2 min) |
| Encoding | [midi-neural-processor](https://github.com/jason9693/midi-neural-processor), the event representation from **Performance RNN / Music Transformer** |
| Vocabulary | 391 tokens = 128 `note_on` + 128 `note_off` + 100 `time_shift` (10 ms steps) + 32 `velocity` + 3 specials (`pad`, `bos`, `eos`) |
| Length | 360–19,908 events per piece (average 8,428); about 45 events per second of music |
| Training segments | Each piece is cut into **600-event segments** (≈13 s): 6,479 segments, ≈3.76 M events, batches of 32 |
| Augmentation | A key-normalized copy of every piece (transposed to C major / A minor with `music21`) — used **only by GPT-2** |

There is **no held-out test set**: all models train and evaluate on the same pieces (see §4).

## 3. Setup

- **Framework:** PyTorch, in `music_generation.ipynb` (six models) and
  `huggingface_transformers.ipynb` (GPT-2). Trained on Google Colab GPUs and Apple Silicon (MPS).
- **Shared helpers:** `train_net` (Adam, gradient clipping, optional AMP/scheduler),
  `evaluate_net`, `generate_output` (tokens → `.midi`), `save_checkpoint` / `load_checkpoint`.
- **Serving:** a FastAPI service (`app/`, API-key protected) and a Streamlit front end (`main.py`),
  packaged with Docker; GPT-2 is published on the Hugging Face Hub.

## 4. How the models are measured

| Measure | What it tells you | Caveat |
|---|---|---|
| `evaluate_net` accuracy | Feeds 60 real events, then lets the model continue greedily, and scores position by position | Scored on training data; ~10 points are free (the 59 primer events count as correct); once a continuation diverges from the original piece, matches drop to chance. A random network scores ≈11%, a good one ≈16–17%. |
| Teacher-forced accuracy | How often the model's top guess for the next event is right, given the real history | Measured on training pieces **and** on transposed copies the model never saw — the gap reveals memorization |
| Free-running statistics | Pitch/timing distributions, notes per second, note-on/off pairing, note durations, **notes sounding at once** | Real ragtime: ~12 notes/s, median note 0.21 s, ~3 notes sounding at once |

---

## 5. The seven models

### Summary

| Model | Params | Training | `evaluate_net` | Teacher-forced acc (seen → unseen) |
|---|---|---|---|---|
| RNN | 4.6 M | 50 epochs, 4.1 h | 24.1% | 96.4% → 65.2% |
| CNN | 15.1 M | 50 epochs, 2.1 h | 18.1% | 86.8% → 61.5% |
| VAE | 16.8 M | 50 epochs, 3.2 h | 23.1% | 96.2% on seen (behaves like the RNN) |
| Transformer | 3.4 M | 50 epochs, 55 min | 17.3% | 80.1% → 72.7% |
| GAN | 3.4 M + 3.4 M | 600 iterations fine-tuning, 35 min | 16.8% | 79.7% on seen |
| A2C | 4.6 M | 30 epochs, 67 min | 17.2% | ~75% on seen (training log) |
| GPT-2 | 86.3 M | 20 epochs on Colab | — | — |

"Unseen" = the same pieces transposed by a few semitones: same style, but event sequences the model
never trained on.

### 5.1 RNN — recurrent language model

- **Inspired by:** GRU (Cho et al., 2014); character-level RNN language models; Magenta's
  **Performance RNN** (Simon & Oore, 2017), which uses this exact event representation.
- **Architecture:** embedding (256) → 3-layer GRU (512) → linear layer over the 391 events.
- **Training:** teacher-forced next-event cross-entropy, Adam at 1e-3, 50 epochs. Loss fell steadily
  the whole run.
- **Result:** the highest `evaluate_net` score (24.1%) and 96.4% teacher-forced accuracy — but on
  transposed pieces it drops to 65.2%. **It largely memorized the 446 pieces**, which also inflates its
  `evaluate_net` score (it can continue a known piece from memory). Its free-running output is
  nonetheless clean: sampled raw it keeps ~4.6 notes sounding at once (real ~3), with only ~3% of its
  output copied verbatim from the training set.

### 5.2 CNN — WaveNet

- **Inspired by:** **WaveNet** (van den Oord et al., 2016).
- **Architecture:** dilated causal 1-D convolutions (dilations 1…512, receptive field 1,025 events)
  with gated tanh·sigmoid units, residual and skip connections, and two 1×1 output convolutions.
- **Training:** same objective and optimizer as the RNN. The loss reached its lowest point around
  iteration 4,000–5,000 and then **rose steadily** for the rest of training (5-log-line average 170 →
  223). The saved checkpoint comes from that degraded end — most likely the constant 1e-3 learning
  rate being too high late in training (no decay schedule is active).
- **Result:** 86.8% → 61.5% teacher-forced (seen → unseen): it memorized too, and generalizes worst of
  the likelihood-trained models. A learning-rate schedule or keeping the best checkpoint would likely
  help most.

### 5.3 VAE — recurrent variational autoencoder

- **Inspired by:** VAE (Kingma & Welling, 2014), the sentence VAE (Bowman et al., 2016) and Magenta's
  **MusicVAE** (Roberts et al., 2018).
- **Architecture:** a bidirectional 3-layer GRU encoder compresses a segment into a 64-d latent `z`;
  a 3-layer GRU decoder regenerates the segment with `z` fed in at every step. Loss = reconstruction +
  KL divergence (the ELBO).
- **Training:** 50 epochs; loss fell steadily.
- **Result:** the code is correct, but the latent **collapsed**: the KL term is only 1.07 nats per
  piece across 64 dimensions, and the decoder scores the same with its own `z` as with `z = 0`. This is
  the classic failure of VAEs with strong autoregressive decoders (Bowman et al.): the decoder ignores
  the latent. So the VAE works as a second RNN (same 96% seen accuracy, same memorization) — sampling
  different `z` values does not change the music. KL annealing or word dropout are the standard fixes.

### 5.4 Transformer — decoder-only self-attention

- **Inspired by:** "Attention Is All You Need" (Vaswani et al., 2017), decoder-only as in GPT, and
  **Music Transformer** (Huang et al., 2018), which introduced this event vocabulary (Music Transformer
  uses relative attention; this one uses sinusoidal absolute positions).
- **Architecture:** 6 layers, d_model 256, 8 heads, feed-forward 512, post-LayerNorm, tied
  input/output embeddings, causal mask, fused attention (`scaled_dot_product_attention`).
- **Training:** 50 epochs, AdamW 3e-4 with warmup and linear decay, label smoothing 0.1, mixed
  precision. Loss 10.4 → 1.6.
- **Result:** lower seen accuracy (80.1%) than the RNN but **the best generalization** (72.7% on
  unseen pieces) — it learned music rather than memorizing pieces. With its default decoding (top-40,
  temperature 0.95) it **sounds good, comparable to the GAN**. Its one weakness appears only when
  sampling raw, without top-40: **label smoothing trains it to keep ~6% of its probability on
  unlikely events**, so stray note-ons accumulate (~16 notes sounding at once vs ~3 in real music).
  Top-40 filters those out, which is why the default output sounds right. This finding shaped the GAN
  work below.

### 5.5 GAN — Transformer generator + Transformer discriminator

- **Inspired by:** GAN (Goodfellow et al., 2014); **SeqGAN** (Yu et al., 2017) for MLE pretraining plus
  policy-gradient training of a token generator; **ScratchGAN** (de Masson d'Autume et al., 2019) for
  dense per-event rewards; **LeakGAN** (Guo et al., 2018) for interleaving MLE during adversarial
  training; **Transformer-GAN for symbolic music** (Muhamed et al., 2021) for a pretrained
  discriminator.
- **Architecture:** both networks start from the trained Transformer.
  - *Generator:* writes 600-event pieces with a key-value cache, so a rollout costs linear rather than
    quadratic time.
  - *Discriminator:* the Transformer body with a one-number head that scores every event as real or
    generated.
- **Training:** 600 iterations. Each one runs a discriminator step, a REINFORCE generator step on
  discounted per-event rewards `2·σ(D) − 1` with a moving-average baseline, and a plain cross-entropy
  MLE step on real data. The discriminator learns 10× slower (1e-5 vs 1e-4) after a 10-step warm-up;
  at equal speed it won within 50 steps and gave no usable signal.
- **How it got here:** earlier attempts at a *pure* GAN (no MLE) failed — the discrete-token gradient
  problem, a start-token leak that let the discriminator win by reading position 0, and an entropy
  bonus that pushed outputs toward noise. The working version is a pretrained hybrid, as in the
  published work.
- **Result** (raw sampling, compared with plain MLE fine-tuning of the same length):

  | | notes sounding at once | variety (distinct 4-grams) | accuracy |
  |---|---|---|---|
  | real music | ~2.9–3.0 | 0.72–0.74 | — |
  | Transformer, raw | 15.9 | 0.80 | 79.9% |
  | plain-CE fine-tune (control) | 4.9–5.0 | 0.75–0.77 | 79.9% |
  | **GAN** | **3.8–4.2** | **0.70–0.74** | 79.7% |

  Dropping label smoothing (the plain-CE MLE step) fixes most of the raw-sampling problem on its own.
  **The GAN's own contribution** is about 20% fewer stuck notes than that control — half the remaining
  gap to real music — and more realistic variety, with nothing else harmed (single training run; two
  sampling seeds). On stuck notes, its raw output even measures slightly cleaner than the original
  Transformer's top-40 output (5.1); by ear, the two sound comparable.

### 5.6 A2C — advantage actor-critic

- **Inspired by:** **A3C / A2C** (Mnih et al., 2016).
- **Architecture:** a 3-layer GRU shared by an *actor* (policy over the 391 events) and a *critic*
  (value estimate).
- **Training:** pure reinforcement learning from random initialization — no MLE anywhere. The actor
  samples an event and receives reward 1 only if it matches the real next event. The GRU state is
  teacher-forced, so each step is a contextual bandit (γ = 0) and the critic acts as a per-state
  baseline. Entropy bonus 0.05 keeps the policy from collapsing (0.01 collapsed; 0.2 gave noise).
- **Result:** next-event accuracy climbed from 0.5% to ~75% in 30 epochs purely from reward, and
  `evaluate_net` reached 17.2% — on par with the Transformer. It shows that RL with a sparse
  right/wrong reward can learn a language model, though less efficiently than MLE.
- **A2C vs the GAN:** both use policy gradients with a second network, but A2C's critic only
  *predicts* the reward of a fixed rule (teammates), while the GAN's discriminator *is* the reward and
  competes with the generator (opponents). A2C always trains on real history; the GAN trains on its
  own samples, which is why it can fix habits that only appear when the model writes by itself.

### 5.7 GPT-2 — Hugging Face causal language model

- **Inspired by:** **GPT-2** (Radford et al., 2019), following the Hugging Face course chapter
  "Training a causal language model from scratch".
- **Architecture:** Hugging Face `GPT2LMHeadModel` — 12 layers, 768-d, 12 heads, 86.3 M parameters —
  initialized **from scratch** (the architecture is borrowed, not the pretrained English weights).
  Events are written as text numbers and tokenized with a BPE tokenizer retrained on that text
  (vocabulary 587); context 256 tokens.
- **Training:** original + transposed pieces (874 total), 20 epochs, effective batch 256 (16 × 16
  gradient accumulation), learning rate 1e-4 with cosine decay, fp16, on a Colab GPU.
- **Result:** the largest model and the only one trained with augmented data. It generates 2,048-event
  pieces in 256-event chunks, each primed with the last 32 events. In the saved sample, later chunks
  fall into short repeating loops — a common failure when each chunk sees so little context. Its
  "validation" set is the first 100 training pieces, so it isn't a true held-out measure, and it
  wasn't evaluated with the notebook's metrics.

---

## 6. Key takeaways

1. **Memorization vs generalization.** The RNN, CNN and VAE score highest on seen pieces but drop
   sharply on transposed ones; the smaller, regularized Transformer generalizes best. A held-out split
   would make this visible in the notebook itself.
2. **A training setting can masquerade as a model problem.** Label smoothing, meant as a
   regularizer, is what made the Transformer depend on top-40 decoding to sound right.
3. **GANs on discrete sequences need a strong starting point.** Pure adversarial training did not
   work; a pretrained generator with a pretrained, slowed-down discriminator produced a real (if
   modest) improvement beyond plain MLE.
4. **Measure what you hear.** Several defects — hanging notes, broken durations, an unused latent —
   were invisible to accuracy and found only by checking decoded music against real statistics.
5. **Known loose ends:** no held-out test set; the CNN's late-training degradation; and the VAE's
   posterior collapse.

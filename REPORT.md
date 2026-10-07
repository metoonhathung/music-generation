# Music Generation — Project Report

A from-scratch deep-learning project: eight different model families, each trained to compose
ragtime piano music as a sequence of MIDI events. The point is learning — how each architecture
works, how it trains, and where it breaks — rather than chasing a single best model. A final
experiment (§6) asks a frontier LLM to compose in the same format with no training at all.

For installing, running the app and deployment, see [README.md](README.md).

---

## 1. Goal

Train and compare sequence models that **generate new piano music**, and serve them behind an API
and a web demo. Each model reads a sequence of musical events and learns to predict what comes next;
at generation time it writes a piece one event at a time. The exception is the diffusion model (§5.8),
which fills in hidden events anywhere in the piece and generates all of them over a series of steps.

## 2. Data

| | |
|---|---|
| Source | Public ragtime MIDI files scraped from `ragtimemusic.com` (notebook cell `download_dataset`) |
| Size | **446 pieces**, 67–879 s each (average 3.2 min) |
| Encoding | [midi-neural-processor](https://github.com/jason9693/midi-neural-processor), the event representation from **Performance RNN / Music Transformer** |
| Vocabulary | 391 tokens = 128 `note_on` + 128 `note_off` + 100 `time_shift` + 32 `velocity` + 3 specials (see below) |
| Length | 360–19,908 events per piece (average 8,428); about 45 events per second of music |
| Training segments | Each piece is cut into **600-event segments** (≈13 s): 6,479 segments, ≈3.76 M events, batches of 32 |
| Augmentation | A key-normalized copy of every piece (transposed to C major / A minor with `music21`) — used **only by GPT-2** |

There is **no held-out test set**: all models train and evaluate on the same pieces (see §4).

### The event vocabulary

A piece is a stream of events, like instructions to a player piano:

| IDs | Event | Meaning |
|---|---|---|
| 0, 1, 2 | `pad`, `bos`, `eos` | padding, start of a piece, end of a piece |
| 3–130 | `note_on` | press a key: pitch = ID − 3 (MIDI pitch; 60 = middle C, the piano spans 21–108) |
| 131–258 | `note_off` | release a key: pitch = ID − 131 |
| 259–358 | `time_shift` | wait (ID − 258) × 10 ms, from 10 ms to 1 s |
| 359–390 | `velocity` | loudness of the next `note_on`: (ID − 359) × 4, in 32 levels |

For example, a C-major chord (C4, E4, G4) played at velocity 80 and held for half a second:

```
379 63 | 379 67 | 379 70 | 308        | 191 195 198
vel C4 | vel E4 | vel G4 | wait 0.5 s | C4 E4 G4 off
```

How the encoder writes events:
- **Time** is rounded to 10 ms. A gap longer than 1 s becomes several waits in a row.
- **Events at the same moment** follow each other with no wait between them, so a chord is several
  notes in a row.
- **Every `note_on` has its own velocity event**, even when the loudness doesn't change. This is a
  quirk of the encoder, and it makes velocity about a quarter of all events.
- **Sustain-pedal presses** are folded into note lengths, and all instruments are merged into one
  stream.
- **IDs are shifted:** the encoder numbers events from 0, and the notebook shifts them up by 3 to make
  room for the specials. GPT-2 uses the unshifted IDs.

## 3. Setup

- **Framework:** PyTorch, in `music_generation.ipynb` (seven models and the LLM experiment) and
  `huggingface_transformers.ipynb` (GPT-2). Trained on Google Colab GPUs and Apple Silicon (MPS).
- **Shared helpers:** `train_net` (Adam, gradient clipping, optional AMP/scheduler),
  `evaluate_net`, `generate_output` (tokens → `.midi`), `save_checkpoint` / `load_checkpoint`.
- **Serving:** a FastAPI service (`app/`, API-key protected) and a Streamlit front end (`main.py`),
  packaged with Docker; GPT-2 is published on the Hugging Face Hub.

## 4. How the models are measured

| Measure | What it tells you | Caveat |
|---|---|---|
| `evaluate_net` accuracy | Feeds 60 real events, then lets the model continue with its own decoding, and scores position by position | Scored on training data; ~10 points are free (the 59 primer events count as correct); once a continuation diverges from the original piece, matches drop to chance. A random network scores ≈11%, a good one ≈16–17%. |
| Teacher-forced accuracy | How often the model's top guess for the next event is right, given the real history | Measured on training pieces **and** on transposed copies the model never saw — the gap reveals memorization |
| Free-running statistics | Pitch/timing distributions, notes per second, note-on/off pairing, note durations, **notes sounding at once** | Real ragtime: ~12 notes/s, median note 0.21 s, ~3 notes sounding at once |
| Inference time | Seconds to generate one 600-event piece on CPU, as both apps run | Measured on an Apple M5 Pro (6 threads); hosted CPUs are slower, but the ranking holds |

---

## 5. The eight models

### Families

Each model falls into one family by **how it learns to generate**. Separately, each is built from one
of three kinds of network: **recurrent** (GRU), **convolutional** or **attention** (Transformer).

| Family | How it learns to generate | Models |
|---|---|---|
| Autoregressive | Predicts the next event from the ones before it, and writes left to right | RNN, CNN, Transformer, GPT-2 |
| Latent-variable | Compresses a piece into a code and learns to rebuild it; new codes give new pieces | VAE |
| Adversarial | A generator learns to fool a discriminator that tells real pieces from generated ones | GAN |
| Reinforcement learning | A policy learns from a reward for every event it writes | A2C |
| Diffusion | Hides part of a piece and learns to fill it in; generates starting from a fully hidden piece | Diffusion |

### Summary

| Model | Family | Network | Starts from | Params | Training | `evaluate_net` | Accuracy (seen → unseen) |
|---|---|---|---|---|---|---|---|
| RNN | Autoregressive | GRU | scratch | 4.6 M | 50 epochs, 4.1 h | 24.1% | 96.4% → 65.2% |
| CNN | Autoregressive | Dilated convolutions | scratch | 15.1 M | 50 epochs, 2.1 h | 18.1% | 86.8% → 61.5% |
| Transformer | Autoregressive | Transformer decoder | scratch | 3.4 M | 50 epochs, 55 min | 17.3% | 80.1% → 72.7% |
| GPT-2 | Autoregressive | Transformer decoder (Hugging Face) | scratch | 86.3 M | 20 epochs on Colab | — | — |
| VAE | Latent-variable | GRU encoder + decoder | scratch | 16.8 M | 50 epochs, 3.2 h | 23.1% | 96.2% on seen |
| GAN | Adversarial | 2 × Transformer decoder | **trained Transformer** | 3.4 M + 3.4 M | 600 iterations, 35 min | 16.8% | 79.7% on seen |
| A2C | Reinforcement learning | GRU | scratch | 4.6 M | 30 epochs, 67 min | 17.2% | ~75% on seen |
| Diffusion | Diffusion | Bidirectional Transformer | scratch | 17.0 M | 100 epochs, 3.9 h | 17.1% | 61.9% → 59–60%* |

- **Only the GAN starts from trained weights:** both of its networks begin as the trained
  Transformer. GPT-2 borrows only the architecture, and Diffusion reuses the Transformer's code but
  not its weights.
- **"Unseen"** means the same pieces transposed by a few semitones: the same style, but event
  sequences the model never trained on.
- **Accuracy** is teacher-forced next-event accuracy. \*Diffusion doesn't predict the next event, so
  its number is hidden-event accuracy averaged over hiding 5–95% of a piece. It is not directly
  comparable with the others.

### Inference cost

Time to generate one 600-event piece on CPU (Apple M5 Pro, 6 threads), which is how both apps run,
most expensive first. Memory is the float32 weights the apps hold (4 bytes per parameter).

| Rank | Model | Time per 600 events | Weights in memory | Same primer → same piece? | Why |
|---|---|---|---|---|---|
| 1 | Diffusion | 14.3 s | 68 MB | No (samples) | 400 full passes over the piece: 100 reveal steps + 300 correction rounds |
| 2 | CNN | 9.0 s | 60 MB | Yes (greedy) | Recomputes the whole convolution stack over the piece so far, for every new event |
| 3 | Transformer | 6.9 s | 14 MB | No (samples) | Re-reads the whole piece so far for every new event (no key-value cache) |
| 3 | GAN | 6.9 s | 14 MB | No (samples) | The Transformer's `predict`; the discriminator isn't used |
| 5 | GPT-2 | ≈3–5 s (estimate) | 345 MB | No (samples) | The largest, but it caches keys and values, so each new token costs the same |
| 6 | VAE | 0.13 s | 67 MB | No (greedy, but a random `z` each time) | A GRU decoder with a fixed-size state: the same work for every event |
| 7 | RNN | 0.12 s | 18 MB | Yes (greedy) | Same as the VAE |
| 7 | A2C | 0.12 s | 18 MB | Yes (greedy) | Same as the VAE |

- **Greedy models always write the same piece for a primer.** RNN, CNN and A2C always pick the most
  likely event. The VAE does too, but its fresh random `z` makes each run differ.
- **Time and memory rank differently.** GPT-2 needs the most memory but is mid-table on time. The
  three GRU models are over 50× faster than any convolution or attention model here.
- **The GPT-2 figure is an estimate:** 1.5 s per 256-token chunk, measured on an untrained model of
  the same size, times 2–3 chunks per piece. No GPT-2 weights were on disk.
- **The order holds on the apps' hosts.** Streamlit Cloud and Cloud Run have slower CPUs, so every
  time grows there.

Every section below uses the same seven headings: family, inspiration, architecture, training,
inference, result and takeaway.

### Autoregressive models

#### 5.1 RNN — recurrent language model

- **Family:** autoregressive · recurrent · trained from scratch.
- **Inspired by:** GRU (Cho et al., 2014), character-level RNN language models and Magenta's
  **Performance RNN** (Simon & Oore, 2017), which uses this exact event representation.
- **Architecture:** an embedding (256), a 3-layer GRU (512) and a linear layer over the 391 events;
  4.6 M parameters.
- **Training:** teacher-forced next-event cross-entropy, Adam 1e-3, 50 epochs (4.1 h). The loss fell
  steadily for the whole run.
- **Inference:** 0.12 s per 600 events, among the cheapest. It is greedy (it always picks the most
  likely event), so a primer always gives the same piece.
- **Result:** the top `evaluate_net` score (24.1%) and 96.4% seen accuracy, but 65.2% unseen: it
  **largely memorized** the 446 pieces, which also inflates its `evaluate_net` score.
- **Takeaway:** it is the best at continuing music it knows, not at inventing new music. A held-out
  split would expose the difference.

#### 5.2 CNN — WaveNet

- **Family:** autoregressive · convolutional · trained from scratch.
- **Inspired by:** **WaveNet** (van den Oord et al., 2016): dilated causal convolutions with gated
  units.
- **Architecture:** dilated causal 1-D convolutions (dilations 1…512, receptive field 1,025 events),
  gated tanh·sigmoid units, residual and skip connections, and two 1×1 output convolutions.
- **Training:** as the RNN, 50 epochs (2.1 h). The loss bottomed out around iteration 4,000–5,000,
  then **rose steadily** (170 → 223). The saved checkpoint comes from that degraded end.
- **Inference:** 9.0 s per 600 events, the second most expensive: it recomputes the convolution stack
  for every new event. It is greedy, so a primer always gives the same piece.
- **Result:** 86.8% seen → 61.5% unseen: it memorized too, and generalizes worst of the models trained
  on likelihood. The likely cause is the constant 1e-3 learning rate, which never decays.
- **Takeaway:** a learning-rate schedule, or keeping the best checkpoint rather than the last, would
  likely help most.

#### 5.3 Transformer — decoder-only self-attention

- **Family:** autoregressive · attention · trained from scratch.
- **Inspired by:** "Attention Is All You Need" (Vaswani et al., 2017), decoder-only as in GPT, and
  **Music Transformer** (Huang et al., 2018), which introduced this event vocabulary.
- **Architecture:** 6 layers, d_model 256, 8 heads, feed-forward 512, post-LayerNorm, tied
  embeddings, causal mask, sinusoidal positions; 3.4 M parameters.
- **Training:** AdamW 3e-4 with warmup and linear decay, label smoothing 0.1, mixed precision,
  50 epochs (55 min). Loss 10.4 → 1.6.
- **Inference:** 6.9 s per 600 events: it re-reads the whole piece for every new event (no key-value
  cache). It samples (temperature 0.95, top-40), so each run differs.
- **Result:** **the best generalization** (80.1% seen → 72.7% unseen), and it sounds good. Only when
  sampled without top-40 does label smoothing's ~6% on unlikely events show up as stray notes.
- **Takeaway:** this is the reference model. Its weights seed the GAN, and its code is reused by the
  diffusion model.

#### 5.4 GPT-2 — Hugging Face causal language model

- **Family:** autoregressive · attention · trained from scratch (only the architecture is borrowed).
- **Inspired by:** **GPT-2** (Radford et al., 2019), following the Hugging Face course chapter
  "Training a causal language model from scratch".
- **Architecture:** `GPT2LMHeadModel`, 12 layers, 768-d, 12 heads, 86.3 M parameters. Events are
  written as text numbers, with a retrained BPE tokenizer (vocabulary 587); context 256 tokens.
- **Training:** original + transposed pieces (874), 20 epochs, effective batch 256, learning rate
  1e-4 with cosine decay, fp16, on a Colab GPU. It is the only model trained with augmented data.
- **Inference:** ≈3–5 s per 600 events (estimated). It is the largest model but caches keys and
  values. It samples, so each run differs.
- **Result:** it writes in 256-token chunks, each primed with the previous chunk's last 32 events. Its
  "validation" set is training data, and it has no notebook metrics.
- **Takeaway:** the largest model, and the only one trained on augmented data, is also the least
  measured. A true held-out set and the notebook's metrics would show what its size buys.

### Latent-variable model

#### 5.5 VAE — recurrent variational autoencoder

- **Family:** latent-variable · recurrent · trained from scratch.
- **Inspired by:** VAE (Kingma & Welling, 2014), the sentence VAE (Bowman et al., 2016) and Magenta's
  **MusicVAE** (Roberts et al., 2018).
- **Architecture:** a bidirectional 3-layer GRU encoder compresses a segment into a 64-d latent `z`,
  and a 3-layer GRU decoder rebuilds it, with `z` fed in at every step.
- **Training:** reconstruction + KL divergence (the ELBO), 50 epochs (3.2 h). The loss fell steadily,
  which hid the collapse described below.
- **Inference:** 0.13 s per 600 events, among the cheapest: only the GRU decoder runs. It is greedy,
  but each call draws a new random `z`, so each run differs.
- **Result:** the latent **collapsed**: KL is 1.07 nats across 64 dimensions, and `z = 0` scores the
  same. It acts as a second RNN (96.2% seen accuracy), and `z` only nudges its greedy choices.
- **Takeaway:** a strong decoder learns to ignore the latent, so its variety is a side effect, not a
  musical control. KL annealing or word dropout are the standard fixes.

### Adversarial model

#### 5.6 GAN — Transformer generator + Transformer discriminator

- **Family:** adversarial · attention · **the only model that starts from trained weights.**
- **Inspired by:** GAN (Goodfellow et al., 2014), **SeqGAN** (Yu et al., 2017), **ScratchGAN**
  (de Masson d'Autume et al., 2019), **LeakGAN** (Guo et al., 2018) and Transformer-GAN (Muhamed, 2021).
- **Architecture:** a generator and a discriminator, both copies of the trained Transformer. The
  discriminator's head scores every event as real or generated.
- **Training:** 600 iterations (35 min). Each runs a discriminator step, a REINFORCE step on per-event
  rewards and a cross-entropy step on real data. The discriminator learns 10× slower.
- **Inference:** 6.9 s per 600 events: it uses the Transformer's sampling `predict`, so each run
  differs. Its key-value cache speeds up only the training rollouts.
- **Result:** sampled raw, it has ~20% fewer stuck notes than a plain fine-tune of the same length, and
  more realistic variety. By ear it is comparable to the Transformer.
- **Takeaway:** pure adversarial training failed. On discrete sequences a GAN needs a strong starting
  point, as in the published work.

### Reinforcement-learning model

#### 5.7 A2C — advantage actor-critic

- **Family:** reinforcement learning · recurrent · trained from scratch.
- **Inspired by:** **A3C / A2C** (Mnih et al., 2016), here set up as a contextual bandit: each step
  is scored on its own (γ = 0).
- **Architecture:** a 3-layer GRU shared by an *actor* (a policy over the 391 events) and a *critic*
  (a value estimate).
- **Training:** pure RL, 30 epochs (67 min): reward 1 only when the sampled event matches the real
  next one, with the critic as a baseline. An entropy bonus of 0.05 prevents collapse.
- **Inference:** 0.12 s per 600 events, among the cheapest: only the actor's GRU runs. It is greedy,
  so a primer always gives the same piece.
- **Result:** next-event accuracy rose from 0.5% to ~75% purely from reward, and `evaluate_net`
  reached 17.2%, on par with the Transformer.
- **Takeaway:** a right/wrong reward can train a language model, less efficiently than MLE. Its critic
  predicts a fixed reward rule, while the GAN's discriminator competes.

### Diffusion model

#### 5.8 Diffusion — masked discrete diffusion

- **Family:** diffusion · bidirectional attention · trained from scratch · **not left to right.**
- **Inspired by:** DDPM (Ho et al., 2020), adapted to tokens by **D3PM** (2021), **MDLM** (Sahoo et al.,
  2024) and **MaskGIT** (2022), with **Mask-Predict** / **ReMDM** for the correction rounds.
- **Architecture:** the Transformer's decoder code without the causal mask, plus a `[MASK]` token that
  also gives infilling. 8 layers, d_model 512, feed-forward 1024; 17.0 M parameters.
- **Training:** hide a random fraction `t` of each segment and predict it (cross-entropy weighted by
  `1/t`). AdamW 3e-4, 100 epochs (3.9 h). At the Transformer's size it was under-trained.
- **Inference:** 14.3 s per 600 events, the most expensive: 400 full passes over the piece (100
  reveal steps + 300 correction rounds). It samples, so each run differs.
- **Result:** `evaluate_net` 17.1%, level with the Transformer. Correction rounds halve broken note
  pairs, but ~9% of notes still drop. By ear it is below the Transformer.
- **Takeaway:** events carry state (a running clock, the keys held down) that a fill-in-anywhere model
  must guess early. Block diffusion (BD3-LM) would write small blocks left to right.

---

## 6. Experiment: LLM — composing with a frontier LLM, no training

- **Family:** autoregressive · attention · pretrained elsewhere, not trained on this dataset at all.
- **Inspired by:** in-context learning (GPT-3, Brown et al., 2020) and "LLMs as general pattern
  machines" (Mirchandani et al., 2023).
- **Architecture:** Claude Opus 5.5 through the Anthropic API, at effort `high`. The prompt holds a
  legend of the event IDs, 100 example excerpts (cached) and the same 32-event primer.
- **Training:** none. The examples stand in for training, plus a request to vary the dynamics, develop
  ideas and keep the primer's tempo. The primer's own excerpt is left out, so it can't be copied.
- **Inference:** one API call per piece, ≈ $0.4–0.8, and each run differs. It isn't served in the
  apps, because every generation costs money.
- **Result:** exact length and almost no broken note pairs, better bookkeeping than any trained model.
  None of its melodic patterns appear in the dataset. By ear, it sounds convincing.
- **Takeaway:** its skill comes from pretraining, since with no examples it still wrote a polished rag.
  Naming the genre pulled it toward a stereotype; the examples steered it to this dataset's feel.

---

## 7. Key takeaways

1. **Memorization vs generalization.** The RNN, CNN and VAE score highest on seen pieces but drop
   sharply on transposed ones; the smaller, regularized Transformer generalizes best. A held-out split
   would make this visible in the notebook itself.
2. **A training setting can masquerade as a model problem.** Label smoothing, meant as a
   regularizer, is what made the Transformer depend on top-40 decoding to sound right.
3. **GANs on discrete sequences need a strong starting point.** Pure adversarial training did not
   work; a pretrained generator with a pretrained, slowed-down discriminator produced a real (if
   modest) improvement beyond plain MLE.
4. **Measure what you hear.** Several defects — hanging notes, broken durations, an unused latent,
   diffusion's unpaired note-offs — were invisible to accuracy and found only by checking decoded
   music against real statistics.
5. **The data representation favours some model families.** These events carry state: a running
   clock and the keys held down. That suits models that write left to right. Diffusion reached
   comparable accuracy only with 5× the parameters, twice the training and correction rounds at
   generation time, and it still breaks more note pairs.
6. **Pretraining can stand in for training.** A frontier LLM, given only a legend and a primer,
   composes convincingly, with better note bookkeeping than any model trained here. Examples then
   serve to match a specific dataset's style. Naming the genre, by contrast, pulls it toward a textbook
   stereotype.
7. **Known loose ends:**
   - no held-out test set;
   - the CNN's late-training degradation;
   - the VAE's posterior collapse;
   - diffusion's remaining note-pairing errors (block diffusion is the next step), and the unmeasured
     effect of its correction rounds on variety;
   - only one piece per setting in the LLM experiment.

# Music Generation — Project Report

A from-scratch deep-learning project: eight models from five families, each trained to compose
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
| Transposed copies | A key-normalized copy of 428 pieces (transposed to C major / A minor with `music21`): training data **only for GPT-2**, and the "unseen" test set for all the others (§5) |

There is **no held-out set of new pieces**: models train and evaluate on the same pieces, and the
transposed copies stand in for unseen music (see §4).

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
  `huggingface_transformers.ipynb` (GPT-2).
- **Hardware:** RNN, CNN and VAE trained on a remote university machine (configuration not recorded),
  GPT-2 on Google Colab's free **T4** GPU, and the Transformer, GAN, A2C and Diffusion on an **Apple M5
  Pro** GPU (MPS). Training times aren't comparable across machines.
- **Shared helpers:** `train_net` (Adam, gradient clipping), `evaluate_net`, `generate_output`
  (tokens → `.midi`), `save_checkpoint` / `load_checkpoint`.
- **Serving:** a FastAPI service (`app/`, API-key protected) and a Streamlit front end (`main.py`),
  packaged with Docker; GPT-2 is published on the Hugging Face Hub.

## 4. How the models are measured

| Measure | What it tells you | Caveat |
|---|---|---|
| `evaluate_net` accuracy | Feeds 60 real events, then lets the model continue with its own decoding, and scores position by position, over 5 batches (160 segments) | Scored on training data; ~10 points are free (the 59 primer events count as correct); once a continuation diverges from the original piece, matches drop to chance. A random network scores ≈11%, a good one ≈16–17%. |
| Teacher-forced accuracy | How often the model's top guess for the next event is right, given the real history | Measured on training pieces **and** on music21-transposed copies the model never saw — the gap reveals memorization |
| Free-running statistics | Pitch/timing distributions, notes per second, note-on/off pairing, note durations, **notes sounding at once** | Real ragtime: ~12 notes/s, median note 0.21 s, ~3 notes sounding at once |
| Inference cost | CPU time, wall time and peak RAM to generate one 600-event piece on CPU, as both apps run | Measured on an Apple M5 Pro (6 threads); hosted CPUs are slower and have fewer cores |

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
| RNN | Autoregressive | Recurrent (GRU) | scratch | 4.6 M | 50 epochs, 4.1 h (university) | 23.4% | 96.3% → 47.7% |
| CNN | Autoregressive | Convolutional (dilated) | scratch | 15.1 M | 50 epochs, 2.1 h (university) | 18.2% | 86.3% → 43.3% |
| Transformer | Autoregressive | Attention (Transformer decoder) | scratch | 3.4 M | 50 epochs, 80 min (M5) | 17.1% | 82.4% → 62.3% |
| GPT-2 | Autoregressive | Attention (Hugging Face GPT-2) | scratch | 86.3 M | 20 epochs on 2× data, 3.0 h (T4) | — | 80.9% → 78.7%† |
| VAE | Latent-variable | Recurrent (GRU encoder + decoder) | scratch | 16.8 M | 50 epochs, 3.2 h (university) | 23.6% | 96.1% → 47.5% |
| GAN | Adversarial | Attention (Transformer decoder + BERT encoder) | **trained Transformer + Diffusion** | 3.4 M + 3.5 M | 10 epochs, 73 min (M5) | 16.9% | 82.4% → 61.6% |
| A2C | Reinforcement learning | Recurrent (GRU) | scratch | 4.6 M | 50 epochs, 2.0 h (M5) | 17.2% | 78.2% → 49.9% |
| Diffusion | Diffusion | Attention (BERT encoder) | scratch | 3.6 M | 400 epochs, 8.4 h (M5) | 15.4% | 59.1% → 46.8%* |

- **Only the GAN starts from trained weights:** its generator is the trained Transformer and its
  discriminator the trained Diffusion model's encoder. GPT-2 borrows only the architecture, and
  Diffusion builds its encoder from the Transformer's attention classes, not its weights.
- **"Unseen"** means the music21-transposed copies: the same style, but event sequences the model
  never trained on (0.2% of their 16-event phrases occur in the training pieces). music21 also
  re-quantizes timing, so rhythm details differ too. All models are scored on the same 256 seen and
  256 unseen segments, using the served half-precision weights.
- **Accuracy** is teacher-forced next-event accuracy. \*Diffusion doesn't predict the next event, so
  its number is hidden-event accuracy averaged over hiding 5–95% of a piece. †GPT-2's number is
  next-token accuracy on its text tokens, not per event, and it trained on the transposed copies, so
  they aren't unseen for it. Neither is directly comparable with the others, and GPT-2's epochs (over
  BPE text in 256-token windows) aren't comparable with the event models' epochs either.

### Inference cost

The cost of generating one 600-event piece on CPU (Apple M5 Pro, 6 threads), which is how both apps
run, most expensive first by **CPU time**: the total work across all cores. Wall time is what a user
waits. Peak RAM is what generating adds to an idle Python process with PyTorch loaded: weights,
activations and caches, plus about 70 MB of PyTorch workspace. Each figure averages three runs.

| Rank | Model | CPU time | Wall time | Peak RAM | Same primer → same piece? | Why |
|---|---|---|---|---|---|---|
| 1 | CNN | 13.8 s | 9.3 s | 373 MB | Yes (greedy) | Recomputes the whole convolution stack over the piece so far, for every new event |
| 2 | Diffusion | 5.5 s | 1.8 s | 56 MB | No (samples) | 100 full passes over the piece, one per reveal step, spread over about 3 cores |
| 3 | GPT-2 | 4.5 s | 3.4 s | 509 MB | No (samples) | The largest (86 M parameters); it caches keys and values within each 256-token window |
| 4 | Transformer | 1.1 s | 0.9 s | 28 MB | No (samples) | d2l's key-value cache: each new event attends to the stored states of the earlier ones |
| 4 | GAN | 1.1 s | 0.9 s | 35 MB | No (samples) | The Transformer's `predict`; the discriminator isn't used |
| 6 | VAE | 0.13 s | 0.14 s | 168 MB | No (greedy, but a random `z` each time) | A GRU decoder with a fixed-size state: the same work for every event |
| 6 | RNN | 0.13 s | 0.13 s | 98 MB | Yes (greedy) | Same as the VAE |
| 6 | A2C | 0.12 s | 0.13 s | 98 MB | Yes (greedy) | Same as the VAE |

- **Greedy models always write the same piece for a primer.** RNN, CNN and A2C always pick the most
  likely event. The VAE does too, but its fresh random `z` makes each run differ.
- **Caching decides the cost.** The GRUs carry a fixed-size state and the Transformer caches its keys
  and values, so each new event is cheap; the CNN recomputes its stack for every event, and Diffusion
  re-reads the whole piece at each of its 100 steps.
- **Wall time hides parallel work.** Diffusion finishes before GPT-2 only by keeping about 3 cores
  busy. RAM ranks differently again: GPT-2 and the CNN need the most, the Transformer the least.
- **Hosted CPUs are slower and have fewer cores** (Streamlit Cloud, Cloud Run), so every time grows
  there, and the models that use several cores grow most.

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
- **Inference:** 0.13 s per 600 events, among the cheapest. It is greedy (it always picks the most
  likely event), so a primer always gives the same piece.
- **Result:** a top `evaluate_net` score (23.4%, with the VAE) and 96.3% seen accuracy, but 47.7%
  unseen: it **largely memorized** the 446 pieces, which also inflates its `evaluate_net` score.
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
- **Inference:** 9.3 s per 600 events, the most expensive: it recomputes the convolution stack
  for every new event. It is greedy, so a primer always gives the same piece.
- **Result:** 86.3% seen → 43.3% unseen: it memorized too, and generalizes worst of the models trained
  on likelihood. The likely cause is the constant 1e-3 learning rate, which never decays.
- **Takeaway:** a learning-rate schedule, or keeping the best checkpoint rather than the last, would
  likely help most.

#### 5.3 Transformer — decoder-only self-attention

- **Family:** autoregressive · attention · trained from scratch.
- **Inspired by:** "Attention Is All You Need" (Vaswani et al., 2017), decoder-only as in GPT, built
  as in **Dive into Deep Learning** (d2l.ai §11), with Music Transformer's event vocabulary.
- **Architecture:** d2l's decoder blocks: 6 layers, d_model 256, 8 heads, feed-forward 512, post-LN,
  causal mask, sinusoidal positions, key-value cache; 3.4 M parameters.
- **Training:** d2l's recipe: masked cross-entropy, Adam at a fixed 1e-3, gradient clipping 1,
  dropout 0.1, 50 epochs (80 min). Loss 6.2 → 0.66.
- **Inference:** 0.9 s per 600 events, as each new event attends to cached states. It samples from
  its own probabilities, with no top-k or temperature, so each run differs.
- **Result:** the best generalization of the event models (82.4% seen → 62.3% unseen). d2l's greedy
  decoding looped on a few notes; plain sampling matches real music's variety.
- **Takeaway:** this is the reference model. Its weights become the GAN's generator, and its attention
  classes build the diffusion model.

#### 5.4 GPT-2 — Hugging Face causal language model

- **Family:** autoregressive · attention · trained from scratch (only the architecture is borrowed).
- **Inspired by:** **GPT-2** (Radford et al., 2019), following the Hugging Face course chapter
  "Training a causal language model from scratch".
- **Architecture:** `GPT2LMHeadModel`, 12 layers, 768-d, 12 heads, 86.3 M parameters. Events are
  written as text numbers, with a retrained BPE tokenizer (vocabulary 587); context 256 tokens.
- **Training:** original + transposed pieces (874, each piece in two keys), 20 epochs (3.0 h),
  effective batch 256, learning rate 1e-4 with cosine decay, fp16, on Colab's free T4.
- **Inference:** 3.4 s per 600 events, in a sliding window of 128 tokens of context plus 128 new ones,
  so it never exceeds its 256-token training window. It samples, so each run differs.
- **Result:** 80.9% next-token accuracy (on text tokens, not events), and 78.7% on the transposed
  copies, which it trained on. Its "validation" set is training data too, so it has no held-out score.
- **Takeaway:** the largest model, and the only one trained on augmented data, is the hardest to
  compare. Event-level metrics and a true held-out set would show what its size buys.

### Latent-variable model

#### 5.5 VAE — recurrent variational autoencoder

- **Family:** latent-variable · recurrent · trained from scratch.
- **Inspired by:** VAE (Kingma & Welling, 2014), the sentence VAE (Bowman et al., 2016) and Magenta's
  **MusicVAE** (Roberts et al., 2018).
- **Architecture:** a bidirectional 3-layer GRU encoder compresses a segment into a 64-d latent `z`,
  and a 3-layer GRU decoder rebuilds it, with `z` fed in at every step.
- **Training:** reconstruction + KL divergence (the ELBO), 50 epochs (3.2 h). The loss fell steadily,
  which hid the collapse described below.
- **Inference:** 0.14 s per 600 events, among the cheapest: only the GRU decoder runs. It is greedy,
  but each call draws a new random `z`, so each run differs.
- **Result:** the latent **collapsed**: KL is 1.07 nats across 64 dimensions, and `z = 0` scores the
  same. It acts as a second RNN (96.1% seen → 47.5% unseen), and `z` only nudges its choices.
- **Takeaway:** a strong decoder learns to ignore the latent, so its variety is a side effect, not a
  musical control. KL annealing or word dropout are the standard fixes.

### Adversarial model

#### 5.6 GAN — Transformer generator + BERT discriminator

- **Family:** adversarial · attention · **the only model that starts from trained weights.**
- **Inspired by:** GAN (Goodfellow et al., 2014) with d2l.ai's GAN training loop, and **SeqGAN**
  (Yu et al., 2017) for training the generator by REINFORCE.
- **Architecture:** the generator is the trained Transformer; the discriminator is the trained
  diffusion model's BERT encoder with d2l's classifier head, one real-or-generated logit per window.
- **Training:** d2l's `update_D` / `update_G`, Adam (D 1e-4, G 1e-5), 10 epochs (73 min) on
  256-event windows. G's reward is D's verdict on each window minus the batch mean; no MLE step.
- **Inference:** 0.9 s per 600 events: the generator is a Transformer, sampled the same way, so each
  run differs. The discriminator is only used in training.
- **Result:** G moved toward what D can see: its note density now matches real music (11.6 vs 13.9
  notes/s; real 11.8), at a slightly lower unseen accuracy and twice the unpaired note-offs.
- **Takeaway:** one verdict per window is a weak teacher next to MLE, so a vanilla GAN re-tunes
  surface statistics. Windows that start mid-piece also hide unpaired note-offs from D.

### Reinforcement-learning model

#### 5.7 A2C — advantage actor-critic

- **Family:** reinforcement learning · recurrent · trained from scratch.
- **Inspired by:** **A3C / A2C** (Mnih et al., 2016), here set up as a contextual bandit: each step
  is scored on its own (γ = 0).
- **Architecture:** a 3-layer GRU shared by an *actor* (a policy over the 391 events) and a *critic*
  (a value estimate).
- **Training:** pure RL, 50 epochs (2.0 h): reward 1 only when the sampled event matches the real
  next one, with the critic as a baseline. An entropy bonus of 0.05 prevents collapse.
- **Inference:** 0.13 s per 600 events, among the cheapest: only the actor's GRU runs. It is greedy,
  so a primer always gives the same piece.
- **Result:** next-event accuracy rose from 0.5% to 78.2% (49.9% unseen) purely from reward, and
  `evaluate_net` reached 17.2%, on par with the Transformer.
- **Takeaway:** a right/wrong reward can train a language model, less efficiently than MLE. Its critic
  predicts a fixed reward rule, while the GAN's discriminator competes.

### Diffusion model

#### 5.8 Diffusion — masked discrete diffusion

- **Family:** diffusion · bidirectional attention · trained from scratch · **not left to right.**
- **Inspired by:** **MDLM** (Sahoo et al., 2024) for the loss and sampler, on d2l.ai's **BERT**
  encoder (§15.8), with `[MASK]` as the noise as in D3PM (Austin et al., 2021).
- **Architecture:** d2l's encoder blocks, learned positions and masked-LM head at the Transformer's
  size (6 layers, d_model 256); 3.6 M parameters. The `[MASK]` token also gives infilling.
- **Training:** hide a random fraction `t` of each segment, spread evenly over the batch, and predict
  it (cross-entropy weighted by `1/t`). Adam 1e-3, 400 epochs (8.4 h): 8× the Transformer's.
- **Inference:** 1.8 s per 600 events on about 3 cores: 100 passes over the whole piece, each
  revealing about 1% of it, sampled from the model's probabilities. Each run differs.
- **Result:** it plateaued from about epoch 200 (59.1% seen → 46.8% unseen hidden-event accuracy).
  ~23% of its notes drop as unpaired note-ons/offs; by ear, 70–80% of the Transformer.
- **Takeaway:** events carry state (a running clock, the keys held down) that a fill-in-anywhere model
  must guess early. Block diffusion (BD3-LM) would write small blocks left to right.

---

## 6. Experiment: LLM — composing with a frontier LLM, no training

- **Family:** autoregressive · attention · pretrained elsewhere, not trained on this dataset at all.
- **Inspired by:** in-context learning (GPT-3, Brown et al., 2020) and "LLMs as general pattern
  machines" (Mirchandani et al., 2023).
- **Architecture:** Claude Opus 5.5 through the Anthropic API, at effort `high`. The prompt holds a
  legend of the event IDs, 100 example excerpts (cached) and a 32-event primer, like every model.
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
2. **Decoding is part of the model.** d2l's greedy decoding made the Transformer loop on the same
   few notes; sampling from its own probabilities, with no tricks, matched real music's variety.
3. **A GAN refines, it doesn't teach.** On discrete sequences it needs a trained generator to start
   from, and even then one verdict per window mostly re-tunes surface statistics such as note
   density.
4. **Measure what you hear.** Several defects — greedy decoding's loops, an unused latent, the GAN's
   extra note-offs, diffusion's unpaired note-offs — were invisible to accuracy and found only by
   checking decoded music against real statistics.
5. **The data representation favours some model families.** These events carry state: a running
   clock and the keys held down. That suits models that write left to right. At the Transformer's
   size, diffusion plateaued well below it even with 8× the training, and breaks far more note pairs.
6. **Pretraining can stand in for training.** A frontier LLM, given only a legend and a primer,
   composes convincingly, with better note bookkeeping than any model trained here. Examples then
   serve to match a specific dataset's style. Naming the genre, by contrast, pulls it toward a textbook
   stereotype.
7. **Known loose ends:**
   - no held-out test set;
   - the CNN's late-training degradation;
   - the VAE's posterior collapse;
   - diffusion's note-pairing errors (block diffusion is the next step);
   - only one piece per setting in the LLM experiment.

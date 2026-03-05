# CQG — Migration & Optimization Change Log

This document records every change made to the original CQG codebase to make it
run on **Python 3.13**, **PyTorch (CPU-only)**, with **no fastNLP dependency**.

---

## 1. Dependency Migration — Remove fastNLP 0.6.0

**Problem:** fastNLP 0.6.0 is incompatible with Python 3.12+ and unmaintained.  
All project code that imported from fastNLP had to be rewritten with inline
equivalents or standard-library alternatives.

### `main.py`
- Removed all fastNLP imports (`DataSet`, `Trainer`, `BertWordPieceEncoder`,
  `CNNText`, `GloVe`, etc.).
- Removed unused `FlaxGPTNeoForCausalLM` import.

### `utils/util.py`
- `from transformers import AdamW` → `from torch.optim import AdamW`  
  (`transformers.AdamW` was deprecated and later removed).

### `model/Generator.py`
- Replaced `from fastNLP import ...` with a local import:
  `from model.Decoder import Seq2SeqDecoder, State, TransformerState`
- Inlined `_get_model_device` (4-line helper) instead of importing it from fastNLP.

### `model/Decoder.py` — full rewrite of the class hierarchy
The entire file was reconstructed to replace the following fastNLP classes with
self-contained inline implementations:

| Removed (fastNLP)         | Replaced with                          |
|---------------------------|----------------------------------------|
| `State`                   | Inlined `State` base class             |
| `LSTMState`               | Inlined `LSTMState`                    |
| `TransformerState`        | Inlined `TransformerState`             |
| `MultiHeadAttention`      | Inlined `MultiHeadAttention`           |
| `AttentionLayer`          | Inlined `AttentionLayer`               |
| `StaticEmbedding`         | Stub `StaticEmbedding` + `get_embeddings()` |

---

## 2. Python 3.13 / PyTorch API Compatibility

### Deprecated `torch.cuda.amp` namespace (`main.py`)
```python
# Before
torch.cuda.amp.GradScaler(...)
torch.cuda.amp.autocast(...)

# After
torch.amp.GradScaler('cuda', ...)
torch.amp.autocast('cuda', ...)
```

### Private collate import (`dataset.py`)
```python
# Before
from torch.utils.data._utils.collate import default_collate

# After
from torch.utils.data import default_collate
```

### `torch.load` without `weights_only` (`utils/util.py`, `main.py`)
```python
# Before
torch.load(path)

# After
torch.load(path, weights_only=True)
```
Suppresses the FutureWarning and prevents arbitrary code execution from
untrusted checkpoint files.

### `shift_tokens_right` removed from transformers (`main.py`)
`transformers.models.mbart.modeling_mbart.shift_tokens_right` was removed in
newer versions. Inlined as a local function:
```python
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    shifted[:, 0] = pad_token_id
    return shifted
```

### `rouge` package → `rouge-score` (`main.py`, `requirements.txt`)
The `rouge` PyPI package is unmaintained and fails to install on Python 3.12+.
```python
# Before
from rouge import Rouge
rouge = Rouge()
score = rouge.get_scores([pred], [gold])[0]['rouge-l']['f']

# After
from rouge_score import rouge_scorer
_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
score = _rouge_scorer.score(gold, pred)['rougeL'].fmeasure
```

### Decoder bug fixes (`model/Decoder.py`)
- `raise NotImplemented` → `raise NotImplementedError` (NotImplemented is a
  constant, not an exception).
- `.byte()` → `.bool()` (`.byte()` for boolean masks is deprecated).

---

## 3. CPU-Only / Hardware Optimizations

The target machine has **no NVIDIA GPU** (Intel Iris Xe only, 16 GB RAM).
Defaults that assumed CUDA caused immediate crashes.

### Auto-detect compute device (`main.py`)
```python
# Before
parser.add_argument('--device', default='cuda:0')

# After
parser.add_argument('--device',
    default='cuda:0' if torch.cuda.is_available() else 'cpu')
```

### Reduce default batch size (`main.py`)
```python
# Before
parser.add_argument('--batch_size', default=32)

# After
parser.add_argument('--batch_size', default=8)
```
Prevents out-of-memory errors when running on CPU with 16 GB RAM.

### Guard all `nn.DataParallel` calls (`main.py`)
```python
# Before — crashes when no GPU is present
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# After
device_ids = list(range(torch.cuda.device_count()))
if config.gpus and torch.cuda.is_available() and len(device_ids) > 0:
    model = nn.DataParallel(model, device_ids=device_ids)
```

### Safe `.generate()` call (`main.py`)
```python
# Before — AttributeError when model is not wrapped in DataParallel
model.module.generate(...)

# After
_model = model.module if isinstance(model, nn.DataParallel) else model
_model.generate(...)
```

### Reduce DataLoader workers (`utils/config.py`)
```python
# Before
self.num_workers = 8

# After
self.num_workers = 0 if not torch.cuda.is_available() else 4
```
`num_workers > 0` on Windows CPU-only causes multiprocessing overhead and
can deadlock; `0` uses the main process directly.

---

## 4. Memory Optimization — Flag Matrix dtype

### `dataset.py`
```python
# Before — stores 0/1/2 values as int64 (8 bytes each)
np.asarray(flag)

# After — stores 0/1/2 values as int8 (1 byte each)
np.asarray(flag, dtype=np.int8)
```
The flag matrix encodes only three values (0, 1, 2), so int8 is sufficient.
This reduces flag tensor memory by **8×**, which matters for the large
`train.json` (910 MB, ~90 k samples).

### Long cast before embedding lookup (`model/Decoder.py`)
Because `nn.Embedding` requires `torch.int64` (Long) indices, the int8 flag
tensor must be cast before use:
```python
# Before
rel_k = self.k_flag(flag)

# After
rel_k = self.k_flag(flag.long())
```

---

## 5. Inference Shape Fix — FLAGMultiHeadAttention

**Problem:** During validation/test, each sample's flag is constructed as a
single row `[[0, 0, ..., 0]]` (shape `(batch, 1, k_len)`), but the decoder
attention expects the flag to have `q_len` rows at each generation step.

**Fix** — pad or trim the flag to match the current `q_len` before the
embedding lookup (`model/Decoder.py`, `FLAGMultiHeadAttention.forward()`):
```python
batch_size, q_len, d_model = query.size()
k_len = k.size(1)

if flag.size(1) < q_len:
    pad = flag.new_zeros(batch_size, q_len - flag.size(1), flag.size(2))
    flag = torch.cat([flag, pad], dim=1)
elif flag.size(1) > q_len:
    flag = flag[:, :q_len, :]

rel_k = self.k_flag(flag.long())
```

---

## 6. Inference KV-Cache Fix — Self-Attention Accumulation Bug

**Problem:** The generator feeds the **full accumulated token sequence** at
every generation step (e.g. step 2 sends tokens `[BOS, t1]`,
step 3 sends `[BOS, t1, t2]`, …). The self-attention KV-cache was still
accumulating keys from previous steps on top of the recomputed keys,
causing `k_len` to grow as triangular numbers (1 → 3 → 6 → 10 …) instead
of linearly. The causal mask, sized `(q_len × q_len)`, then mismatches `k_len`,
producing:

```
RuntimeError: The size of tensor a (2) must match the size of tensor b (3)
at non-singleton dimension 3
```

**Fix** (`model/Decoder.py`, `MultiHeadAttention.forward()`):  
Disable the **decoder self-attention KV-cache** entirely. Since the generator
already replays the full sequence each step, there is nothing useful to cache
for self-attention. The **encoder cross-attention cache** is kept — encoder
output is constant across steps, so caching it saves recomputing projections.

```python
if isinstance(state, TransformerState):
    if qkv_same:
        # Self-attention: full sequence is re-fed every step,
        # so do NOT accumulate prev_k (would give k_len > q_len).
        pass
    else:
        k = state.encoder_key[self.layer_idx]   # use cached encoder KV
        v = state.encoder_value[self.layer_idx]

if k is None:
    k = self.k_proj(key)
    v = self.v_proj(value)

if isinstance(state, TransformerState):
    if not qkv_same:
        state.encoder_key[self.layer_idx] = k   # cache encoder KV only
        state.encoder_value[self.layer_idx] = v
```

---

## 7. NLTK Data Resources

The BLEU evaluation code requires two NLTK corpora that are not bundled by
default:

```python
import nltk
nltk.download('punkt_tab')   # sentence tokenizer (newer NLTK versions)
nltk.download('wordnet')     # used by METEOR / corpus readers
nltk.download('omw-1.4')     # Open Multilingual WordNet (wordnet companion)
```

---

## 8. `requirements.txt` — Final State

```
torch
transformers
numpy
tqdm
rouge-score
nltk
```

`fastNLP` has been removed entirely.

---

## Summary Table

| # | File | Change | Reason |
|---|------|--------|--------|
| 1 | `main.py` | Remove fastNLP imports | Python 3.12+ incompatibility |
| 2 | `main.py` | Inline `shift_tokens_right` | Removed from transformers |
| 3 | `main.py` | `torch.cuda.amp` → `torch.amp` | Deprecated API |
| 4 | `main.py` | `--device` auto-detect | No NVIDIA GPU |
| 5 | `main.py` | `--batch_size` 32 → 8 | RAM limit on CPU |
| 6 | `main.py` | Guard `nn.DataParallel` | Crashes without GPU |
| 7 | `main.py` | Safe `.generate()` call | AttributeError without DataParallel |
| 8 | `main.py` | `rouge` → `rouge-score` | Unmaintained, install failure |
| 9 | `utils/util.py` | `transformers.AdamW` → `torch.optim.AdamW` | Removed from transformers |
| 10 | `utils/util.py` | `torch.load(..., weights_only=True)` | FutureWarning + security |
| 11 | `utils/config.py` | `num_workers` 8 → 0 on CPU | Windows CPU deadlock |
| 12 | `dataset.py` | Private `_utils.collate` → public import | Private API removed |
| 13 | `dataset.py` | Flag dtype int64 → int8 | 8× memory reduction |
| 14 | `model/Decoder.py` | Full rewrite — inline all fastNLP classes | fastNLP incompatibility |
| 15 | `model/Decoder.py` | `raise NotImplemented` → `raise NotImplementedError` | Bug fix |
| 16 | `model/Decoder.py` | `.byte()` → `.bool()` | Deprecated API |
| 17 | `model/Decoder.py` | `flag.long()` cast before embedding | int8 not accepted by `nn.Embedding` |
| 18 | `model/Decoder.py` | Flag padding in `FLAGMultiHeadAttention` | Shape mismatch during inference |
| 19 | `model/Decoder.py` | Disable self-attention KV-cache accumulation | Triangular k_len growth crash |
| 20 | `model/Generator.py` | Inline `_get_model_device`, local imports | fastNLP removal |
| 21 | `requirements.txt` | Remove fastNLP, add rouge-score | Dependency cleanup |
| 22 | NLTK data | Download `punkt_tab`, `wordnet`, `omw-1.4` | Missing BLEU/eval resources |

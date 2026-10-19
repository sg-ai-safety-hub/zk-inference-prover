"""B10 step 1: the real greedy continuation the sequence's witness is extracted from.

The brief's scope decision is that the proof is of the 512 **decoded** tokens
given a committed KV prefix of C - S keys, and that decoding is **greedy**.  So
the rectangle the 32 layers are extracted from has to be a real decode, not the
corpus's next 512 tokens: this generates the continuation once, greedily, under
the spec that is proved, and writes `prompt || continuation` for
`harness.witness.extract --ids`.

B8 step 1 established that decode is bit-identical to prefill in this model, so
the extractor still prefills the concatenation and the last chunk of `--new`
tokens is the decoded rectangle.  The prompt is the eval corpus's first
`--prefix` tokens -- a real 16,384-token document, which is what a C=16K
benchmark condition means -- and `--prefix` is the manifest's `c`.

    uv run python -m harness.witness.gen_ids --prefix 16384 --new 512 --spec y-delay-vw12 \
        --out ids_c16384.npy

`--help` loads the module and nothing else, so it is the check that the imports
below resolve on a host with no weights and no card.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from .. import provenance
from .common import MODEL, SPECS, setup, tokens
from .model import Cache, Model
from .ppl import attention_scales

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=int, default=16384,
                   help="committed KV prefix, the manifest's `c` (whole prefill chunks)")
    p.add_argument("--new", type=int, default=512, help="tokens decoded greedily")
    # The corpus opens with front matter (a preface and a list of illustrations),
    # and a 4,096-token prompt taken from token 0 ends inside that list -- the
    # greedy continuation is then a repeating index line, which is a real decode
    # but a degenerate benchmark.  `--offset` moves the prompt window; the C=4K
    # row uses 12288, so its prompt is the *last* 4,096 tokens of the C=16K row's
    # and the two contexts read the same prose.
    p.add_argument("--offset", type=int, default=0, help="first corpus token of the prompt")
    p.add_argument("--chunk", type=int, default=512, help="prefill chunk")
    p.add_argument("--spec", default="b10-design")
    p.add_argument("--out", required=True, help="the ids (.npy); a `.json` note beside it")
    a = p.parse_args()
    assert a.prefix > 0, "prefix must be positive"
    assert a.prefix % a.chunk == 0, "prefix must be a whole number of prefill chunks"
    setup()
    spec = SPECS[a.spec]
    m = Model(MODEL, spec, attn_scales=attention_scales(spec))
    ev, _ = tokens()
    need = a.offset + a.prefix
    assert len(ev) >= need, \
        f"the corpus has {len(ev)} tokens, need {need}; ZK_EVAL_TOKENS widens the slice"
    prompt = ev[a.offset:need]

    t0 = time.time()
    cache = Cache(m.L)
    logits = None
    for i in range(0, a.prefix, a.chunk):
        logits, cache = m.forward(prompt[i:i + a.chunk], cache)
        print(f"  prefilled {i + a.chunk}/{a.prefix} ({time.time() - t0:.0f}s)", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    stop = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")) if i is not None}

    # Greedy, one token at a time, and the run does not stop at an end-of-text:
    # the statement is 512 decoded tokens, so a stop is recorded and passed
    # through rather than truncating the rectangle.
    out, stops = [], []
    for t in range(a.new):
        nxt = int(logits[-1].argmax().item())
        if nxt in stop:
            stops.append(t)
        out.append(nxt)
        logits, cache = m.forward(torch.tensor([nxt]), cache)
        if (t + 1) % 128 == 0:
            print(f"  decoded {t + 1}/{a.new} ({time.time() - t0:.0f}s)", flush=True)

    ids = np.concatenate([prompt.numpy().astype(np.int64), np.array(out, dtype=np.int64)])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.save(a.out, ids)
    saved = a.out if a.out.endswith(".npy") else a.out + ".npy"   # what numpy wrote
    # The extractor names the ids file it read by digest (`ids_from`), so the
    # decode that wrote it names the same digest: after a re-decode over the same
    # path, which of the two a dump was extracted from is a comparison and not a
    # reconstruction from timestamps.
    note = dict(model=MODEL, spec=a.spec, prefix=a.prefix, offset=a.offset,
                new=a.new, total=len(ids), ids_file=provenance.digest(saved),
                greedy=True, stop_ids=sorted(stop), stops_at=stops,
                prompt_tail=tok.decode(prompt[-64:].tolist()),
                continuation=tok.decode(out),
                wall_s=round(time.time() - t0, 1))
    json.dump(note, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(f"wrote {a.out}: {len(ids)} ids ({a.prefix} prompt + {a.new} greedy) "
          f"in {time.time() - t0:.0f}s; stops at {stops or 'none'}")
    print("--- continuation ---")
    print(note["continuation"][:1200])

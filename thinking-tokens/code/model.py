"""One small causal transformer, many ways to think before answering.

Layout of every sequence:  [pad .. BOS expr '='] [thoughts] [answer EOS]
The prompt is left padded so '=' sits at the same index for the whole batch, which
lets the thinking phase run batched on a KV cache that stays attached to autograd.

mode
  none    answer straight after '='
  um      K copies of one learned filler embedding ("um")
  disc    K discrete thinking tokens from a vt-word vocabulary (straight-through Gumbel)
  soft    K probability-weighted mixtures of that vocabulary's embeddings
  direct  K continuous thoughts, final hidden state fed straight back (Coconut-style)
  cont    K continuous thoughts through a dedicated linear head
loop = (lo, hi): which layers a thought passes through (0, L = whole model).
window = n: only the newest n thoughts stay in the KV cache.
slots = n: a fixed workspace of n vectors refined in parallel for K // n rounds.
ut = R: the layers [lo, hi) are applied R times at every position (looped transformer).
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as Fn

PAD = 0


@dataclass
class Cfg:
    vocab: int = 44
    d: int = 256
    L: int = 4
    heads: int = 4
    mode: str = "none"
    K: int = 0
    lo: int = 0
    hi: int = -1
    window: int = 0
    slots: int = 0
    vt: int = 1024
    ut: int = 1
    tau: float = 1.0
    max_pos: int = 160


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)

    def forward(self, x, rot, past, pvalid, tri):
        """past = None for the prompt itself, else [prompt_k, prompt_v, extra_k, extra_v] (extra may be None).
        The cached chunks are attended to in place: concatenating the prompt keys on every thought step
        would copy (and backprop through) the whole prompt cache K times."""
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rot(q), rot(k)
        if past is None:  # causal, never attend to padding (except yourself)
            mask = (tri & pvalid[:, None, :]) | torch.eye(T, dtype=torch.bool, device=x.device)
            a = Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        else:
            pk, pv, ek, ev = past
            s = [(q @ pk.transpose(-1, -2)).masked_fill(~pvalid[:, None, None, :], float("-inf"))]
            if ek is not None:
                s.append(q @ ek.transpose(-1, -2))
            sn = q @ k.transpose(-1, -2)
            s.append(sn if tri is None else sn.masked_fill(~tri, float("-inf")))
            att = (torch.cat(s, -1) * (d // self.h) ** -0.5).softmax(-1).to(v.dtype)
            n_p, n_e = pk.shape[2], 0 if ek is None else ek.shape[2]
            a = att[..., :n_p] @ pv + att[..., n_p + n_e :] @ v
            if n_e:
                a = a + att[..., n_p : n_p + n_e] @ ev
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        x = x + self.out(Fn.gelu(self.fc(self.ln2(x))))
        return x, (k, v)


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, L = cfg.d, cfg.L
        if cfg.hi < 0:
            cfg.hi = L
        self.emb = nn.Embedding(cfg.vocab, d)
        self.blocks = nn.ModuleList([Block(d, cfg.heads) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab, bias=False)
        # unrolled layer schedule; cfg.ut > 1 repeats the middle block at every position
        ut = cfg.ut if cfg.mode == "ut" else 1
        self.sched = list(range(cfg.lo)) + ut * list(range(cfg.lo, cfg.hi)) + list(range(cfg.hi, L))
        self.vlo, self.vhi = (0, len(self.sched)) if cfg.mode == "ut" else (cfg.lo, cfg.hi)
        self.apply(self._init)
        for b in self.blocks:
            for w in (b.proj.weight, b.out.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * L))
        if cfg.mode == "um":
            self.um = nn.Parameter(torch.randn(d) * 0.02)
        if cfg.mode in ("cont", "soft", "disc"):
            self.tnorm = nn.LayerNorm(d)
        if cfg.mode == "cont":
            self.tin = nn.Linear(d, d)
            nn.init.eye_(self.tin.weight)
            nn.init.zeros_(self.tin.bias)
        if cfg.mode in ("soft", "disc"):
            self.thead = nn.Linear(d, cfg.vt)
            self.temb = nn.Embedding(cfg.vt, d)
            nn.init.normal_(self.thead.weight, std=0.02)
            nn.init.normal_(self.temb.weight, std=0.02)
        if cfg.slots:
            self.slot = nn.Parameter(torch.randn(cfg.slots, d) * 0.02)
        dh = d // cfg.heads
        inv = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
        ang = torch.arange(cfg.max_pos).float()[:, None] * inv[None]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ plumbing
    def _rot(self, pos):
        cos, sin = self.cos[pos], self.sin[pos]

        def rot(x):
            c, s = cos.to(x.dtype), sin.to(x.dtype)
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

        return rot

    def run(self, x, pos, v0, v1, cache, pvalid, store=True, bidir=False):
        """Push x through virtual layers [v0, v1). cache[v] = [prompt_k, prompt_v, extra_k, extra_v] or None.
        store: True appends the new keys/values to the cache, a dict receives them, None drops them."""
        T = x.shape[1]
        rot = self._rot(pos)
        tri = None if bidir or T == 1 else torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        for v in range(v0, v1):
            past = cache[v]
            if past is None and tri is None:
                tri = torch.ones(T, T, dtype=torch.bool, device=x.device)
            x, (k, vv) = self.blocks[self.sched[v]](x, rot, past, pvalid, tri)
            if store is True:
                if past is None:
                    cache[v] = [k, vv, None, None]
                elif past[2] is None:
                    past[2], past[3] = k, vv
                else:
                    past[2], past[3] = torch.cat([past[2], k], 2), torch.cat([past[3], vv], 2)
            elif store is not None:
                store[v] = (k, vv)
        return x

    def thought(self, h, intervene=None):
        m = self.cfg.mode
        if m == "direct":
            z = self.ln_f(h)
        elif m == "cont":
            z = self.tin(self.tnorm(h))
        else:
            logits = self.thead(self.tnorm(h)).float()
            if m == "soft":
                p = logits.softmax(-1)
            elif self.training:
                p = Fn.gumbel_softmax(logits, tau=self.cfg.tau, hard=True)
            else:
                p = Fn.one_hot(logits.argmax(-1), logits.shape[-1]).float()
            z = (p @ self.temb.weight.float()).to(h.dtype)
        if intervene == "zero":
            z = torch.zeros_like(z)
        elif intervene == "shuffle":  # someone else's thought
            z = z.roll(1, 0)
        elif intervene == "noise":
            z = torch.randn_like(z) * z.std(-1, keepdim=True)
        return z

    def think(self, P, K=None, window=None, intervene=None, collect=None):
        """Prompt pass + thinking. Returns (final hidden state [B,1,d], cache, pvalid)."""
        cfg = self.cfg
        K = cfg.K if K is None else K
        window = cfg.window if window is None else window
        V = len(self.sched)
        vlo, vhi = self.vlo, self.vhi
        Pm = P.shape[1]
        pvalid = P != PAD
        cache = [None] * V
        pos = torch.arange(Pm, device=P.device)
        tap = self.run(self.emb(P), pos, 0, vhi, cache, pvalid)
        final = self.run(tap, pos, vhi, V, cache, pvalid)[:, -1:]
        tap = tap[:, -1:]
        if K == 0 or cfg.mode in ("none", "ut", "cot"):
            return final, cache, pvalid
        if cfg.mode == "um":
            x = self.um.expand(P.shape[0], K, -1)
            h = self.run(x, Pm + torch.arange(K, device=P.device), 0, V, cache, pvalid)
            return h[:, -1:], cache, pvalid
        if cfg.slots:
            n = cfg.slots
            spos = Pm + torch.arange(n, device=P.device)
            W = self.thought(tap, intervene) + self.slot
            last_kv = {}
            for r in range(K // n):
                H = self.run(W, spos, vlo, vhi, cache, pvalid, store=last_kv, bidir=True)
                if collect is not None:
                    collect.append(H)
                if r < K // n - 1:
                    W = self.thought(H, intervene) + self.slot
            for v in range(vlo, vhi):
                cache[v][2], cache[v][3] = last_kv[v]
            tap, last_pos = H[:, -1:], spos[-1:]
        else:
            for k in range(K):
                if window and k >= window:  # a ring of `window` thought slots: the new thought evicts the oldest
                    for v in range(vlo, vhi):
                        c = cache[v]
                        c[2], c[3] = (None, None) if window == 1 else (c[2][:, :, 1 - window :], c[3][:, :, 1 - window :])
                z = self.thought(tap, intervene)
                last_pos = torch.full((1,), Pm + k, device=P.device)
                tap = self.run(z, last_pos, vlo, vhi, cache, pvalid)
                if collect is not None:
                    collect.append(tap)
        if vhi < V:  # the last thought finishes the trip through the output layers
            tap = self.run(tap, last_pos, vhi, V, cache, pvalid)
        return tap, cache, pvalid

    # --------------------------------------------------------------------- API
    def forward(self, P, A, **kw):
        """Teacher-forced logits [B, A, vocab] for the answer field A (answer, EOS, PAD...)."""
        final, cache, pvalid = self.think(P, **kw)
        Pm, Am = P.shape[1], A.shape[1]
        pos = Pm + torch.arange(Am - 1, device=P.device)
        h = self.run(self.emb(A[:, :-1]), pos, 0, len(self.sched), cache, pvalid, store=None)
        return self.head(self.ln_f(torch.cat([final, h], 1)))

    def loss(self, P, A, **kw):
        logits = self(P, A, **kw)
        return Fn.cross_entropy(logits.flatten(0, 1).float(), A.flatten().long(), ignore_index=PAD)

    @torch.no_grad()
    def exact(self, P, A, **kw):
        """Greedy decoding reproduces A iff every teacher-forced argmax is right."""
        pred = self(P, A, **kw).argmax(-1)
        return ((pred == A) | (A == PAD)).all(1)

    @torch.no_grad()
    def generate(self, P, max_new, eos=2, conf=False, **kw):
        """Greedy decoding. conf=True also returns the log-probability of the decoded answer."""
        final, cache, pvalid = self.think(P, **kw)
        Pm = P.shape[1]
        out = []
        logits = self.head(self.ln_f(final)).float()
        done = torch.zeros(P.shape[0], dtype=torch.bool, device=P.device)
        logp = torch.zeros(P.shape[0], device=P.device)
        for i in range(max_new):
            tok = logits.argmax(-1)
            logp += torch.where(done, 0.0, logits.log_softmax(-1).gather(-1, tok[..., None])[:, 0, 0])
            out.append(tok)
            done |= tok[:, 0] == eos
            if done.all() or i == max_new - 1:
                break
            pos = torch.full((1,), Pm + i, device=P.device)
            h = self.run(self.emb(tok), pos, 0, len(self.sched), cache, pvalid)
            logits = self.head(self.ln_f(h)).float()
        out = torch.cat(out, 1)
        return (out, logp) if conf else out

    def tle(self, n_prompt, n_ans, K=None):
        """Token-layer evaluations for one problem with KV caching (the FLOPs proxy)."""
        cfg = self.cfg
        K = cfg.K if K is None else K
        V = len(self.sched)
        base = (n_prompt + n_ans) * V
        if K == 0 or cfg.mode in ("none", "ut", "cot"):
            return base
        if cfg.mode == "um":
            return base + K * V
        return base + K * (self.vhi - self.vlo) + (V - self.vhi)

    def describe(self):
        return asdict(self.cfg) | {"params": sum(p.numel() for p in self.parameters()), "sched": self.sched}

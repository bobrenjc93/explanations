# Three more hours: harness, checks and page generator

The second half of the experiment of `../../two-compilers-one-model`. Everything is as it ran, on one machine with one
H100; paths are that machine's. The harness (`bench.py`, `measure.py`, `run`, `tools/`) and the checks in `acc/` are
described in that folder's `code/README.md`. What is new here:

- `RULES.md` has the section both agents were given for the second half; `check.py` (`check`) is the command that runs
  the coordinator's checks on a side's current state.
- `acc/probe2.py`: the probe added for this page (a module attribute that is not a tensor is edited after compilation).
  `acc/cross_cache.sh`: compiles a state for the benchmark's random-weight model first and for the pretrained model
  second, from an empty compile cache.
- `make6.py`, `text6.py`, `appendix6.py` write this page; they read both halves through `make.py`.

`../results`: each side's final `config.py`, its journal and log over both halves, `final.patch` (its changes to the
compiler's source against the start), `tokens.json` (time and tokens at every recorded measurement), and `out/` with
the outputs of every check.

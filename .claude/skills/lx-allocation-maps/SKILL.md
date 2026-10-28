---
name: lx-allocation-maps
description: "Capture CP-SAT LX buffer allocations for compiled Spyre graphs, validate and plot them, and author an agent-only Markdown report. Use when collecting or visualizing LX allocation watermarks."
---

# LX Allocation Maps

Capture static CP-SAT buffer placement records from any model compile, parse the
JSONL into graph summaries, and generate per-graph allocation maps and operation
ledgers. Capture, parse, and plot are standalone CLIs under
`.claude/skills/lx-allocation-maps/`; users can run them directly without this
skill or an agent. Only the final experiment Markdown interpretation is
agent-authored.

## Important semantics

- A capture record is one allocator solve / compiled graph, **not** one runtime
  invocation. A model can invoke a graph many times or contain nested compiled
  regions. Capture a runtime trace separately if mapping graph IDs to call counts
  or compile regions matters.
- Plotted lifetimes and addresses are the frontend planner's static solve. They
  do not show emitted bundle instruction order, runtime occupancy, HBM traffic,
  or a model-wide peak across graphs.
- The plotted watermark is the highest resident buffer end address at each
  operation index, not the sum of resident buffer sizes.
- `capacity_bytes` is the frontend allocator's available range for that compile
  configuration, not necessarily the physical 2 MiB LX size.
- These scripts instrument `CoOptimizingAllocator._post_solve`. They capture
  the joint-placement allocator used by normal CP-SAT co-optimization. They do
  not promise records from placement-only `ScratchpadAllocator` solvers.
- The hook gathers FX source metadata and writes JSON during compilation. That
  adds compile-time and disk overhead; **never include hook-enabled timings in a
  latency comparison**. For timing, use a separate hook-free run with matching
  compiled cache policy.
- The capture hook is loaded only for the child command: `capture.py` prepends its
  own directory to `PYTHONPATH`. Do not install `sitecustomize.py` globally or
  in a model environment.
- Capture reads AOT context from PyTorch internals and imports Spyre Inductor
  internals. Use the same Python environment and checkout used to compile.

## Manual CLI workflow

Run from the torch-spyre checkout. Set `SKILL` to the checked-out script
directory, and choose an experiment directory under the repository. Preserve
all outputs there; do not use `/tmp` for experiment data.

```bash
SKILL="$PWD/.claude/skills/lx-allocation-maps"
RUN="$PWD/agentic-experiments/$(date -u +%Y%m%dT%H%M%SZ)-allocation-map"
mkdir -p "$RUN"
```

### 1. Capture while compiling / running the target

The target command must actually compile the graphs of interest. For a Python
script that runs a compiled model:

```bash
python "$SKILL/capture.py" --output-dir "$RUN/capture" -- \
  python path/to/run_model.py --your-model-args
```

Pass the normal compile configuration to the child process through exported
environment variables. For CP-SAT joint placement, the defaults are
`LAYOUT_SOLVER=cpsat`, `CO_OPTIMIZING_LX_PLANNING=1`, and `LX_PLANNING=1`;
verify them in `capture/run.json`. Require the child to use the target Python
runtime and repository. Use a distinct compiler cache when the allocation
configuration changes; preserve it under the experiment directory if graph
matching is needed. Spyre supports one device process at a time: run jobs
serially.

For vLLM, launch from its checkout using that checkout's environment and wrap
the actual benchmark command. Example one-request capture:

```bash
cd /path/to/spyre-inference
SKILL=/path/to/torch-spyre/.claude/skills/lx-allocation-maps
RUN="$PWD/agentic-experiments/<new-timestamp>-allocation-map"
mkdir -p "$RUN"
export TORCHINDUCTOR_CACHE_DIR="$RUN/torchinductor-cache"
env SPYRE_NUM_CPUS=8 \
  uv run --no-sync python "$SKILL/capture.py" \
    --output-dir "$RUN/capture" -- \
    uv run --no-sync vllm bench latency \
      --model google/gemma-4-26B-A4B-it \
      --input-len 1024 --output-len 64 --batch-size 1 \
      --max-model-len 1088 --num-iters-warmup 1 --num-iters 1
```

Adjust model and shapes to the user's workload. This one-warmup/one-measurement
example is for capture smoke only, not a stable latency benchmark. Record the
requested and resolved concurrency/token-budget settings from vLLM logs.



### Startup warnings, model support and run issues
Preserve and classify the full `capture/command.log`; do not suppress warnings or
assume a warning means capture failed. vLLM can warn that a model/shape is not
in the Spyre model registry while proceeding; report it as unvalidated even if
the run succeeds. Scheduler warnings can describe a requested
`max_num_batched_tokens` later overridden by the Spyre platform; record both
requested and resolved values. Triton/CPU-Triton fallback and Hugging Face Hub
authentication/rate-limit warnings are distinct from CP-SAT capture status;
record them and state whether they affected completion. Successful execution is
not an independent numerical correctness check.

Keep an issue log with the step/command, observed message or failure, diagnosis,
resolution/workaround, and proposed skill improvement. Record compiler warnings
(including work-division/provenance warnings) rather than treating captured
allocations as correctness evidence.
For repeatable measurements, pin the compile cache and solver settings explicitly
before launching the CLI; for example:

```bash
export LAYOUT_SOLVER=cpsat CO_OPTIMIZING_LX_PLANNING=1 LX_PLANNING=1
export DXP_LX_FRAC_AVAIL=0.45 SENCORES=32 SPYRE_NUM_CPUS=8
export TORCHINDUCTOR_CACHE_DIR="$RUN/torchinductor-cache"
export TORCH_COMPILE_DEBUG=1
export TORCH_COMPILE_DEBUG_DIR="$RUN/torch_compile_debug"
```

Pass the exact interpreter and module path expected by the model:

```bash
python "$SKILL/capture.py" --output-dir "$RUN/capture" -- \
  /path/to/python /path/to/run_model.py --your-model-args
```

Use a new empty `--output-dir` per capture run so records from multiple
processes/configurations cannot be mistaken for one capture set.

`capture.py` writes:

- `capture/allocations.<pid>.jsonl`: one complete JSON object per allocator
  solve, with graph identity, solver, buffers, operation origins, operation
  timeline, address watermark, capacity, and invariant flags;
- `capture/processes.<pid>.jsonl`: confirms which Python processes loaded the
  hook;
- `capture/run.json`: command, working directory, Python executable and key
  compile/allocation environment values;
- `capture/command.log`: target stdout and stderr.

If the script never triggers `CoOptimizingAllocator` or does not compile, no
allocation JSONL will be produced. Check the command log and process marker;
do not report an empty capture as a valid zero-allocation result.

### 2. Parse and validate the raw records

```bash
python "$SKILL/parse.py" "$RUN/capture/allocations.<pid>.jsonl" \
  --output-dir "$RUN/parsed"
```

The command validates JSON/schema, solver identity, graph operation counts,
resident-buffer spans, global watermark, and per-operation watermark. A valid
run exits zero and writes:

- `parsed/summary.json`: one row per captured graph with operation/source
  coverage, resident/spilled counts, watermark, capacity and fraction;
- `parsed/graphs.tsv`: compact spreadsheet-friendly view.

Invalid records produce an error and nonzero exit; fix or explain them before
plotting or reporting. `plot.py --record N` selects a capture record by its
zero-based line order in the JSONL.

### 3. Plot every graph or select individual records

Render every valid record:

```bash
python "$SKILL/plot.py" "$RUN/capture/allocations.<pid>.jsonl" \
  --output-dir "$RUN/plots"
```

Render one record with `--record N`; repeat the option to render a subset:

```bash
python "$SKILL/plot.py" "$RUN/capture/allocations.<pid>.jsonl" \
  --output-dir "$RUN/plots" --record 7
```

Plotting requires Matplotlib. If unavailable, install it to the absolute
experiment-local directory and use that same absolute path for `PYTHONPATH`:

```bash
uv pip install --target "$RUN/plot-deps" matplotlib
PYTHONPATH="$RUN/plot-deps" python "$SKILL/plot.py" \
  "$RUN/capture/allocations.<pid>.jsonl" --output-dir "$RUN/plots"
```

Each rendered graph produces PNG, HTML operation ledger, TSV operation ledger,
and JSON graph summary. `plots/manifest.json` indexes all outputs. The plotter
rechecks CP-SAT solver identity and recorded watermark/capacity invariants
before rendering. PNG color indicates peak-setting versus other resident
buffers; the red line is the per-operation address watermark; dashed line is
planner capacity. The HTML ledger includes active resident buffers, operation
watermark, FX source files/code, and operation names where available.

### 4. Agent-authored Markdown report — agent-dependent

Do not add a Markdown-report generator to the standalone CLIs. When the user
asks for a written report, the agent should inspect `parsed/summary.json`, the
raw capture JSONL, `capture/run.json`, relevant compile logs, and runtime trace
if available. Write `report.md` manually, recording:

- workload, shapes/dtypes, compile controls, device/runtime, repository
  revisions, commands, and run counts;
- captured graph inventory and operation/source coverage; graph-call counts
  only when supported by separately collected runtime evidence;
- highest watermarks and capacities, with links to the specific PNG/ledger/TSV;
- CP-SAT and invariant validation status;
- interpretation limits: static planner address extents, no inference of runtime
  LX occupancy, final instruction schedule, dynamic transactions, or aggregate
  live peak across graphs.

Never treat plot generation or non-empty captures as correctness proof. State
whether numerical correctness was independently checked and with what method.

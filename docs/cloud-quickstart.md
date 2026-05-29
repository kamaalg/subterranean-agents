# Cloud Quickstart

Before you run anything cloud-side, you need:

1. **A Modal account** ([sign up free](https://modal.com/signup)) — note that runs
   beyond the free tier ($30/month) require a credit card on file.
2. **An Anthropic API key** — used for data generation and evaluation, **not training itself**.
   A full 3B reproduction uses ~$15–50 of Claude tokens; an 8B run uses ~$40–120.
3. *(Optional)* **A Hugging Face token** if your base model is gated (Qwen 2.5/3 are not).

Run `agent2model cloud setup` once to wire all three together, then `agent2model cloud doctor`
any time you're unsure if things are configured.

## One-line setup

```bash
pip install "agent2model[cloud]"
agent2model cloud setup
```

The wizard is idempotent — each step inspects current state and skips itself if
the precondition is already met. It will:

1. Ask whether you already have a Modal account, opening
   [the signup page](https://modal.com/signup) if you don't.
2. Run `modal token new` for you when `~/.modal.toml` is missing (this opens a
   browser to authenticate; control returns automatically).
3. Prompt for your Anthropic API key with hidden input and create the
   `anthropic-secret` Modal Secret via `modal.Secret.create_deployed`. Your
   key is sent only to Modal — never printed or logged.

After the wizard, the same `agent2model cloud doctor` checklist is printed as
a final summary so you can see in one view what passed.

## Verify

```bash
agent2model cloud doctor
```

The doctor command is a read-only preflight — safe to run any time, costs
nothing. It checks, in order:

| # | Check | Severity |
|---|---|---|
| 1 | `modal` Python package is importable | critical |
| 2 | `~/.modal.toml` token file exists and is non-empty | critical |
| 3 | The `anthropic-secret` Modal Secret resolves in your workspace | critical |
| 4 | The local `ANTHROPIC_API_KEY` bills (1-token ping to Claude Haiku) | informational |
| 5 | A Hugging Face token works, if one is set in env or `~/.cache/huggingface/token` | informational |

A green checkmark means the precondition is satisfied; a red line includes the
exact shell command to fix it. The command exits non-zero **only** when a
critical check failed — informational results are advisory.

!!! note "Local key vs. Modal secret"
    Check 4 hits the `ANTHROPIC_API_KEY` env var on **your laptop** (used by
    `agent2model generate` and `agent2model eval` when running locally).
    Check 3 verifies the **Modal-side** secret, which is what the Modal workers
    in `cloud run` and the `reproduce_*` entrypoints use. They are independent;
    rotating one does not rotate the other.

## Run your first workflow

Once doctor is green you can launch the full pipeline:

```bash
# A user-supplied flowchart (YAML or LangGraph .py):
agent2model cloud run my_workflow.yaml --size 3b --n 2000 --epochs 20

# Or a paper reproduction:
modal run -m agent2model.cloud.modal_app::reproduce_travel
```

Every cloud entrypoint prints a **cost estimate** and asks for confirmation
before any spend happens:

```text
This run (travel, 3b, 2000 convos, 20 epochs, 200 eval scenarios) is estimated to cost:
  Modal GPU (train_3b, ~3.50h on A10G):  ~$3.85
  Anthropic API (generate, 2000 convos): ~$44.10
  Anthropic API (evaluate, 200 scenarios): ~$4.20
  TOTAL (excl. serve):              ~$52.15
Notes:
  - Rough estimate, ~+/-2x; actual depends on conversation length and turn count.
  - Modal A10G rate $1.10/hr; A100-80GB $3.40/hr; 8xA100-80GB $24.00/hr.
  - `--yes` skips this prompt (for CI / non-interactive runs).
Continue? [y/N]:
```

Pass `--yes` (or `-y`) to skip the prompt in non-interactive contexts; the
command fails fast with a clear error if stdin is not a TTY and `--yes` was
not supplied.

## Estimated costs

Rough order-of-magnitude estimates for the three paper reproductions, with
default `n`/`epochs`/`eval_n`. Actual cost depends on conversation length,
judge verbosity, and GPU contention — treat the numbers as ±2x.

| Recipe | Size | Generate | Eval | Train | **Total** |
|---|---|---|---|---|---|
| `travel` | 3B (A10G) | ~$44 | ~$4 | ~$4 (3.5h) | **~$52** |
| `zoom` | 8B (8x A100) | ~$275 | ~$4 | ~$12 (0.5h) | **~$292** |
| `insurance` | 8B (8x A100) | ~$138 | ~$4 | ~$12 (0.5h) | **~$154** |

These are *pre-cache* estimates. The data generator uses Anthropic prompt
caching aggressively (the flowchart spec is identical across every turn of a
run), which usually cuts generation cost ~50–60%.

Serving the compiled model is billed at the **per-hour** GPU rate of the
chosen instance (`A10G ~$1.10/hr`, `A100-80GB ~$3.40/hr`) for as long as the
endpoint is live and is **not** included in the totals above.

## Troubleshooting

- **`doctor` says modal is not installed.** Run
  `pip install "agent2model[cloud]"`.
- **`doctor` says the modal token is missing.** Run `modal token new`. The
  Modal CLI opens a browser, you authenticate, and control returns.
- **`doctor` says the `anthropic-secret` does not exist.** Run
  `agent2model cloud setup` and complete the third step, or create the secret
  manually at [modal.com/secrets](https://modal.com/secrets) (named exactly
  `anthropic-secret`, with key `ANTHROPIC_API_KEY`).
- **The Anthropic ping fails.** The local `ANTHROPIC_API_KEY` is either unset,
  malformed, or has run out of credits. The Modal Secret is independent — your
  cloud runs may still bill. Fix locally with `export ANTHROPIC_API_KEY=sk-ant-...`.
- **`cloud run` exits before launching.** It refuses to run when stdin is
  non-interactive and `--yes` was not passed, so the cost prompt does not hang
  a CI job. Re-run with `--yes`.
- **The estimate looks high.** It is conservative on purpose (estimator assumes
  no prompt caching; the generator caches aggressively at runtime). Treat the
  estimate as a ceiling; the per-run `cost.json` written under
  `build/<name>/cost.json` is the ground truth.

See the [Cloud deployment](cloud.md) page for the full reference of every
Modal worker function, the volume layout, and the RunPod alternative.

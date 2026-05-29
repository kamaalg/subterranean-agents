# RunPod recipes (secondary)

RunPod is the **secondary** cloud target — [Modal](../README.md) is the primary,
one-command path. RunPod is more manual: you launch a pod from a JSON spec, the
pod runs `setup.sh`, which installs `agent2model` and invokes the right
CLI command. Build artifacts live on the pod's persistent volume mounted at
`/workspace`.

## Files

| File | Purpose |
|---|---|
| `train_3b.json` | Single-GPU pod spec for the 3B path (Qwen2.5-3B-Instruct). |
| `train_8b.json` | 8x A100 80GB pod spec for the 8B DeepSpeed ZeRO-3 path. |
| `serve.json` | Single-GPU pod spec exposing the vLLM OpenAI endpoint on `:8000`. |
| `setup.sh` | Installs the package and runs a stage: `generate`, `train`, `evaluate`, `serve`. |

Each spec sets a `dockerArgs` that runs `bash setup.sh <stage>` from `/workspace`,
and an `env` block of `AGENT2MODEL_*` variables the script reads (see the header
comment in `setup.sh` for the full list).

## Flow

The pipeline mirrors the four CLI commands; only generation/eval need an
`ANTHROPIC_API_KEY`, only train/serve need a GPU.

1. **Prepare the build dir.** Compile your flowchart locally and upload the
   `build/<example>/` directory (with `flowchart.json`) to the pod volume at
   `/workspace/build/<example>/`. (`agent2model compile <yaml> --out build/<example>`.)

2. **Generate data** (CPU/API-bound — can run on a cheap pod or locally):

   ```bash
   AGENT2MODEL_EXAMPLE=travel ANTHROPIC_API_KEY=sk-ant-... bash setup.sh generate
   ```

3. **Train.** Create the pod from the spec (RunPod CLI or console). For the 3B path:

   ```bash
   runpodctl create pod --templateId "$(cat train_3b.json | ...)"   # or use the console
   ```

   In practice: import the JSON in the RunPod console "Deploy" form, or pass the
   fields to `runpodctl`/the GraphQL API. The pod runs
   `bash setup.sh train`, which calls `agent2model train` with the spec's
   `AGENT2MODEL_BASE_MODEL` / `AGENT2MODEL_EPOCHS`. The best checkpoint lands at
   `/workspace/build/<example>/model/best`.

   - `train_3b.json`: 1 GPU (A40/A10G class), 20 epochs.
   - `train_8b.json`: 8x A100 80GB, ZeRO-3, 10 epochs (the trainer's 8B preset
     wires in `agent2model/training/deepspeed/zero3.json`).

4. **Evaluate** (CPU/API-bound):

   ```bash
   AGENT2MODEL_EXAMPLE=travel ANTHROPIC_API_KEY=sk-ant-... bash setup.sh evaluate
   ```

   Writes `eval_report.json` / `eval_report.pdf` into the build dir.

5. **Serve.** Deploy `serve.json`; the pod runs `bash setup.sh serve`, exposing an
   OpenAI-compatible endpoint on port `8000` (`/v1/chat/completions`, `/v1/models`).
   Point any OpenAI client at `https://<pod-id>-8000.proxy.runpod.net/v1`.

## Notes

- RunPod has no managed secret store as integrated as Modal's; set
  `ANTHROPIC_API_KEY` in the pod `env` (or the console) for the generate/evaluate
  stages. Treat the value as sensitive.
- These specs use the `runpod/pytorch` CUDA 12.1 / Python 3.11 base image. Pin a
  different `imageName` if you need another CUDA/torch combo.
- For a fully unattended run, chain stages in a single pod by editing `dockerArgs`
  to call `setup.sh generate && setup.sh train && setup.sh evaluate` — but keep the
  GPU pod's generation step short, or run generation on a separate cheap pod first.

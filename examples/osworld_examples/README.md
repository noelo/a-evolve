# OSWorld Benchmark with A-EVOLVE

Run the A-EVOLVE propose+curator evolution loop on [OSWorld](https://github.com/xlang-ai/OSWorld) — a benchmark of 369 GUI tasks across 10 desktop application domains.

## Architecture

- **Agent**: Claude with `computer_use` tool (screenshot + accessibility tree)
- **Environment**: AWS EC2 VMs with Ubuntu desktop, one per task
- **Evaluation**: Environment-specific checks (file content, UI state, etc.)
- **Evolution**: Per-topic skill curation + general cross-topic skills

## Setup

### 1. Clone OSWorld

```bash
git clone https://github.com/xlang-ai/OSWorld
cd OSWorld
pip install -e .
```

### 2. AWS Configuration

OSWorld VMs run on EC2 instances. You need:
- An AWS account with EC2 access
- A VPC subnet with internet access
- A security group allowing inbound ports: 5000 (server), 9222 (Chrome CDP), 5900 (VNC)

```bash
export OSWORLD_PATH=/path/to/OSWorld
export AWS_REGION=us-east-1
export AWS_SUBNET_ID=subnet-xxxxx
export AWS_SECURITY_GROUP_ID=sg-xxxxx
export ANTHROPIC_API_KEY=sk-ant-xxxxx
```

### 3. AMI Selection

| Region | AMI | Chrome | Notes |
|--------|-----|--------|-------|
| us-east-1 | `ami-0d23263edb96951d8` | < 147 | Stable, recommended |
| us-west-2 | `ami-083ebc5e7cee75c51` | 147 | Requires `--user-data-dir` fix |

The `us-east-1` AMI is the default OSWorld AMI and is recommended for reproducibility.

### 4. Chrome 147 Fix (us-west-2 AMI only)

Chrome 147+ requires a non-default `--user-data-dir` for `--remote-debugging-port` to work. The fix is already applied in `OSWorld/desktop_env/controllers/setup.py` — it automatically copies the Chrome profile to `/tmp/chrome-debug-profile` and adds the flag when launching Chrome with remote debugging.

If using the us-west-2 AMI, ensure this patch is applied to your OSWorld installation.

## Running

```bash
# From a-evolve root
bash examples/osworld_examples/run_osworld.sh
```

### Key Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--task-file` | JSON file listing tasks | `test_all.json` (369 tasks) |
| `--provider aws` | VM provider | Required |
| `--solver-model 1` | Solver model tier (1=Opus, 2=Sonnet) | 1 |
| `--curator-model 1` | Curator model tier | Same as solver |
| `--batch-size N` | Tasks per evolution batch | 16 |
| `--workers N` | Parallel VMs | 16 |
| `--max-steps N` | Max agent steps per task | 100 |
| `--max-skills-per-topic N` | Skills kept per domain | 5 |
| `--max-general-skills N` | Cross-domain skills kept | 5 |
| `--no-evolve` | Disable evolution (baseline) | False |
| `--no-seed-skills` | Start with empty workspace | False |
| `--shuffle` | Randomize task order | False |
| `--lazy-load` | Show skill names only, use read_skill tool for body | False |

### Example Configurations

```bash
# Pure baseline (no skills, no evolution)
python examples/osworld_examples/evolve_osworld.py \
    --task-file $OSWORLD_PATH/evaluation_examples/test_all.json \
    --provider aws --solver-model 1 \
    --no-evolve --no-seed-skills \
    --max-steps 30 --batch-size 10 --workers 10 \
    --output-dir outputs/osworld_baseline

# Full evolution
python examples/osworld_examples/evolve_osworld.py \
    --task-file $OSWORLD_PATH/evaluation_examples/test_all.json \
    --provider aws --solver-model 1 \
    --curator-model 1 --selector-model 1 \
    --no-seed-skills \
    --max-skills-per-topic 5 --max-general-skills 5 \
    --shuffle --shuffle-seed 42 \
    --max-steps 100 --batch-size 16 --workers 16 \
    --output-dir outputs/osworld_evolve
```

## Cost Estimate

- Each task uses one `t3.xlarge` EC2 instance (~$0.17/hr)
- With 16 workers and 100 steps/task, a full 369-task run takes ~6-8 hours
- Approximate cost: ~$50-80 in EC2 + ~$200-400 in API calls (Opus)

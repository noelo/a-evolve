# OSWorld Benchmark with A-EVOLVE

Run the A-EVOLVE propose+curator evolution loop on [OSWorld](https://github.com/xlang-ai/OSWorld) — a benchmark of 369 GUI tasks across 10 desktop application domains.

## Architecture

- **Agent**: Claude with `computer_use` tool (screenshot + accessibility tree)
- **Environment**: AWS EC2 VMs with Ubuntu desktop, one per task
- **Evaluation**: Environment-specific checks (file content, UI state, etc.)
- **Evolution**: Per-topic skill curation + general cross-topic skills

## Setup

### 1. Install OSWorld

```bash
git clone https://github.com/xlang-ai/OSWorld
cd OSWorld
pip install -e .
```

Task data (`evaluation_examples/test_all.json`, 369 tasks across 10 domains) is included in the OSWorld repo.

For pre-downloaded cache files (used during task setup), download from [Google Drive](https://drive.google.com/file/d/1XlEy49otYDyBlA3O9NbR0BpPfr2TXgaD/view?usp=drive_link) and extract to `OSWorld/cache/`.

### 2. AWS Configuration

OSWorld uses a Host-Client architecture: your host machine manages EC2 instances, each running an Ubuntu desktop as the task environment.

#### 2.1 AWS Credentials

```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (us-east-1)
```

#### 2.2 Security Group

Create a security group with the following **inbound rules**:

| Type | Protocol | Port | Source | Purpose |
|------|----------|------|--------|---------|
| SSH | TCP | 22 | 0.0.0.0/0 | SSH access |
| Custom TCP | TCP | 5000 | 172.31.0.0/16 | OSWorld backend service |
| Custom TCP | TCP | 5910 | 0.0.0.0/0 | NoVNC visualization |
| Custom TCP | TCP | 8006 | 172.31.0.0/16 | VNC service |
| Custom TCP | TCP | 8080 | 172.31.0.0/16 | VLC service |
| Custom TCP | TCP | 9222 | 172.31.0.0/16 | Chrome CDP |

Outbound: Allow all traffic.

#### 2.3 VPC and Subnet

The host machine and all OSWorld VMs must reside in the same VPC subnet. Use the subnet of your host instance (visible in EC2 console → Instance → Networking).

#### 2.4 Environment Variables

Create a `.env` file in the OSWorld directory or export directly:

```bash
export OSWORLD_PATH=/path/to/OSWorld
export AWS_REGION=us-east-1
export AWS_SUBNET_ID=subnet-xxxxxxxxx
export AWS_SECURITY_GROUP_ID=sg-xxxxxxxxx
```

AWS credentials must have Bedrock access with Claude models enabled (us-east-1 or us-west-2). Configure via `aws configure` or environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

The AMI `ami-0d23263edb96951d8` (us-east-1) is the official OSWorld VM image and is used by default.

### 3. Install a-evolve

```bash
cd /path/to/a-evolve
pip install -e ".[osworld]"
```

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

## Results

| Configuration | Tasks | Pass Rate |
|---------------|-------|-----------|
| Baseline (no skills, no evolution) | 369 | 65.6% |
| **A-EVOLVE (propose+curator)** | 369 | **68.3%** |

+2.7% absolute improvement. The evolve run uses `--no-seed-skills --shuffle --shuffle-seed 42 --max-steps 100` with Opus as solver/curator/selector, batch-size 16.

## Cost Estimate

- Each task uses one `t3.xlarge` EC2 instance (~$0.17/hr)
- With 16 workers and 100 steps/task, a full 369-task run takes ~6-8 hours
- Approximate cost: ~$50-80 in EC2 + ~$200-400 in API calls (Opus)

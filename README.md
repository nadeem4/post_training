# Post-training Methods for LLMs

This repo collects post-training methods for Large Language Models (LLMs) with
small, focused implementations and runnable examples. The goal is to make
alignment and reinforcement post-training practical, understandable, and
reproducible.

## Scope

- Post-training methods that start from a pretrained model.
- Minimal, readable implementations over full-scale training stacks.
- RL fundamentals in Gymnasium to build intuition for later LLM alignment.

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Repository Layout

- `gymnasium/`: RL foundations (CartPole examples).
- `requirements.txt`: Python dependencies.
- `README.md`: learning path and run instructions.

## Current Examples

### Gymnasium: CartPole Random Policy

Runs a single random rollout to verify environment setup.

Code: [gymnasium/cartpole_random.py](gymnasium/cartpole_random.py)

```bash
python gymnasium/cartpole_random.py
```

### Gymnasium: CartPole Q-Learning

Trains a discretized Q-learning agent and evaluates it.

Code: [gymnasium/cartpole_q_learning.py](gymnasium/cartpole_q_learning.py)

```bash
python gymnasium/cartpole_q_learning.py
```

Evaluation renders by default.

## Learning Path (Planned)

Each step builds on the previous one; diagrams show simplified dataflow.

### 1) Reinforcement Learning Foundations

Topics:
- Value-based RL (Q-learning, intuition and DQN basics)
- Exploration vs exploitation
- Why policy optimization is needed
- Where classical RL begins to fail for LLMs

Diagram:
```mermaid
flowchart LR
  Env["Environment"] -->|"State s(t)"| Agent["Agent"]
  Agent -->|"Action a(t)"| Env
  Env -->|"Reward r(t+1)"| Agent
  Env -->|"State s(t+1)"| Agent
```

### 2) Supervised Fine-tuning (SFT) / Instruction Tuning

Topics:
- Dataset formatting (prompt/response pairs)
- Loss functions (cross-entropy)
- Establishing evaluation baselines
- Makes pretrained models follow instructions

Diagram:
```mermaid
flowchart LR
  Pretrained["Pretrained Model"] --> Train["SFT Training"]
  Data["SFT Data (Instruction + Response)"] --> Train
  Train --> Instruction["Instruction-Following Model"]
```

### 3) Preference Optimization (No-RL Alignment)

Includes:
- DPO
- IPO
- KTO
- ORPO

Focus:
- Align models directly using preference pairs
- Often cheaper and more stable than PPO

Diagram:
```mermaid
flowchart LR
  Prompt[User Prompt] --> Base[Base Model]
  Base --> Responses[Candidate Responses]
  Responses --> Prefs[Preference Labels]
  Prefs --> Opt[Preference Optimization]
  Base --> Opt
  Opt --> Aligned[Aligned Model]
```

### 4) Reward Modeling + RLHF (PPO / GRPO Variants)

Coverage:
- Reward model training from preference data
- PPO-based RLHF
- GRPO-style policy optimization (often reducing explicit reward models)
- Why RL can still help (reasoning, safety shaping, controllability)

Diagram:
```mermaid
flowchart LR
  SFT[Supervised Fine-Tuning] --> PPO[PPO RLHF]
  Pref[Preference Data] --> RM[Reward Model Training]
  RM --> PPO
  PPO --> Aligned[Final Aligned Model]
  SFT --> GRPO[GRPO Optimization]
  Pref --> GRPO
  GRPO --> Aligned
```

### 5) RLAIF (AI-Generated Preferences)

- Replaces human labels with LLM judgement
- Enables preference optimization at scale
- Reduces dependence on human annotators

Diagram:
```mermaid
flowchart LR
  Prompt[User Prompt] --> Policy[Policy Model]
  Policy --> Responses[Candidate Responses]
  Responses --> Judge[Judge Model]
  Judge --> Prefs[AI Preference Labels]
  Prefs --> Update[Preference Optimization]
  Policy --> Update
  Update --> Updated[Updated Policy Model]
```

### 6) Constitutional and Safety Tuning

- Constitutional AI foundations
- Self-critique loops guided by a ruleset
- Safety layered across the pipeline

Diagram:
```mermaid
flowchart LR
  Output[Model Output] --> Critique[Self-Critique]
  Constitution[Constitution / Rules] --> Critique
  Critique --> Revise[Revision]
  Revise --> Safer[Safer Output]
```

### 7) Evaluation and Comparison Harnesses

- Standardized evaluation harnesses
- Benchmark frameworks
- Reasoning, helpfulness, and safety scoring

Diagram:
```mermaid
flowchart LR
  Model --> Bench[Benchmark Suite]
  Bench --> Metrics[Metrics + Regression Tracking]
```

### 8) Distillation and Post-training Compression

- Alignment-preserving distillation
- Smaller, deployable aligned models
- Practical deployment-focused tradeoffs

Diagram:
```mermaid
flowchart LR
  Teacher[Large Aligned Model] --> Data[Distillation Data]
  Student[Smaller Model] --> Distill[Distillation Training]
  Data --> Distill
  Distill --> Small[Smaller Aligned Model]
```

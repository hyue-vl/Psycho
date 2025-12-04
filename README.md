# Psycho-World: DreamCUB-Inspired POMDP Intervention Framework

This repository implements the Psycho-World architecture described in the prompt: a dual-belief POMDP environment for computational psychotherapy. The goal is to transform qualitative clinical guidance into executable code that supports data processing, model pretraining, simulation, and reinforcement learning with personalized memory.

## Core Concepts
- **POMDP Formalization**: Each state `s_t = (h_t, b_t)` couples observable dialogue history with latent belief structures (emotion, cognition, intent, psychodynamics, risk).
- **Action Structure**: Agent actions `a_t = (k_t, u_t)` bind a therapeutic strategy label to a generated utterance.
- **Belief & Observation Models**: The world simulator factorizes transition dynamics into belief evolution and observation generation, guided by COKE cognitive chains and DreamCUB heuristics.
- **Reward Shaping**: Rewards combine belief improvement, explicit risk penalties, and empathy consistency scores.

## Repository Layout

```
psycho_world/
  agents/               # Dreaming planner + PPO/MCTS abstractions
  data/                 # ETL + belief annotation pipelines
  envs/                 # Gymnasium-compatible Psycho-World environment
  knowledge/            # Dual knowledge base connectors (DKB + PKB)
  memory/               # MemGPT-powered personalized state manager
  models/               # Cognitive encoder + world simulator wrappers
  prompts/              # Instruction templates for latent state inference
  training/             # Supervised and RL training loops
  utils/                # Shared helpers (logging, serialization, reward)
```

## Development Roadmap
1. Build the **COKE ETL** to populate a Chroma + Neo4j dual store (`data/etl_coke.py`).
2. Generate **belief-annotated dialogues** from ESConv/PsyQA via LLM hindsight labeling (`data/data_annotation.py`).
3. Pretrain **cognitive encoder** (`models/cognitive_encoder.py`) and **world simulator** (`models/world_model.py`) using the annotated corpus (`training/*.py`).
4. Wrap the simulator inside a **Gymnasium environment** (`envs/psycho_env.py`) with therapeutic rewards.
5. Implement **dreaming agents** (`agents/dreaming_agent.py`) that plan via MCTS/PPO while querying the dual knowledge base.
6. Integrate **MemGPT** (`memory/pkb_manager.py`) to persist personalized trajectories.

## Getting Started
1. Install dependencies (see `pyproject.toml` once finalized) and configure API keys for OpenAI/Llama/MemGPT.
2. Run `python -m psycho_world.data.etl_coke --config configs/coke.yaml` to build the DKB.
3. Execute `python -m psycho_world.data.data_annotation --dataset esconv` to produce POMDP-ready trajectories.
4. Launch pretraining scripts in `psycho_world/training/`.
5. Start RL fine-tuning with `python -m psycho_world.training.train_psycho_agent`.

## Status
The repo currently contains scaffolding for each component with clear docstrings and TODO markers. Replace stubbed methods with concrete implementations as you connect real datasets, APIs, and model weights.
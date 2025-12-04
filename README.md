# Psycho-World: Dual-Belief POMDP Simulator for Psychotherapy R&D

Psycho-World turns the DreamCUB research sketch into runnable code: conversations are modeled as a partially observable Markov decision process (POMDP) where the latent belief state of a user evolves according to cognitive therapy heuristics. The repository provides data tooling, simulator abstractions, planning/training loops, and convenient configuration so you can prototype therapeutic agents quickly.

## Highlights
- **Dual knowledge base**: ingest COKE cognitive chains (domain/prior knowledge) and synchronize personalized memories through MemGPT-like services.
- **Belief-centric simulator**: Gymnasium environment stitches together a cognitive encoder, a world model, and a therapeutic reward.
- **Planning + dreaming**: an MCTS-style agent reasons over simulated rollouts before acting, enabling safer exploration.
- **Training pipelines**: scripts for annotation, supervised fine-tuning, and RL-style policy improvement.
- **Configurable everywhere**: dataclasses + JSON loader ensure hyperparameters and endpoints stay in sync across scripts.

## System Flow
1. **Data/Knowledge prep** – Convert COKE chains into Chroma + Neo4j stores (`psycho_world/data/etl_coke.py`) and annotate ESConv/PsyQA turns with belief labels (`psycho_world/data/data_annotation.py`).
2. **Model pretraining** – Fine-tune the cognitive encoder (`q_φ`) and world simulator (`p_θ`) using the annotated corpus via `psycho_world/training/train_*.py`.
3. **Environment wiring** – Instantiate `PsychoWorldEnv` with the pretrained models, reward function, and a user profile.
4. **Dreaming agent** – Use `DreamingAgent` to run short-horizon rollouts inside the environment before emitting the next utterance.
5. **Memory + analytics** – Persist risk signals and trajectory summaries to the personalized knowledge base (`PKBManager`) for longitudinal tracking.

## Repository Map
- `psycho_world/`: core package (agents, envs, knowledge, memory, models, prompts, training, utils, config).
- `configs/`: JSON/TOML configs describing runtime wiring (see `configs/default.json`).
- `pyproject.toml`: dependency + packaging metadata.
- `README.md`: this guide.

## File-by-File Reference

| Path | Purpose |
| --- | --- |
| `psycho_world/__init__.py` | Package metadata helper exposing `get_version()`. |
| `psycho_world/config.py` | Dataclasses for each subsystem plus an `LLMConfig`/`RuntimeConfig` loader that parses JSON configs. |
| `psycho_world/agents/__init__.py` | Exports agent entrypoints. |
| `psycho_world/agents/dreaming_agent.py` | Limited-depth MCTS planner that simulates trajectories in `PsychoWorldEnv`. |
| `psycho_world/data/__init__.py` | Bundles ETL/annotation utilities for easy import. |
| `psycho_world/data/data_annotation.py` | LLM-powered hindsight labeling over dialogues; produces belief-tagged datasets. |
| `psycho_world/data/etl_coke.py` | Converts COKE JSON into Chroma vectors and (optionally) Neo4j graph records. |
| `psycho_world/envs/__init__.py` | Convenience re-export for Gym environments. |
| `psycho_world/envs/psycho_env.py` | Gymnasium environment that loops history → belief inference → simulation → reward. |
| `psycho_world/knowledge/__init__.py` | Re-exports knowledge connectors. |
| `psycho_world/knowledge/retriever.py` | Vector retriever for domain knowledge + PKB adapter backed by MemGPT. |
| `psycho_world/memory/__init__.py` | Memory namespace exports. |
| `psycho_world/memory/pkb_manager.py` | Thin manager around MemGPT clients (load user, append belief updates, log sessions). |
| `psycho_world/models/__init__.py` | Convenience exports for model wrappers. |
| `psycho_world/models/cognitive_encoder.py` | LLM wrapper that infers latent belief states via `format_cognitive_prompt`. |
| `psycho_world/models/world_model.py` | Hybrid world simulator that retrieves knowledge, builds prompts, and predicts next beliefs/utterances. |
| `psycho_world/models/reward.py` | Therapeutic reward shaping (belief delta + risk penalty + empathy score). |
| `psycho_world/prompts/__init__.py` | Prompt factory exports. |
| `psycho_world/prompts/cognitive.py` | Chinese-language instruction template + helper used by data + encoder modules. |
| `psycho_world/training/__init__.py` | Collects training entrypoints. |
| `psycho_world/training/train_cognitive_encoder.py` | Supervised fine-tuning loop for `q_φ` with Hugging Face Trainer. |
| `psycho_world/training/train_world_model.py` | Teacher-forced training loop for the world simulator `p_θ`. |
| `psycho_world/training/train_psycho_agent.py` | RL-style outer loop that streams trajectories from the dreaming agent. |
| `psycho_world/utils/__init__.py` | Utility exports. |
| `psycho_world/utils/logging.py` | Standardized console logger factory. |
| `psycho_world/utils/serialization.py` | JSON helpers (ensure ASCII-safe belief serialization). |
| `pyproject.toml` | Package metadata and dependency pins (LangChain, HF, Gymnasium, Torch). |
| `configs/default.json` | Ready-to-edit runtime configuration that plugs into `load_runtime_config()`. |

> Tip: Although many modules currently stub external dependencies (LLMs, MemGPT, candidate generators), docstrings spell out exactly what needs to be wired once credentials or models are available.

## Configuration
- Declare your stack in JSON (see `configs/default.json`). The schema mirrors the dataclasses in `psycho_world/config.py`:
  - `llm`: provider/model/temperature/base URL information.
  - `knowledge_base`: file paths for the COKE corpus, Chroma persistence dir, and Neo4j credentials.
  - `annotation`, `training`, `dreaming`: hyperparameters shared across scripts.
- Load it anywhere with:
  ```python
  from psycho_world.config import load_runtime_config

  runtime_cfg = load_runtime_config("configs/default.json")
  print(runtime_cfg.training.learning_rate)
  ```
- For experimentation you can keep multiple config files (e.g., `configs/prod.json`, `configs/offline.json`) and swap them via CLI flags.

## Typical Workflow
1. **Install**: `pip install -e .` (uses dependencies from `pyproject.toml`).
2. **Build the dual KB**: `python -m psycho_world.data.etl_coke --input data/coke.json --persist ./data/coke_db`.
3. **Annotate dialogues**: adapt `DialogueAnnotator` for ESConv/PsyQA dumps and export to `./data/processed`.
4. **Pretrain models**: call the training scripts with a `TrainingConfig` built from `runtime_cfg.training`.
5. **Run dreaming simulations**: instantiate `PsychoWorldEnv`, load weights, then feed the env + config into `train_psycho_agent`.
6. **Persist sessions**: connect `PKBManager` to MemGPT (or compatible API) to store belief deltas + transcripts.

## Development Notes
- All modules are annotated and decomposed for rapid iteration—swap in your own LLM/embedding clients as long as they expose the same `generate()` interface.
- External services (OpenAI, Neo4j, MemGPT) are intentionally imported lazily so offline testing is still possible.
- Extend the reward model, retrieval logic, or planner without touching unrelated modules thanks to the clean dataclass-driven configuration boundary.
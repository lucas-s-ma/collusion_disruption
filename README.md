# Collusion disruption: entry into an AI pricing market

This repository runs a matched simulation of seller entry into a market priced by AI agents. Two incumbent LLM agents price for 100 periods. The completed market is then copied into three paths: no entry, entry by another LLM agent, and entry by a rule-based seller that maximizes predicted current-period profit. The experiment uses the differentiated-product demand model and the P1/P2 economic prompt prefixes from [Fish, Gonczarowski, and Shorrer (2026)](https://arxiv.org/abs/2404.00806).

The code is intended to compare price and profit paths, not to infer collusion from high prices alone. The P1 and P2 runs use different starting histories, so their difference is descriptive rather than a clean estimate of a prompt effect.

## Repository layout

| Path | Contents |
| --- | --- |
| [`experiment.py`](experiment.py) | Simulation, DeepSeek requests, resumable storage, CSV exports, and plots |
| [`tests/`](tests/) | Offline tests for the market and run workflow |
| [`docs/`](docs/) | [Reference paper](docs/fish_etal_2026_algorithmic_collusion_llms.pdf) and [project proposal](docs/project_proposal.pdf) |
| [`.env.example`](.env.example) | Template for the API key used by live runs |

`runs/` holds generated data. `analysis/` holds local analysis and presentation files. Both, along with `.vscode/` and `.env`, are ignored by Git.

## Setup

Use Python 3.10 or newer. On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a live run, create `.env` from `.env.example` if it does not already exist, then set `DEEPSEEK_API_KEY` in `.env`. An environment variable of the same name takes precedence. Offline commands do not need a key.

```sh
cp .env.example .env
```

## Run the experiment

Check the default design and exercise the full pipeline without API calls:

```sh
python experiment.py --dry-run
python experiment.py --mock --run-id smoke --output-dir /tmp/cs535_smoke --rounds 6 --entry-round 3 --window 3
python -m unittest discover -s tests -v
```

The defaults are 200 periods, entry at the **start of period 101** (`--entry-round 100`), one starting history, and a 100-period observation window. One complete history requires 900 LLM decisions before retries: 200 before entry and 700 across the three continuation paths. A live run requires the explicit `--live` flag:

```sh
python experiment.py --live --run-id p1_example --prompt-mode conservative --request-limit 900
python experiment.py --live --run-id p2_example --prompt-mode risky --request-limit 900
```

`conservative` uses P1, which discourages actions that undermine profitability. `risky` uses P2, which encourages more aggressive price tests. Both instruct LLM agents to maximize long-run profit. The rule-based entrant receives neither prompt. The incumbents and strategic entrant receive the same selected prompt; no-entry agents continue to see a two-seller market after period 100.

`--request-limit` caps cumulative request attempts, including retries. If a run stops at the limit, resume it with a larger limit; configuration such as prompt mode, model, and round schedule is fixed for that run:

```sh
python experiment.py --live --resume runs/p1_example --request-limit 1200
python experiment.py --plot-only --resume runs/p1_example
```

The second command rebuilds CSV files and figures without provider calls. Run `python experiment.py --help` for all controls, including `--histories`, `--rounds`, `--window`, and price bounds.

## Model and saved results

In each period, active sellers choose prices simultaneously. Sales follow a deterministic logit demand function with 100 potential purchases, an outside option, unit cost 1, and prices between 0 and 4.50 by default. The rule-based entrant knows this demand function and best responds to the incumbents' previous-period prices. LLM agents see public past prices and their own past outcomes and notes, but not the demand formula or rivals' current-period choices.

Each run writes its configuration, a resumable SQLite state, CSV exports, and PNG/SVG figures under `runs/<run-id>/`. The SQLite database is authoritative; exports and plots can be regenerated with `--plot-only`. API keys are not written to run data.

## References

- Fish, S., Gonczarowski, Y. A., and Shorrer, R. (2026). [*Algorithmic Collusion by Large Language Models*](https://arxiv.org/abs/2404.00806). A [local PDF](docs/fish_etal_2026_algorithmic_collusion_llms.pdf) is included.
- Calvano, E., Calzolari, G., Denicolò, V., and Pastorello, S. (2020). [*Artificial Intelligence, Algorithmic Pricing, and Collusion*](https://doi.org/10.1257/aer.20190623). *American Economic Review* 110(10), 3267–3297.

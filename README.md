# TimeWarp for Inspect AI

An [Inspect AI](https://inspect.aisi.org.uk/) implementation of **TimeWarp**, a benchmark that measures how robust web agents are to changes in web UI over time.

- Paper: [TimeWarp: Evaluating Web Agents by Revisiting the Past](https://arxiv.org/abs/2603.04949) (Md Farhan Ishmam and Kenneth Marino, NeurIPS 2026 Evaluations and Datasets Track)
- Project page: <https://timewarp-web.github.io>
- Upstream benchmark (environments, BrowserGym tasks, training code): <https://github.com/sparklabutah/timewarp>
- Dataset: [`sparklabutah/timewarp`](https://huggingface.co/datasets/sparklabutah/timewarp)

TimeWarp has three locally served websites: Wiki, News and Shop. Each site comes in six UI versions, following web design from the early 2000s to today. The same 231 goals are posed in every version, so a drop in success between versions comes from the UI rather than the task. Agents answer questions, compare information across pages and sites, and place orders. They report results in a final free-text message, which deterministic verifiers then score.

| Version | Wiki | News | Shop |
| --- | --- | --- | --- |
| v1 | Wikipedia 2001 | BBC News 1998-2001 | Amazon 1999-2004 |
| v2 | Wikipedia 2002-03 | BBC News 2002-07 | Amazon 2005-11 |
| v3 | Wikipedia 2003-04 | BBC News 2008-15 | Amazon 2012-14 |
| v4 | Wikipedia 2005-22 | BBC News 2016-22 | Amazon 2015-25 |
| v5 | Wikipedia 2023-25 | BBC News 2023-25 | Temu 2025 |
| v6 | Minimal | Minimal | WebShop |

Era labels follow Figure 2 of the paper. v6 is a control interface rather than a historical era.

## Usage

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and Docker with Compose. Each sample runs in its own Docker Compose project with two containers:

- **Browser:** the [Playwright MCP](https://github.com/microsoft/playwright-mcp) server with headless Chromium.
- **TimeWarp servers:** the Wiki, News and Shop servers for the sample's UI version.

```bash
git clone https://github.com/sparklabutah/timewarp-inspect.git
cd timewarp-inspect
uv sync
uv run inspect eval timewarp/timewarp --model openai/gpt-5-nano
```

The first run builds the `timewarp-env` image from `src/timewarp/sandbox/env/Dockerfile`:

- Size: about 4.2 GB (measured on arm64).
- Build time: 20-40 minutes, depending on network and CPU.
- What it pulls: the upstream environment code at a pinned commit, the environment data from Hugging Face at a pinned revision, and a CPU build of PyTorch, which the WebShop search engine (Pyserini) needs.

Each sample starts its own servers. On a 10-core laptop they took about 2.5 minutes to become ready and used about 3.3 GB of memory. The compose file allows the servers 5 GB and the browser 2 GB per sample. Choose `--max-samples` / `--max-sandboxes` to fit your machine.

The sandbox needs outbound internet access. The page themes load Bootstrap, jQuery and fonts from public CDNs, and Shop product images come from Amazon's image hosts, as in the upstream environments. The browser blocks requests to every other origin (see [Differences from the paper](#differences-from-the-paper)).

### Options

```bash
# The paper reports the mean of 3 runs on the test split
uv run inspect eval timewarp/timewarp --model openai/gpt-5-nano --epochs 3

# One environment and selected UI versions
uv run inspect eval timewarp/timewarp -T categories=shop -T ui_versions=[1,6]

# Give vision-capable models a screenshot tool
uv run inspect eval timewarp/timewarp -T screenshots=true

# Use a different judge for the two LLM-judged goals
uv run inspect eval timewarp/timewarp --model-role grader=anthropic/claude-sonnet-5-5
```

| Parameter | Default | Description |
| --- | --- | --- |
| `split` | `"test"` | `"test"` (goals 1-103, used for the paper's results), `"train"` (goals 104-231) or `"all"`. |
| `ui_versions` | all | UI versions to include, 1-6. |
| `categories` | all | `"wiki"`, `"news"`, `"shop"` and/or `"multi"` (goals that span several sites). |
| `max_steps` | `30` | Agent steps per episode, including the final answer, as in the paper. A step is one browser action. The task also caps model turns at `max_steps * 5` for any solver. |
| `screenshots` | `false` | Add Playwright MCP's `browser_take_screenshot` tool. |
| `solver` | `browser_agent()` | Replace the agent. A replacement must leave its final answer in the transcript as the result of a tool named `submit`; with Inspect's `react()`, pass `submit=AgentSubmit(keep_in_messages=True)`. |
| `scorer` | `timewarp_scorer()` | Replace the scorer. |

Sample IDs have the form `<task_id>_v<version>`, e.g. `--sample-id 1_v3`.

## Dataset

Each goal in `data.json` (revision `246edb1`) becomes one sample per UI version. The test split has 103 goals, so 618 samples:

| Category | Goals | Samples |
| --- | --- | --- |
| Wiki | 31 | 186 |
| News | 22 | 132 |
| Shop | 27 | 162 |
| Multi | 23 | 138 |

The train split has 128 goals (768 samples). In the dataset, train goals carry human-refined step-by-step plans (`additional_instructions`). The upstream TimeTraj method uses these plans to collect teacher trajectories, and they include the answer, so this implementation never shows them to the agent.

## Scoring

The scorer reads the last answer the agent sent with the `submit` tool. As in the upstream harness, a sample with no submitted answer scores 0.

Each goal declares one or more verifiers, and an answer must pass all of them:

| Verifier | Goals | Checks |
| --- | --- | --- |
| `string_match` | 156 (+8 combined) | Required and forbidden entries, matched on word boundaries after normalization. Some goals only check the first sentence. |
| `number_match` | 39 (+5 combined) | The expected numbers appear in any format (`7,000,000`, `7 million`), optionally within a tolerance. |
| `list_match` | 26 (+3 combined) | Every item of an enumeration appears, in order when the goal requires it. |
| `llm_judge` | 2 | The `grader` model role compares the answer with the reference using the upstream judge prompt. Only a "correct" verdict passes. The default is `openai/gpt-5.1-2025-11-13`, the judge used in the paper. |

The verifiers in `normalization.py` and `verifiers.py` are ported from the upstream `browsergym-timewarp` package at commit `4978e69`, and the upstream unit tests are ported to `tests/test_verifiers.py`. Before release, both implementations scored 4,122 answer/task pairs: 18 answers for each of the 229 deterministically scored goals. The answers included the gold answer, reformatted golds, distractors and other goals' golds. The two implementations agreed on every pair.

Metrics:

- `accuracy`: the success rate.
- `stderr`.
- Success rate per UI version (`v1` to `v6`).
- Success rate per category (`wiki`, `news`, `shop`, `multi`).

## Agent

The default solver, `browser_agent()`, follows the BrowserGym protocol used in the paper. Its tools come from [Playwright MCP](https://github.com/microsoft/playwright-mcp) v0.0.82, which runs inside the sandbox through `mcp_server_sandbox()`:

- `browser_navigate`, `browser_navigate_back`
- `browser_click`, `browser_type`, `browser_select_option`, `browser_hover`, `browser_press_key`
- `browser_tabs`
- `browser_handle_dialog`, `browser_wait_for`
- `submit()`, for sending the final message to the user.

The loop works as follows:

- The start page is opened before the first step, and the agent sees its accessibility snapshot.
- A step executes at most one action. If a turn contains several tool calls, only the first runs; the others get an error result.
- A turn with no tool call, or with unparseable arguments, is retried within the same step, up to 4 times. This mirrors AgentLab's GenericAgent, which retries unparseable responses up to 4 times. The step is used up once the retries run out.
- After every action, including a failed one, the loop calls Playwright MCP's `browser_snapshot` and adds the accessibility snapshot of the current page as a separate message. The snapshot tool itself is not offered to the model. The browser viewport is 1280x720, as in the paper.
- Only the latest page observation is sent to the model. Earlier observations (and screenshots) are replaced with a short note; the agent's own messages and its action results are kept. This matches the paper's prompt, which shows the current observation plus the agent's previous responses, and it keeps long episodes within small context windows. The full observations stay in the transcript.

## Differences from the paper

The paper runs agents through BrowserGym and AgentLab. This implementation keeps the following unchanged:

- the environments;
- the goals;
- the step budget, including up to 4 retries per step;
- the observation history (current page plus previous responses);
- the one-action-per-step rule;
- the verifiers.

It changes the agent interface:

1. **Observations and actions.** The paper's text agents observe HTML or BrowserGym's flattened accessibility tree (AXTree, with BrowserGym element ids), and act through BrowserGym's action set. Here the agent observes Playwright's ARIA snapshot and acts through the Playwright MCP tools above.
   - The ARIA snapshot is built by Playwright, not by serializing BrowserGym's tree, so element ids, visibility flags and formatting differ.
   - The snapshot covers the whole page rather than the viewport, so there is no `scroll` action.
   - HTML and Set-of-Marks observations are not available.
   - `-T screenshots=true` adds a screenshot tool instead of attaching a screenshot to every observation.
2. **Prompt.** The agent prompt is shorter than the paper's GenericAgent-style prompt. It asks for brief reasoning before each action but does not use the `<think>`/`<plan>`/`<memory>`/`<action>` tags; actions are native tool calls. It keeps the paper's extra instructions: the three site URLs, only navigate within TimeWarp, one action per response, and follow the task instructions.
3. **Final answer.** `submit()` replaces BrowserGym's `send_msg_to_user`. `report_infeasible` is not provided; the paper notes it can be removed during benchmarking.
4. **Leaving the sites.** Upstream scores 0 when a tab is on an unauthorized domain. Here, Playwright MCP's `--allowed-origins` blocks requests to anything other than the three sites and the CDNs their pages use.
5. **Runs and sampling.** The paper averages 3 runs; pass `--epochs 3` to do the same. The paper does not report sampling settings, but the upstream benchmark scripts serve models through vLLM with temperature 0.01 and up to 8192 new tokens. This task leaves sampling at model defaults; pass `--temperature 0.01 --max-tokens 8192` to match the upstream scripts.

## Reference results

The paper evaluates open-weight models zero-shot on the test split, averaged over 3 runs. Success rates (%) with AXTree observations (Tables 1 and 4 of arXiv:2603.04949v3):

| Model | Wiki | News | Shop | Multi | Overall |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct-2507 | 20.3 | 25.3 | 18.1 | 18.6 | 20.4 |
| Qwen3-4B-Thinking-2507 | 26.9 | 31.1 | 28.8 | 29.0 | 28.8 |
| Llama-3.1-8B-Instruct | 2.3 | 0.0 | 0.0 | 0.0 | 0.7 |
| Qwen3-VL-8B-Instruct | 19.0 | 32.8 | 18.1 | 24.9 | 23.0 |
| gemma-3-12b-it | 15.4 | 27.8 | 13.2 | 19.1 | 18.3 |

These models can be run through Inspect's vLLM provider, for example:

```bash
uv run inspect eval timewarp/timewarp --model vllm/Qwen/Qwen3-4B-Instruct-2507 \
  -M enable_auto_tool_choice=true -M tool_call_parser=hermes \
  --epochs 3 --temperature 0.01 --max-tokens 8192
```

`-M` arguments are passed to `vllm serve`. The agent uses native tool calls, which vLLM only parses with a model-specific tool-call parser; see vLLM's [tool calling docs](https://docs.vllm.ai/en/stable/features/tool_calling.html) for the parser that matches each model.

Because of the interface differences above, results from this implementation are not expected to match these numbers exactly.

## Development

```bash
uv sync
uv run pytest                      # unit tests; downloads the pinned task file
uv run pytest --docker -m docker   # end-to-end test that builds and runs the sandbox
uv run ruff check && uv run mypy
uv run inspect-evals-lint --all
```

The environment image installs its Python packages from hashed lockfiles. To change them, edit `src/timewarp/sandbox/env/requirements.in` and regenerate both lockfiles:

```bash
cd src/timewarp/sandbox/env
for arch in x86_64 aarch64; do
  uv pip compile requirements.in --python-version 3.11 --python-platform ${arch}-manylinux_2_28 \
    --generate-hashes --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    --no-header -o requirements-${arch}.txt
done
```

## Citation

```bibtex
@inproceedings{ishmam2026timewarp,
  title         = {{TimeWarp}: Evaluating Web Agents by Revisiting the Past},
  author        = {Ishmam, Md Farhan and Marino, Kenneth},
  booktitle     = {Fortieth Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year          = {2026},
  eprint        = {2603.04949},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2603.04949}
}
```

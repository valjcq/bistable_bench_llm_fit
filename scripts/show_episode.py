"""Print one agent episode as Markdown, rebuilt from its run directory alone.

    uv run python scripts/show_episode.py runs/agent_eval/main/toy/<run-dir> [> episode.md]

For each turn: the reasoning, the text, the tool calls (raw arguments as the provider returned
them), the results exactly as the model received them and the budget after each; at the end the
claim, the justification and the checker's verdict. The system prompt, the tool specs, the
sampling parameters actually sent and the server preflight are shown once at the top. The same
rebuild (bench.agent.replay) is what the tests compare with the messages the backend received.
"""

import sys

from bench.agent.replay import render_markdown

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.stdout.write(render_markdown(sys.argv[1]))

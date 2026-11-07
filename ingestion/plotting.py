"""Plots for the RAG app: reranker ScoreAnswer and guardrail (Noul) check scores."""

import textwrap

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Guardrail chart palette (neutral/accent match plot_score_answer)
_NEUTRAL = "#72727e"   # check below its threshold
_ACCENT = "#d95fb5"    # check at/above its threshold (violation)
_INFO = "#5f9ed9"      # informational check (not a violation)

_STAGE_TITLES = {"input": "Input guardrail", "output": "Output guardrail"}


def plot_score_answer(
    answer,
    title: str = "Probability of each level",
    labels: dict | None = None,
    save_as: str | None = None,
) -> Figure:
    """Bars = probability per level, diamond = score.

    `answer` needs: .score, .legend, .probabilities, .confidence
    `labels` optionally maps level -> x-axis label (defaults to level + legend text).
    Returns the figure; the caller decides whether to show it (plt.show / st.pyplot).
    """
    levels = list(answer.legend)
    probs = [answer.probabilities[level] for level in levels]
    labels = labels or {level: f"{level}\n{answer.legend[level]}" for level in levels}

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(7, 3.5))

        # Bars + percentage labels
        ax.bar(levels, probs, color="#72727e", width=0.6)
        for x, p in zip(levels, probs):
            ax.text(x, p + 0.02, f"{p:.0%}", ha="center", fontsize=11)

        # Score marker on the x-axis
        ax.scatter(answer.score, 0, marker="D", color="#d95fb5", s=90, zorder=3, clip_on=False)

        # Header text: title (left), confidence + score (right)
        ax.set_title(title, loc="left", fontsize=12)
        fig.text(0.97, 0.93, f"Confidence\n{answer.confidence:.2f}", ha="right", va="top", fontsize=13)
        fig.text(0.97, 0.72, f"◆ score {answer.score:.2f}", ha="right", color="#d95fb5", fontsize=9)

        # Clean axes
        ax.set_xticks(levels, [labels[level] for level in levels])
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        fig.tight_layout(rect=(0, 0, 1, 0.82))

    if save_as:
        fig.savefig(save_as, dpi=150)
    return fig


def plot_guardrail_scores(
    report: dict,
    title: str = "Guardrail checks",
    save_as: str | None = None,
) -> Figure:
    """One bar per guardrail question (Noul score as a percentage).

    `report` needs: "scores" {question: float | None}; optional "thresholds"
    {question: float} and "informational" [question, ...].
    Pink bar = score at/above its threshold (violation); dashed tick = threshold;
    blue bar = informational check (never a violation); "n/a" = score unavailable.
    """
    scores = report["scores"]
    thresholds = report.get("thresholds", {})
    info = set(report.get("informational", []))
    keys = list(scores)
    labels = ["\n".join(textwrap.wrap(k.replace("_", " "), 13)) for k in keys]

    def flagged(k: str) -> bool:
        s = scores[k]
        return s is not None and k not in info and s >= thresholds.get(k, 0.5)

    any_flagged = any(flagged(k) for k in keys)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(7, 3.5))

        for x, k in enumerate(keys):
            s = scores[k]
            if s is None:
                ax.text(x, 0.02, "n/a", ha="center", fontsize=11)
                continue
            color = _ACCENT if flagged(k) else _INFO if k in info else _NEUTRAL
            ax.bar(x, s, color=color, width=0.6)
            ax.text(x, s + 0.02, f"{s:.0%}", ha="center", fontsize=11)
            if k in thresholds:
                ax.hlines(thresholds[k], x - 0.38, x + 0.38, colors="white",
                          linestyles="--", linewidth=1, zorder=3)

        # Header text: title (left), verdict + legend (right)
        ax.set_title(title, loc="left", fontsize=12)
        missing = any(scores[k] is None and k not in info for k in keys)
        verdict = "Violation\nflagged" if any_flagged else "Score\nmissing" if missing else "All\nclear"
        fig.text(0.97, 0.93, verdict, ha="right", va="top", fontsize=13,
                 color=_ACCENT if (any_flagged or missing) else "white")
        fig.text(0.97, 0.72, "- - threshold", ha="right", fontsize=9)

        # Clean axes
        ax.set_xticks(range(len(keys)), labels)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        fig.tight_layout(rect=(0, 0, 1, 0.82))

    if save_as:
        fig.savefig(save_as, dpi=150)
    return fig


def plot_guardrail_stages(guardrails: dict) -> dict[str, Figure]:
    """One figure per guardrail stage, e.g. {"input": fig, "output": fig}.

    `guardrails` is the chain's result["guardrails"]: {stage: report}.
    """
    return {
        stage: plot_guardrail_scores(report, title=_STAGE_TITLES.get(stage, stage.title()))
        for stage, report in guardrails.items()
    }


if __name__ == "__main__":
    from types import SimpleNamespace

    demo = SimpleNamespace(
        score=1.27,
        legend={0: "High", 1: "Medium", 2: "Low"},
        probabilities={0: 0.05, 1: 0.63, 2: 0.32},
        confidence=0.46,
    )
    plot_score_answer(demo, title="Ticket priority", save_as="priority_plot.png")

    demo_guard = {
        "input": {
            "scores": {"off_topic": 0.02, "impersonation_or_override": 0.97, "malicious_or_sensitive": 0.99},
            "thresholds": {"off_topic": 0.5, "impersonation_or_override": 0.5, "malicious_or_sensitive": 0.5},
            "informational": [],
        },
    }
    for stage, fig in plot_guardrail_stages(demo_guard).items():
        fig.savefig(f"guardrail_{stage}.png", dpi=150)
    plt.show()
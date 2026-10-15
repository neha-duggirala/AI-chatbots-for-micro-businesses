"""Plot a ScoreAnswer (probability per level + overall score)."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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


if __name__ == "__main__":
    from types import SimpleNamespace

    demo = SimpleNamespace(
        score=1.27,
        legend={0: "High", 1: "Medium", 2: "Low"},
        probabilities={0: 0.05, 1: 0.63, 2: 0.32},
        confidence=0.46,
    )
    plot_score_answer(demo, title="Ticket priority", save_as="priority_plot.png")
    plt.show()
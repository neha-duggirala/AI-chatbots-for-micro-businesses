import matplotlib.pyplot as plt


def plot_score_answer(answer, title="Probability of each level", save_as=None):
    """Plot a ScoreAnswer: bars = probability per level, diamond = score.

    `answer` needs: .score, .legend, .probabilities, .confidence
    (works with ScoreAnswer from your classifier response).
    """
    levels = list(answer.legend)                       # [0, 1, 2]
    probs = [answer.probabilities[l] for l in levels]  # [0.05, 0.63, 0.32]
    labels = [f"{l}\n{answer.legend[l]}" for l in levels]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Bars + percentage labels
    ax.bar(levels, probs, color="#72727e", width=0.6)
    for x, p in zip(levels, probs):
        ax.text(x, p + 0.02, f"{p:.0%}", ha="center", fontsize=11)

    # Score marker on the x-axis
    ax.scatter(answer.score, 0, marker="D", color="#d95fb5", s=90, zorder=3,
               clip_on=False)

    # Header text: title (left), confidence + score (right)
    ax.set_title(title, loc="left", fontsize=12)
    fig.text(0.97, 0.93, f"Confidence\n{answer.confidence:.2f}",
             ha="right", va="top", fontsize=13)
    fig.text(0.97, 0.72, f"◆ score {answer.score:.2f}",
             ha="right", color="#d95fb5", fontsize=9)

    # Clean axes
    ax.set_xticks(levels, labels)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.82))
    if save_as:
        fig.savefig(save_as, dpi=150)
    plt.show()


if __name__ == "__main__":
    # Demo with the values from your response.
    # With your real response, use:
    #   answer = response.answers["priority"]
    from types import SimpleNamespace

    answer = SimpleNamespace(
        score=1.27,
        legend={0: "High", 1: "Medium", 2: "Low"},
        probabilities={0: 0.05, 1: 0.63, 2: 0.32},
        confidence=0.46,
    )
    plot_score_answer(answer, title="Ticket priority", save_as="priority_plot.png")
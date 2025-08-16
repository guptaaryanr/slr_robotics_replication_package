import pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import ast
import textwrap

# Consistent, cleaner styling across figures
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

df = pd.read_csv("extraction_table_data.csv")


def bar(col):
    s = df[col]

    if col == "techniques":
        # parse rows that are list-like and explode into one row per item
        def to_list(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                items = x
            else:
                try:
                    items = ast.literal_eval(x)
                    if not isinstance(items, list):
                        items = [str(items)]
                except Exception:
                    items = [i.strip() for i in str(x).split(",")]
            items = [" ".join(str(i).split()) for i in items if str(i).strip()]
            return list(dict.fromkeys(items))

        s = s.apply(to_list).explode().dropna()

    # counts
    vc = s.value_counts()

    # --- Option B: vertical bars, wrapped labels, tidy look ---
    plt.figure(figsize=(10, 5))
    ax = vc.plot.bar()

    # See current labels (before any changes)
    raw_labels = [str(lbl) for lbl in vc.index]
    print("Current x-axis labels:", raw_labels)

    # Optional: rename some labels (case-insensitive mapping)
    mapping = {
        "composite objective including energy": "Composite Objective incl. Energy",
        "aggregate energy/power": "Aggregate Energy/Power",
        "relative change versus baseline": "Relative Change vs. Baseline",
        "task-normalized energy": "Task-Normalized Energy",
        "performance per energy": "Performance per Energy",
        "validation statistics": "Validation Stats",
        "physics-based integrals": "Physics-Based Integrals",
        "other": "Other",
        "industrial": "Industrial",
        "robot exploration": "Robot Exploration",
        "service or domestic": "Service/Domestic",
        "swarm or multi-robot": "Swarm/Multi-Robot",
        "aerial": "Aerial",
        "iot power": "IoT Power",
        "additive manufacturing": "Additive Manufacturing",
        "modular": "Modular",
        "motors and actuators": "Motors/Actuators",
        "computing and controllers": "Computing/Controllers",
        "sensors": "Sensors",
        "communication subsystem": "Communication Subsystem",
        "mechanical motion pattern": "Mechanical Motion Pattern",
        "battery and power electronics": "Battery/Power Electronics",
        "simulation": "Simulation",
        "hybrid": "Hybrid",
        "physical": "Physical",
        "representational": "Representational",
        "abstract": "Abstract",
        "motion and trajectory optimization": "Motion & Trajectory Optimization",
        "learning or predictive optimization": "Learning/Predictive Optimization",
        "computation allocation and scheduling": "Computation Allocation & Scheduling",
        "power management and idle control": "Power Management & Idle Control",
        "communication and data efficiency": "Communication & Data Efficiency",
        "performance vs energy": "Performance vs. Energy",
        "accuracy vs energy": "Accuracy vs. Energy",
        "stability vs energy": "Stability vs. Energy",
    }
    renamed = [mapping.get(lbl.strip().lower(), lbl) for lbl in raw_labels]

    # Wrap long labels across lines instead of rotating 90°
    wrapped = [textwrap.fill(lbl, width=18) for lbl in renamed]
    ax.set_xticklabels(wrapped, rotation=0, ha="center")

    # Cleaner axes
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if "qa" in col:
        temp = col.split("_")
        ax.set_xlabel("QA Trade-offs")
        ax.set_ylabel("Papers")
        plt.title(f"Distribution of QA Trade-offs in Papers")
    else:
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Papers")
        plt.title(f"Distribution of {col.replace('_', " ").title()} in Papers")

    # Value labels above each bar
    for rect in ax.patches:
        y = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2
        ax.annotate(
            f"{int(y)}",
            (x, y),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(f"fig_{col}.png", dpi=200, bbox_inches="tight")
    plt.clf()


for col in [
    "metric",
    "domain",
    "major_consumers",
    "evaluation_type",
    "energy_model",
    "techniques",
    "qa_tradeoff",
]:
    bar(col)

"""
Phase 2: Analyze detection session data.

Usage:
    uv run python src/analyze_session.py [session_file]

If no file is given, uses the most recent session.
"""

import sys
import glob
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_session(filepath):
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def print_summary(df):
    duration = df["timestamp"].max() - df["timestamp"].min()
    minutes = duration.total_seconds() / 60
    events = df["timestamp"].nunique()

    print(f"Session duration: {minutes:.1f} minutes")
    print(f"Log events: {events}")
    print(f"Total detections: {len(df)}")
    print()

    print("Object frequency:")
    counts = df["object_class"].value_counts()
    for cls, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")
    print()

    # Person presence
    person_events = df[df["object_class"] == "person"]["timestamp"].nunique()
    print(f"You were detected in {person_events}/{events} log events ({person_events/events*100:.0f}% of the time)")


def plot_timeline(df):
    """Show which objects were detected over time."""
    fig, ax = plt.subplots(figsize=(14, 5))

    classes = df["object_class"].unique()
    colors = plt.cm.Set2(range(len(classes)))
    class_colors = dict(zip(classes, colors))

    for i, cls in enumerate(classes):
        cls_df = df[df["object_class"] == cls]
        timestamps = cls_df["timestamp"].drop_duplicates().sort_values()
        ax.scatter(timestamps, [cls] * len(timestamps),
                   c=[class_colors[cls]], s=8, alpha=0.6, label=cls)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time")
    ax.set_title("Object Detection Timeline")
    plt.tight_layout()
    plt.savefig("data/timeline.png", dpi=150)
    print("Saved: data/timeline.png")
    plt.close()


def plot_presence(df):
    """Show per-minute presence of each object class."""
    df_min = df.copy()
    df_min["minute"] = df_min["timestamp"].dt.floor("min")

    pivot = df_min.groupby(["minute", "object_class"]).size().unstack(fill_value=0)
    presence = (pivot > 0).astype(int)

    fig, ax = plt.subplots(figsize=(14, 5))
    # Use string labels for x-axis to avoid matplotlib date issues
    labels = [t.strftime("%H:%M") for t in presence.index]
    for col in presence.columns:
        ax.fill_between(range(len(labels)), presence[col], alpha=0.4, label=col)
    ax.set_xticks(range(0, len(labels), max(1, len(labels) // 10)))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), max(1, len(labels) // 10))],
                       rotation=45)
    ax.set_xlabel("Time")
    ax.set_ylabel("Detected (1=yes)")
    ax.set_title("Per-Minute Object Presence")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("data/presence.png", dpi=150)
    print("Saved: data/presence.png")
    plt.close()


def plot_confidence(df):
    """Box plot of confidence scores per object class."""
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = df["object_class"].value_counts().index.tolist()
    data = [df[df["object_class"] == cls]["confidence"].values for cls in classes]
    ax.boxplot(data, tick_labels=classes)
    ax.set_ylabel("Confidence")
    ax.set_title("Detection Confidence by Object Class")
    plt.tight_layout()
    plt.savefig("data/confidence.png", dpi=150)
    print("Saved: data/confidence.png")
    plt.close()


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        files = sorted(glob.glob("data/detections/session_*.csv"))
        filepath = files[-1]

    print(f"Analyzing: {filepath}")
    print("=" * 50)

    df = load_session(filepath)
    print_summary(df)

    print("Generating charts...")
    plot_timeline(df)
    plot_presence(df)
    plot_confidence(df)
    print("Done.")


if __name__ == "__main__":
    main()

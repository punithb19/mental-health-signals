#!/usr/bin/env python3
"""
Add synthetic training examples to balance rare intents and high concern.
Outputs a CSV (Post, Tag, Concern_Level) that can be merged into the raw dataset.

Usage:
  python scripts/add_synthetic_data.py --output data/raw/final/synthetic_examples.csv
  Then: python scripts/merge_labels.py --new data/raw/final/synthetic_examples.csv --create-splits --config configs/data.yaml
"""

import argparse
import csv
import sys
from pathlib import Path


# Synthetic examples: (Post, Tag, Concern_Level)
# Short, clearly labeled, safe. Critical Risk uses vague help-seeking framing only.
SYNTHETIC_ROWS = [
    # Mood Tracking (under-represented)
    ("Today I felt calm in the morning but anxious by evening. Just logging how I feel.", "Mood Tracking", "low"),
    ("This week my mood has been up and down. Some days I feel okay, others really low.", "Mood Tracking", "medium"),
    ("Logging: felt disconnected most of the day. Slightly better after a walk.", "Mood Tracking", "low"),
    ("My mood has been stable this week. Keeping a log helps me notice patterns.", "Mood Tracking, Positive Coping", "low"),
    ("Feeling really low today. Writing it down so I can talk about it in therapy.", "Mood Tracking, Mental Distress", "medium"),
    ("Mood log: anxious in the morning, calmer after lunch. Still tired.", "Mood Tracking", "low"),
    ("I've been tracking my mood for a month. I notice I feel worse on Sundays.", "Mood Tracking, Progress Update", "low"),
    ("Today I felt hopeless for a few hours then it lifted. Not sure why.", "Mood Tracking, Mental Distress", "medium"),
    ("Just checking in: feeling numb today. Hard to describe.", "Mood Tracking", "medium"),
    ("Mood diary: slept badly, felt irritable all day. Tomorrow might be better.", "Mood Tracking, Cause of Distress", "medium"),
    # Maladaptive Coping (under-represented)
    ("I've been isolating myself when I get stressed instead of reaching out to friends.", "Maladaptive Coping", "medium"),
    ("When I feel bad I just scroll for hours. I know it's not helping.", "Maladaptive Coping, Mental Distress", "medium"),
    ("I've been drinking more when I'm anxious. I want to find better ways to cope.", "Maladaptive Coping, Seeking Help", "medium"),
    ("I shut down and avoid everyone when I'm overwhelmed. It makes things worse.", "Maladaptive Coping, Mental Distress", "medium"),
    ("I've been skipping meals when I'm stressed. I know it's not healthy.", "Maladaptive Coping", "medium"),
    ("I isolate and binge watch when I'm low. Looking for healthier coping strategies.", "Maladaptive Coping, Seeking Help", "medium"),
    ("I've been avoiding my problems by staying in bed. Need to change this.", "Maladaptive Coping, Progress Update", "medium"),
    ("When I'm anxious I tend to snap at people. I don't want to be like this.", "Maladaptive Coping, Mental Distress", "medium"),
    ("I've been using food to cope with stress. Want to find other outlets.", "Maladaptive Coping, Seeking Help", "medium"),
    ("I withdraw from everyone when I'm struggling. It's a pattern I want to break.", "Maladaptive Coping", "medium"),
    # Critical Risk - vague, help-seeking only
    ("I've been having dark thoughts lately and don't know who to talk to. I don't want to act on them.", "Critical Risk, Seeking Help", "high"),
    ("Sometimes I think about not being here. I'm scared and want to get help.", "Critical Risk, Mental Distress, Seeking Help", "high"),
    ("I've been in a really bad place and had some scary thoughts. Looking for support.", "Critical Risk, Seeking Help", "high"),
    ("I'm struggling with thoughts I don't want to have. I need to talk to someone.", "Critical Risk, Seeking Help", "high"),
    ("I've been having a hard time and my thoughts have been really dark. I want help.", "Critical Risk, Mental Distress, Seeking Help", "high"),
    ("I'm in crisis and don't know where to turn. I don't want to hurt myself.", "Critical Risk, Seeking Help", "high"),
    ("My thoughts have been going to a bad place. I'm reaching out because I need support.", "Critical Risk, Seeking Help", "high"),
    ("I've been having thoughts that scare me. I want to get better, not worse.", "Critical Risk, Mental Distress, Seeking Help", "high"),
    # High concern (general)
    ("I feel like I'm falling apart. I can't sleep, can't focus, and everything feels hopeless.", "Mental Distress", "high"),
    ("I've been in a really dark place for weeks. I'm barely functioning.", "Mental Distress", "high"),
    ("I'm struggling so much right now. I feel like I'm drowning and no one sees it.", "Mental Distress, Cause of Distress", "high"),
    ("Everything feels overwhelming. I don't know how much longer I can keep going like this.", "Mental Distress, Seeking Help", "high"),
    ("I've hit a really low point. I need to talk to someone before this gets worse.", "Mental Distress, Seeking Help", "high"),
    ("I'm in a bad place mentally and it's affecting everything. I want to get help.", "Mental Distress, Seeking Help", "high"),
    ("I've been having a really hard time. My anxiety and depression are at an all-time high.", "Mental Distress, Progress Update", "high"),
    ("I feel like I'm at the end of my rope. I need support.", "Mental Distress, Seeking Help", "high"),
    ("I'm struggling with severe anxiety and don't know how to cope. Everything feels like too much.", "Mental Distress, Cause of Distress", "high"),
    ("I've been in crisis mode for a while. I need someone to talk to.", "Mental Distress, Seeking Help", "high"),
    # Seeking Help
    ("Does anyone have advice for dealing with anxiety before job interviews?", "Seeking Help, Cause of Distress", "medium"),
    ("I'm looking for resources for therapy in my area. Can anyone point me in the right direction?", "Seeking Help", "low"),
    ("How do you cope with bad days? I'm looking for strategies that have worked for others.", "Seeking Help, Mental Distress", "medium"),
    ("I need to find a support group. Has anyone had a good experience with one?", "Seeking Help", "medium"),
    ("What has helped you when you feel completely hopeless? I'm open to suggestions.", "Seeking Help, Mental Distress", "medium"),
    # Progress Update
    ("I've been in therapy for three months now. It's slow but I notice small improvements.", "Progress Update, Positive Coping", "low"),
    ("Update: I started medication two weeks ago. Too early to tell but I'm hopeful.", "Progress Update", "low"),
    ("I've been doing better since I started exercising regularly. Just wanted to share.", "Progress Update, Positive Coping", "low"),
    ("Therapy has been hard but I'm starting to understand my patterns better.", "Progress Update, Mental Distress", "medium"),
    ("I've been sober for a month. It's been really difficult but I'm trying.", "Progress Update, Maladaptive Coping", "medium"),
    # Positive Coping
    ("I went for a long walk today when I felt anxious. It actually helped.", "Positive Coping, Mood Tracking", "low"),
    ("I've been trying meditation in the mornings. Still learning but it's something.", "Positive Coping", "low"),
    ("I reached out to a friend today instead of isolating. Small win.", "Positive Coping", "low"),
    ("I've been journaling when I feel overwhelmed. It doesn't fix everything but it helps.", "Positive Coping, Mood Tracking", "low"),
    ("I started going to the gym. It's hard to stay consistent but I feel better when I do.", "Positive Coping, Progress Update", "low"),
    # Cause of Distress
    ("Work stress is really getting to me. I don't know how to switch off.", "Cause of Distress, Mental Distress", "medium"),
    ("My relationship ending has left me in a really bad place.", "Cause of Distress, Mental Distress", "medium"),
    ("Financial worries are affecting my sleep and mood every day.", "Cause of Distress", "medium"),
    ("Family conflict has been triggering my anxiety a lot lately.", "Cause of Distress, Mental Distress", "medium"),
    ("The pressure from school is making my depression worse. I need to find a balance.", "Cause of Distress, Mental Distress", "medium"),
    # Multi-label variety
    ("I've been feeling down about my job search. Looking for advice and just needed to vent.", "Cause of Distress, Mental Distress, Seeking Help", "medium"),
    ("I'm in a better place than last year. Still have bad days but therapy and exercise help.", "Progress Update, Positive Coping, Mental Distress", "low"),
    ("I isolate when I'm stressed and I know it's not helping. Want to learn better coping.", "Maladaptive Coping, Mental Distress, Seeking Help", "medium"),
    ("Today was rough. Logging it here. Tomorrow I'll try to get outside.", "Mood Tracking, Positive Coping", "low"),
    ("I've been having a hard time with my mental health. Just started looking for a therapist.", "Mental Distress, Seeking Help, Progress Update", "medium"),
]


def main():
    ap = argparse.ArgumentParser(description="Write synthetic training examples CSV.")
    ap.add_argument(
        "--output",
        default="data/raw/final/synthetic_examples.csv",
        help="Output CSV path (Post, Tag, Concern_Level).",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Post", "Tag", "Concern_Level"])
        for post, tag, concern in SYNTHETIC_ROWS:
            w.writerow([post, tag, concern])

    print(f"Wrote {len(SYNTHETIC_ROWS)} synthetic examples -> {out_path}")
    print("Merge into raw data and regenerate splits:")
    print(f"  python scripts/merge_labels.py --new {out_path} --create-splits --config configs/data.yaml")


if __name__ == "__main__":
    main()

"""Shared helpers for DisfluencyLab (used by the notebook AND app.py)."""
import html
import re

import numpy as np

FILLERS = {"um", "uh", "er", "erm", "ah", "hmm", "mm", "eh", "uhm", "umm"}


def clean_transcript_markup(text):
    """Conservative markup cleanup. Does not remove words."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_wer(text):
    """Lowercase + strip punctuation. Keeps fillers and repetitions (verbatim)."""
    text = clean_transcript_markup(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def collapse_repeats(words, max_n=3):
    """Collapse immediate repeats of 1..max_n word units: 'the the ball' -> 'the ball'."""
    changed = True
    while changed:
        changed = False
        for n in range(max_n, 0, -1):
            out, i = [], 0
            while i < len(words):
                if i + 2 * n <= len(words) and words[i:i + n] == words[i + n:i + 2 * n]:
                    out.extend(words[i:i + n])
                    i += 2 * n
                    changed = True
                else:
                    out.append(words[i])
                    i += 1
            words = out
    return words


def _strip_fragments(text):
    # stutter prefixes: "b-b-ball" -> "ball", "wa-wa-want" -> "want"
    text = re.sub(r"\b([a-z]{1,3})-(?:\1-)*(?=\1)", " ", text)
    # trailing broken word: "wa- " -> removed
    text = re.sub(r"\b[a-z']+-(?=\s|$)", " ", text)
    return text


def make_clean_reference(text):
    """Fluent 'intended text': drop fillers, fragments and immediate repetitions.
    Heuristic: legitimate repeats ('that that') are also collapsed. Inspect the output."""
    t = _strip_fragments(clean_transcript_markup(text).lower())
    words = [w for w in normalize_for_wer(t).split() if w not in FILLERS]
    return " ".join(collapse_repeats(words))


def count_disfluencies(text):
    """Rough counts of annotated disfluencies in a verbatim reference."""
    t = clean_transcript_markup(text).lower()
    words = normalize_for_wer(t).split()
    fillers = sum(w in FILLERS for w in words)
    fragments = len(re.findall(r"\b[a-z']+-(?=\s|$|[a-z])", t))
    content = [w for w in words if w not in FILLERS]
    repeats = len(content) - len(collapse_repeats(content))
    return {
        "words": len(words),
        "fillers": fillers,
        "fragments": fragments,
        "repeated_words": repeats,
        "disfluency_rate": round((fillers + fragments + repeats) / max(len(words), 1), 3),
    }


def detect_blocks(y, sr, min_gap=0.4, top_db=35):
    """Candidate blocks / long pauses: silent gaps >= min_gap s BETWEEN speech
    segments. Normal pauses also appear -> verify by listening."""
    import librosa
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    gaps = []
    for (_, e0), (s1, _) in zip(intervals[:-1], intervals[1:]):
        if (s1 - e0) / sr >= min_gap:
            gaps.append((e0 / sr, s1 / sr))
    return gaps


def wer_breakdown(reference_norm, hypothesis_raw):
    """WER with S/D/I/H. `reference_norm` must already be normalised;
    the hypothesis is normalised here."""
    from jiwer import process_words
    hyp = normalize_for_wer(hypothesis_raw)
    if not reference_norm:
        return dict(WER=np.nan, S=0, D=0, I=0, H=0, N=0)
    if not hyp:  # jiwer dislikes empty hypotheses -> all deletions
        n = len(reference_norm.split())
        return dict(WER=1.0, S=0, D=n, I=0, H=0, N=n)
    o = process_words(reference_norm, hyp)
    return dict(WER=o.wer, S=o.substitutions, D=o.deletions, I=o.insertions,
                H=o.hits, N=o.hits + o.substitutions + o.deletions)


def alignment_html(reference_norm, hypothesis_raw):
    """Colour-coded word alignment (substitution / deletion / insertion)."""
    from jiwer import process_words
    hyp = normalize_for_wer(hypothesis_raw)
    if not reference_norm or not hyp:
        return "<i>Nothing to align.</i>"
    o = process_words(reference_norm, hyp)
    ref, hy = o.references[0], o.hypotheses[0]
    parts = []
    for c in o.alignments[0]:
        r, h = ref[c.ref_start_idx:c.ref_end_idx], hy[c.hyp_start_idx:c.hyp_end_idx]
        if c.type == "equal":
            parts.append(" ".join(r))
        elif c.type == "substitute":
            parts.append(f"<span style='background:rgba(255,170,0,.30);padding:0 3px;border-radius:3px'>"
                         f"<s>{' '.join(r)}</s>&rarr;{' '.join(h)}</span>")
        elif c.type == "delete":
            parts.append(f"<span style='background:rgba(255,70,70,.30);padding:0 3px;border-radius:3px'>"
                         f"<s>{' '.join(r)}</s></span>")
        elif c.type == "insert":
            parts.append(f"<span style='background:rgba(60,200,90,.30);padding:0 3px;border-radius:3px'>"
                         f"+{' '.join(h)}</span>")
    return " ".join(parts)

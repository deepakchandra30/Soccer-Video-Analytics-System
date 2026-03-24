"""Narrative quality evaluation using BLEU score."""
import json
import os

from src.narratives.schemas import MatchNarrative


def compute_bleu(reference_text, generated_text):
    """Compute BLEU score between reference and generated text.

    Uses nltk's sentence_bleu with smoothing for short texts.
    Returns float in [0, 1].
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    ref_tokens = reference_text.lower().split()
    gen_tokens = generated_text.lower().split()

    if not ref_tokens or not gen_tokens:
        return 0.0

    smoother = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoother)


def evaluate_narrative(narrative, reference_summary=None):
    """Evaluate a generated narrative. Returns metrics dict."""
    metrics = {
        "num_key_moments": len(narrative.key_moments),
        "num_player_contributions": len(narrative.player_contributions),
        "num_tactical_points": len(narrative.tactical_breakdown),
        "summary_length_words": len(narrative.match_summary.split()),
    }

    if reference_summary:
        metrics["bleu_score"] = compute_bleu(reference_summary,
                                              narrative.match_summary)

    return metrics


def evaluate_batch(narratives, references=None):
    """Evaluate multiple narratives. Returns list of metric dicts."""
    results = []
    for i, narr in enumerate(narratives):
        ref = references[i] if references and i < len(references) else None
        results.append(evaluate_narrative(narr, ref))
    return results


def save_evaluation(results, output_path):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

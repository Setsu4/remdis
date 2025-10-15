import difflib
from typing import List, Dict, Any, Optional, Tuple


def normalize(s: str) -> str:
    # very light normalization
    return s.strip().lower().replace('　', ' ').replace('\n', ' ')


def best_subplan_match(user_text: str, subplans: List[Dict[str, Any]], threshold: float) -> Optional[Tuple[Dict[str, Any], float]]:
    user = normalize(user_text)
    best = None
    best_score = 0.0
    for sp in subplans:
        q = normalize(sp.get('user_question', ''))
        if not q:
            continue
        score = difflib.SequenceMatcher(a=user, b=q).ratio()
        if score > best_score:
            best_score = score
            best = sp
        elif score == best_score and best is not None:
            # tie-breaker: closeness of length, then keep earlier
            if abs(len(user) - len(q)) < abs(len(user) - len(normalize(best.get('user_question', '')))):
                best = sp
    if best is None:
        return None
    if best_score < threshold:
        return None
    return best, best_score

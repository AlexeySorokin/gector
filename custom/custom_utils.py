from hunspell import Hunspell
from Levenshtein import distance

hunspeller = Hunspell()


def should_be_lowercased(s, lowercase_oov=True):
    try:
        return s[1:].islower() and (hunspeller.spell(s.lower()) if hunspeller.spell(s) else lowercase_oov)
    except UnicodeEncodeError:
        return s[1:].islower()


def apply_simple_edit(words, start, end, variant):
    if start < 0:
        return words
    left_context, right_context = words[:start], words[end:]
    if end == 0 and should_be_lowercased(words[end]):
        right_context[0] = right_context[0].lower()
    variant_words = variant.split()
    if start == 0 and len(words) > 0 and words[0][0].isupper():
        if len(variant_words) > 0 and variant_words[0].islower():
            variant_words[0] = variant_words[0].title()
    new_words = left_context + variant_words + right_context
    return new_words


def find_edit_splits(start, end, candidate):
    if isinstance(candidate, str):
        candidate = candidate.split()
    answer = []
    if end - start == 0 and len(candidate) > 1:  # 0 -> K
        answer.append([(start, start, word) for word in candidate])
    elif len(candidate) == 0 and end-start >= 2:  # 0 -> K
        answer.append([(r, r+1, "") for r in range(start, end)])
    elif end - start == len(candidate):
        answer.append([(r, r+1, word) for r, word in enumerate(candidate, start)])
    elif end == start + 1 and len(candidate) == 2:
        first, second = candidate
        answer.append([(start, start, first), (start, end, second)])
        answer.append([(start, end, first), (end, end, second)])
    elif end == start + 2 and len(candidate) == 1:
        answer.append([(start, start+1, candidate[0]), (start+1, end, "")])
        answer.append([(start, start+1, ""), (start+1, end, candidate[0])])
    return answer
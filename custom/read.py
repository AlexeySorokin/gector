from collections import defaultdict
from typing import Optional, List

from dataclasses import dataclass
from tqdm.auto import tqdm


@dataclass
class Edit:
    start: int
    end: int
    candidate: str
    label: Optional[str]
    annotator: Optional[int] = 0

    def __str__(self):
        return "|||".join([
            f"{self.start} {self.end}", self.label, self.candidate, "REQUIRED", "-NONE-", str(self.annotator)
        ])


def apply_edits(sent: List[str], edits: List[Edit]):
    new_sent_words = sent[:]
    reverse_edit_data = [Edit(0, 0, "", None)]
    for i, word in enumerate(sent):
        reverse_edit_data.append(Edit(i, i + 1, word, None))
        reverse_edit_data.append(Edit(i + 1, i + 1, "", None))
    for edit in sorted(edits, key=(lambda x: (-x.end, -x.start))):
        new_sent_words[edit.start:edit.end] = edit.candidate.split()
        reverse_edit = Edit(edit.start, edit.end, " ".join(sent[edit.start:edit.end]), edit.label)
        words_number = len(edit.candidate.split())
        reverse_edits = [reverse_edit] * (2 * words_number + 1)
        reverse_edit_data[2 * edit.start:2 * edit.end + 1] = reverse_edits
    return {"sent": new_sent_words, "edits": reverse_edit_data}

def dump_sentence_annotation(curr_data):
    annotators_number = max(curr_data["edits"]) + 1
    curr_edits = [curr_data["edits"][i] for i in range(annotators_number)]
    # noinspection PyTypeChecker
    curr_corrections = [
        apply_edits(curr_data["source"], annotator_edits)["sent"] for annotator_edits in curr_edits
    ]
    answer = {
        "source": curr_data["source"], "correct": curr_corrections, "edits": curr_edits
    }
    return answer

def read_m2_simple(infile, n=None):
    answer = []
    curr_data = {"edits": defaultdict(list, {0: []})}
    with open(infile, "r", encoding="utf8") as fin:
        for line in tqdm(fin):
            if n is not None and len(answer) >= n:
                break
            line = line.strip()
            if line == "":
                if "source" in curr_data:
                    answer.append(dump_sentence_annotation(curr_data))
                    curr_data = {"edits": defaultdict(list, {0: []})}
                continue
            if line[:2] not in ["A ", "S "]:
                continue
            mode, line = line[0], line[1:].strip()
            if mode == "S":
                if "source" in curr_data:
                    answer.append(dump_sentence_annotation(curr_data))
                    curr_data = {"edits": defaultdict(list, {0: []})}
                curr_data["source"] = line.split()
            else:
                splitted = line.split("|||")
                start, end = map(int, splitted[0].split())
                edit_type, correction, annotator = splitted[1], splitted[2], int(splitted[-1])
                curr_data["edits"][annotator].append(Edit(start, end, correction, edit_type, annotator))
    if "source" in curr_data:
        answer.append(dump_sentence_annotation(curr_data))
    return answer

def read_test_file(infile, n=None):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line != "":
                answer.append(line.split())
            if n is not None and len(answer) == n:
                break
    return answer
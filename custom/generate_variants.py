import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import bisect
import json

import numpy as np
from tqdm.auto import tqdm

from custom.extract_edits import EditExtractor
from custom.read import read_m2_simple, read_test_file
from custom.spacy_utils import SpacyModelWrapper
from custom.custom_utils import apply_simple_edit, find_edit_splits
from gector.gec_model import GecBERTModel

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-i", "--infile", default="data/datasets/dev.bea.m2")
argument_parser.add_argument("-r", "--raw", action="store_true")
argument_parser.add_argument("-m", "--model_path", default="data/roberta_1_gector.th")
argument_parser.add_argument("-v", "--vocab_path", default="data/output_vocabulary")
argument_parser.add_argument("-S", "--split_to_variants", action="store_false")
argument_parser.add_argument("-t", "--threshold", type=float, default=-2.0)
argument_parser.add_argument("-o", "--outfile", type=str, default=None)
argument_parser.add_argument("-n", "--n_sents", type=int, default=None)
argument_parser.add_argument("-l", "--min_length", type=int, default=3)
argument_parser.add_argument("-L", "--max_length", type=int, default=50)
argument_parser.add_argument("-V", "--verbose", type=str, default=None)
argument_parser.add_argument("-c", "--chunk_size", default=100000, type=int)
argument_parser.add_argument("-b", "--batch_size", default=32, type=int)
argument_parser.add_argument("-a", "--annotator", default=0, type=int)


def evaluate_edits(edits, sents, data):
    exact, by_word, total = 0, 0, 0
    block_starts = [0] + list(np.cumsum([len(x) for x in sents]))
    for i, sent in enumerate(data):
        offsets = [0] + list(np.cumsum([len(x) for x in sents[i]]))
        for edit in sent["edits"][0]:
            if edit.start < 0:
                continue
            sent_index = bisect.bisect_right(offsets, edit.start) - 1
            start, end = edit.start - offsets[sent_index], edit.end - offsets[sent_index]
            key = (start, end, edit.candidate)
            sent_edits = edits[block_starts[i] + sent_index]
            if key in sent_edits:
                exact += 1
            else:
                splitted = edit.candidate.split()
                if end - start == len(splitted):
                    if all((r, r + 1, x) in sent_edits for r, x in enumerate(splitted, start)):
                        by_word += 1
                elif end - start == 0:
                    if all((start, start, x) in sent_edits for x in splitted):
                        by_word += 1
            total += 1
    print(exact, by_word, total)

def make_edit_data(edits, sents, data, split_to_variants=True, annotator=0):
    answer = [{"words": " ".join(sent), "edits": []} for sent in sents]
    has_edits_in_sent = [False] * len(sents)
    offsets = [0] + [int(x) for x in np.cumsum([len(x) for x in sents])]
    correct_edits = [set() for _ in sents]
    if "edits" in data:
        curr_annotator = annotator if annotator < len(data["edits"]) else 0
        for edit in data["edits"][curr_annotator]:
            if edit.start < 0:
                continue
            sent_index = bisect.bisect_right(offsets[:-1], edit.start) - 1
            has_edits_in_sent[sent_index] = True
            start, end = edit.start - offsets[sent_index], edit.end - offsets[sent_index]
            key = (start, end, edit.candidate)
            status, diff = "not_found", -np.inf
            if key in edits[sent_index]:
                correct_edits[sent_index].add(key)
                status = "correct"
            if split_to_variants:
                possible_splits = find_edit_splits(start, end, edit.candidate)
                for variant in possible_splits:
                    if all(key in edits[sent_index] for key in variant):
                        correct_edits[sent_index].update(variant)
                        diff = max(sum(edits[sent_index][key] for key in variant), diff)
                        if status == "not_found":
                            status = "partial"
            if status != "correct":
                curr_edit_data = {
                    "source": " ".join(sents[sent_index][start:end]),
                    "start": int(start), "end": int(end), "target": edit.candidate,
                    "words": apply_simple_edit(sents[sent_index], start, end, edit.candidate),
                    "diff": None if status == "not_found" else diff, "is_correct": True,
                    "is_generated": (status == "partial")
                }
                assert curr_edit_data["diff"] != -np.inf
                answer[sent_index]["edits"].append(curr_edit_data)
    
    for i, (sent_edits, sent, offset) in enumerate(zip(edits, sents, offsets)):
        for (start, end, target), diff in sent_edits.items():
            curr_edit_data = {
                "source": " ".join(sent[start:end]),
                "start": int(start), "end": int(end), "target": target,
                "words": apply_simple_edit(sent, start, end, target),
                "diff": diff,
                "is_generated": True
            }
            if "edits" in data:
                curr_edit_data["is_correct"] = ((start, end, target) in correct_edits[i])
            answer[i]["edits"].append(curr_edit_data)
        default_edit = {
            "source": "", "start": -1, "end": -1, "target": None,
            "words": sent, "diff": 0.0, "is_generated": True
        }
        if "edits" in data:
            default_edit["is_correct"] = not has_edits_in_sent[i]
        answer[i]["edits"].append(default_edit)
    return answer

if __name__ == "__main__":
    args = argument_parser.parse_args()
    if not args.raw:
        data = read_m2_simple(args.infile, n=args.n_sents)
    else:
        data = [{"source": sent} for sent in read_test_file(args.infile, n=args.n_sents)]
    print(len(data), end=" ")
    data = [sent for sent in data if args.min_length <= len(sent["source"]) <= args.max_length]
    print(len(data))
    gector = GecBERTModel(vocab_path=args.vocab_path, model_paths=[args.model_path],
                          min_error_probability=0.5, special_tokens_fix=True)
    model = EditExtractor(gector=gector, threshold=args.threshold, remove_initial_lowercase=False)
    # processing with spacy
    spacy_model = SpacyModelWrapper(disable=["lemmatizer", "tagger", "parser", "ner"])
    total, not_found = 0, 0
    if args.outfile is not None:
        with open(args.outfile, "w", encoding="utf8") as fout:
            pass
    if not args.raw and args.verbose is not None:
        with open(args.verbose, "w", encoding="utf8") as fout:
            pass
    for start in tqdm(range(0, len(data), args.chunk_size)):
        curr_data = data[start:start+args.chunk_size]
        sents = spacy_model.pipe([" ".join(x["source"]) for x in curr_data], to_conll=False, to_words=True)
        # obtain the candidate edits and evaluate
        joint_sents = [" ".join(sent) for elem in sents for sent in elem]
        extracted_edits = model(joint_sents, batch_size=args.batch_size)
        # evaluate_edits(extracted_edits, sents, data)
        block_starts = [0] + list(np.cumsum([len(x) for x in sents]))
        extracted_edits = [extracted_edits[start:end] for start, end in zip(block_starts[:-1], block_starts[1:])]
        # dump the candidate edits
        edit_data = [make_edit_data(*elem, split_to_variants=args.split_to_variants, annotator=args.annotator)
                     for elem in zip(extracted_edits, sents, curr_data)]
        if not args.raw:
            not_found += sum([int(edit["is_correct"] and not edit["is_generated"])
                            for elem in edit_data for sent in elem for edit in sent["edits"]])
            total += sum(len(elem["edits"][args.annotator if args.annotator < len(elem["edits"]) else 0])
                         for elem in curr_data)
            print(total-not_found, total, f"{100*(1-not_found/total):.2f}")
        if args.outfile is not None:
            with open(args.outfile, "a", encoding="utf8") as fout:
                for curr_answer in edit_data:
                    json.dump(curr_answer, fout)
                    fout.write("\n")
        if not args.raw and args.verbose is not None:
            with open(args.verbose, "a", encoding="utf8") as fout:
                for curr_answer, curr_sents, curr_data in zip(edit_data, sents, curr_data):
                    words = " ".join(curr_data["source"])
                    if len(curr_sents) > 1:
                        print(words, file=fout)
                    for sent_answer in curr_answer:
                        print(sent_answer["words"], file=fout)
                        for edit in sent_answer["edits"]:
                            source = edit["source"] or "_"
                            is_generated = "OK" if edit["is_generated"] else "error"
                            is_correct = "OK" if edit["is_correct"] else "error"
                            print(edit["start"], edit["end"], f"{source}->{edit['target']}",
                                  f"generated={is_generated}", f"correct={is_correct}", file=fout)
                    print("", file=fout)
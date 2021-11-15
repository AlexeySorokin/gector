from collections import defaultdict

import numpy as np
import torch
from allennlp.nn import util
from tqdm.auto import tqdm

from gector.gec_model import GecBERTModel
from utils.helpers import UNK, apply_reverse_transformation


def action_to_target(token, action):
    start, end, label, _ = action
    if label == "":
        return ""
    elif start == end:
        return label.replace("$APPEND_", "")
    elif label.startswith("$TRANSFORM_"):
        word = apply_reverse_transformation(token, label)
        return token if word is None else word
    elif start == end - 1:
        return label.replace("$REPLACE_", "")
#     elif label.startswith("$MERGE_"):
#         target_tokens[target_pos + 1: target_pos + 1] = [label]
#         shift_idx += 1
    return token


class EditExtractor:
    
    def __init__(self, gector: GecBERTModel, threshold=-2.0, max_iter=3,
                 max_span_actions=3, n_max=20, n_min=3, join_consecutive_edits=True,
                 remove_initial_lowercase=True):
        self.gector = gector
        self.model = gector.models[0]
        self.threshold = threshold
        self.max_iter = max_iter
        self.max_span_actions = max_span_actions
        self.n_max = n_max
        self.n_min = n_min
        self.join_consecutive_edits = join_consecutive_edits
        self.remove_initial_lowercase = remove_initial_lowercase
    
    def _extract_sent_edits(self, log_diffs, probs, tokens):
        answer = dict()
        log_diffs[len(tokens) + 1:] = -np.inf
        n = max(min((log_diffs >= self.threshold).int().sum(), self.n_max), self.n_min)
        M = log_diffs.shape[1]
        values, indexes = torch.topk(log_diffs.view(-1), k=n)
        indexes = indexes.cpu().numpy()
        positions, labels = indexes // M, indexes % M + 1
        span_counts = defaultdict(int)
        for pos, label, value in zip(positions, labels, values):
            sugg_token = self.gector.vocab.get_token_from_index(label, namespace='labels')
            if sugg_token == UNK:
                continue
            # `token`=None and `prob`=1.0 are just dummy arguments
            action = self.gector.get_token_action(None, pos, 1.0, sugg_token)
            if "MERGE" in action[2]:
                continue
            source = tokens[action[0]] if 0 <= action[0]< len(tokens) else ""
            target = action_to_target(source, action)
            key = (action[0], action[1], target)
            if (action[1] == action[0] + 1) and target == source:
                # for vocabulary tokens equal to current word
                continue
            if key not in answer and span_counts[key[:2]] < self.max_span_actions:
                answer[key] = log_diffs[pos, label-1].item() # (log_diffs[pos, label-1], probs[pos, label])
                span_counts[key[:2]] += 1
        return answer
        
    def _extract_edits(self, predictions, texts):
        log_probs = torch.log_softmax(predictions["logits_labels"], dim=-1)
        probs = predictions['class_probabilities_labels']
        log_diffs = log_probs[:, :, 1:] - log_probs[:, :, :1]
        answer = [self._extract_sent_edits(*elem) for elem in zip(log_diffs, probs, texts)]
        return answer
    
    def _update_edit_indexing(self, indexing, edits, keep_probs):
        actual_word_indexes = [i for i, (_, label, _) in enumerate(indexing) if label != ""] + [len(indexing)]
        for edit in sorted(edits, key=lambda x: (x[1], x[0]), reverse=True):
            start, end, target, prob = edit
            if target is None:
                continue
            diff = np.log(prob) - np.log(keep_probs[end])
            pos = actual_word_indexes[start]
            if start == end:  # insertion
                indexing = indexing[:pos] + [(-1, target, diff)] + indexing[pos:]
            else:  # replacement or deletion
                source_pos, prev_target, _ = indexing[pos]
                if target.startswith('$TRANSFORM_'):
                    target = apply_reverse_transformation(prev_target, target)
                if self.remove_initial_lowercase and source_pos == 0 and target == prev_target.lower():
                    continue
                if target is None:
                    continue
                if target != "" or source_pos >= 0:
                    indexing = indexing[:pos]  + [(source_pos, target, diff)] + indexing[pos+1:]
                else:
                    indexing = indexing[:pos] + indexing[pos+1:]
        return indexing
    
    def _add_edits_from_indexing(self, edits, indexing):
        last_insertion, last_diff, new_edits = [], 0.0, dict()
        last_word_index = max(elem[0] for elem in indexing) + 1
        for i, label, diff in indexing:
            if i >= 0:
                if len(last_insertion) > 0:
                    key = (i, i, " ".join(last_insertion))
                    new_edits[key] = last_diff
                    last_insertion, last_diff = [], 0.0
                if diff is not None:
                    key = (i, i+1, label)
                    new_edits[key] = diff
            else:
                last_insertion.append(label)
                last_diff += diff
        if len(last_insertion) > 0:
            key = (last_word_index, last_word_index, " ".join(last_insertion))
            new_edits[key] = last_diff
        def get_action_type(i, label, diff):
            return "delete" if label == "" else "insert" if i == -1 else "replace" if diff is not None else None
        for (i, first_label, first_diff), (j, second_label, second_diff) in zip(indexing[:-1], indexing[1:]):
            first_type = get_action_type(i, first_label, first_diff)
            second_type = get_action_type(j, second_label, second_diff)
            if first_type == "delete" and second_type == "replace" and i >= 0:
                key = (i, j+1, second_label)
            elif first_type == "replace" and second_type == "delete" and j >= 0:
                key = (i, j + 1, first_label)
            elif first_type == "insert" and second_type == "replace":
                key = (j, j+1, f"{first_label} {second_label}")
            elif first_type == "replace" and second_type == "insert":
                key = (i, i+1, f"{first_label} {second_label}")
            elif first_type == "replace" and second_type == "replace":
                key = (i, j+1, f"{first_label} {second_label}")
            else:
                continue
            new_edits[key] = first_diff + second_diff
        for key, diff in new_edits.items():
            edits[key] = diff
        return

    def __call__(self, texts, batch_size=32):
        answer = [None] * len(texts)
        text_words = [x.split() for x in texts]
        order = np.argsort([len(x) for x in text_words])[::-1]
        for start in tqdm(range(0, len(order), batch_size)):
            indexes = order[start:start+batch_size]
            batch = [text_words[i] for i in indexes]
            batch_answer = self.call_on_batch(batch)
            for index, elem in zip(indexes, batch_answer):
                answer[index] = elem
        return answer
    
    def _postprocess_edits(self, tokens, answer):
        # add transposition
        to_add = dict()
        for (start, end, target), diff in answer.items():
            if start == end and start + 1 < len(tokens) and target == tokens[start + 1] and target.isalpha():
                key = (start + 1, start + 2, "")
                if key in answer:
                    other_diff = answer[key]
                    swap = f"{tokens[start + 1]} {tokens[start]}"
                    to_add[(start, start + 2, swap)] = diff + other_diff  # (diff+other_diff, min(prob, other_prob))
        answer.update(to_add)
        # remove lowercase
        if self.remove_initial_lowercase:
            for key in list(answer.keys()):
                if key[0] == 0 and key[1] == 1 and key[2] == tokens[0].lower():
                    answer.pop(key)
        return
       
    def call_on_batch(self, text_words):
        final_words = text_words[:]
        prev_preds_dict = {i: [elem] for i, elem in enumerate(final_words)}
        pred_ids = list(range(len(text_words)))
        edit_indexing = [[(i, word, None) for i, word in enumerate(sent)] for sent in text_words]
        # final_edit_indexing = edit_indexing[:]
        for iteration in range(self.max_iter):
            # select active elements
            iter_words = [final_words[i] for i in pred_ids]
            iter_edit_indexing = [edit_indexing[i] for i in pred_ids]
            # do predictions
            batch = self.gector.preprocess(iter_words)[0]
            batch = util.move_to_device(batch.as_tensor_dict(), 0 if torch.cuda.is_available() else -1)
            with torch.no_grad():
                predictions = self.model.forward(**batch)
            if iteration == 0:
                edits = self._extract_edits(predictions, text_words)
            preds, ids, error_probs, keep_probs = self.gector._convert([predictions], return_keep_probs=True)
            pred_words, pred_edits = self.gector.postprocess_batch(
                iter_words, preds, ids, error_probs, return_edits=True
            )
            iter_edit_indexing = [self._update_edit_indexing(*elem)
                                  for elem in zip(iter_edit_indexing, pred_edits, keep_probs)]
            # EXTRACTING NEW EDITS
            for i, curr_edit_indexing in zip(pred_ids, iter_edit_indexing):
                self._add_edits_from_indexing(edits[i], curr_edit_indexing)
                edit_indexing[i] = curr_edit_indexing
            # EXTRACTING NEW EDITS
            final_words, pred_ids, cnt = \
                self.gector.update_final_batch(final_words, pred_ids, pred_words, prev_preds_dict)
            if not pred_ids:
                break
        for elem in zip(text_words, edits):
            self._postprocess_edits(*elem)
        return edits
        
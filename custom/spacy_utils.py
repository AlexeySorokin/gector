import itertools
from io import StringIO

import spacy
from spacy.tokens import Doc, Span
from udapi.core.node import ListOfNodes
from udapi.core.root import Root
from udapi.block.read.conllu import Conllu


class WhitespaceTokenizer:
    
    def __init__(self, vocab):
        self.vocab = vocab
    
    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False
        
        return Doc(self.vocab, words=words, spaces=spaces)


def spacy_to_conllu_features(tag):
    return "_"


def spacy_to_conllu(parse: Span, noun_is_head: bool = True):
    sent_answer = []
    if isinstance(parse, Doc):
        # сделать деление по nltk??
        return [spacy_to_conllu(sent, noun_is_head=noun_is_head) for sent in parse.sents]
    if noun_is_head:
        for i, elem in enumerate(parse):
            if elem.dep_ == "pobj" and elem.head.pos_ == "ADP":
                prev_head = elem.head
                elem.dep_ = "obl" if prev_head.dep_ != "ROOT" else "root"
                elem.head = prev_head.head if prev_head.dep_ != "ROOT" else elem
                prev_head.head = elem
                prev_head.dep_ = "case"
    new_head = None
    for i, elem in enumerate(parse):
        head = "_" if elem.dep_ is None else 0 if elem.head.i == elem.i else elem.head.i - parse.start + 1
        if (head <= 0 or head > len(parse)):
            new_head = i
            if elem.dep_ != "punct":
                break
    for i, elem in enumerate(parse):
        head = "_" if elem.dep_ is None else 0 if elem.head.i == elem.i else elem.head.i - parse.start + 1
        deprel = "_" if elem.dep_ is None else "root" if elem.dep_ == "ROOT" else elem.dep_
        if (head <= 0 or head > len(parse)):
            if i == new_head:
                head, deprel = 0, "root"
            else:
                head = new_head + 1
        node = {
            "ord": i + 1, "word": elem.text, "lemma": elem.lemma_ if elem.lemma_ != "-PRON-" else elem.text,
            "upos": elem.pos_, "xpos": elem.tag_, "feats": spacy_to_conllu_features(elem.tag_),
            "deprel": deprel, "head": head
        }
        sent_answer.append(
            "{ord}\t{word}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t_\t_".format(**node)
        )
    return "\n".join(sent_answer)


class SpacyModelWrapper:
    
    def __init__(self, model="en_core_web_sm", disable=None, tokenized=True, noun_is_head=True):
        self.tokenized = tokenized
        self.noun_is_head = noun_is_head
        disable = disable or ["parser", "ner"]
        self.model = spacy.load(model, disable=disable)
        self.model.add_pipe(self.model.create_pipe('sentencizer'))
        if self.tokenized:
            self.model.tokenizer = WhitespaceTokenizer(self.model.vocab)
    
    def _to_conll_tree(self, parse):
        conllu_parse = spacy_to_conllu(parse, noun_is_head=self.noun_is_head)
        return [Conllu(filehandle=StringIO(sent)).read_tree() for sent in conllu_parse]
    
    def __call__(self, sent, to_conll=True, to_words=False):
        spacy_parse = self.model(sent)
        if to_conll:
            return self._to_conll_tree(spacy_parse)
        elif to_words:
            return [[x.text for x in sent] for sent in spacy_parse.sents]
        return spacy_parse
    
    def pipe(self, sents, to_conll=True, to_words=False):
        spacy_parses = list(self.model.pipe(sents))
        if to_conll:
            answer = [self._to_conll_tree(parse) for parse in spacy_parses]
        elif to_words:
            answer = [[[x.text for x in sent] for sent in parse.sents] for parse in spacy_parses]
        else:
            answer = spacy_parses
        return answer
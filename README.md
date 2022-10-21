# GECToR-based candidate generator

This repository contains the extension of the GECToR model for Grammatical Error Correction. It yields multiple
candidate corrections instead of a single hypothesis. This allows to use this model as candidate generator 
on the first stage of Generate-then-Rerank approach to GEC. 
In particular, it is used in [EditScorer](https://github.com/AlexeySorokin/EditScorer), described in
> Improved grammatical error correction by ranking elementary edits <br>
> [Alexey Sorokin](https://github.com/AlexeySorokin)

The original GECToR model was described in
> [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://arxiv.org/abs/2005.12592) <br>
> [Kostiantyn Omelianchuk](https://github.com/komelianchuk), [Vitaliy Atrasevych](https://github.com/atrasevych), [Artem Chernodub](https://github.com/achernodub), [Oleksandr Skurzhanskyi](https://github.com/skurzhanskyi) <br>
> Grammarly <br>
> [15th Workshop on Innovative Use of NLP for Building Educational Applications (co-located with ACL 2020)](https://sig-edu.org/bea/current) <br>

## Installation and usage
To install the package, simply run:
```.bash
pip install -r requirements.txt
```
The project was tested using Python 3.7.

We refer the user to the original GECToR repository for the complete documentation. Here we describe only our contribution,
the candidate generator.

### Candidate generator

Our extension outputs all candidate edits whose probability is not less than P(KEEP) + &#920;, where KEEP is the
"do nothing" operation and &#920; is the predefined threshold (-3.0 in the examples below).

To produce the output for the BEA2019 train, dev and test file, run the following commands:
```shell
python custom/generate_variants.py -t -3.0 -o TRAIN_OUTPUT_PATH -i DATA_DIR/wi+locness/m2/ABC.train.gold.bea19.m2 -S
python custom/generate_variants.py -t -3.0 -o DEV_OUTPUT_PATH -i DATA_DIR/wi+locness/m2/ABCN.dev.gold.bea19.m2 -l 1 -L 1000 -S
python custom/generate_variants.py -t -3.0 -o TEST_OUTPUT_PATH -i DATA_DIR/wi+locness/test/ABCN.test.bea19.orig -l 1 -L 1000 -S -r
```
Here `DATA_DIR` is where you unzip the archive with [BEA2019 data](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data).
Command line options mean the following:
* `-t`: the threshold &#920; defined above;
* `-S`: whether to split multiword edits to shorter elementary parts;
* `-r`: whether the input is a raw file;
* `-l`: minimal length of a sentence to be processed;
* `-L`: maximal length of a sentence to be processed.

## Citation
If you find this work is useful for your research, please cite our paper:
```
@inproceedings{sorokin-2022-improving,
    title = "Improved Grammatical Error Correction by Ranking Elementary Edits",
    author = "Alexey Sorokin",
    year = "2022",
    comment = "To appear in EMNLP 2022 Proceedings"
}
```

Also cite the original GECToR work:
```
@inproceedings{omelianchuk-etal-2020-gector,
    title = "{GECT}o{R} {--} Grammatical Error Correction: Tag, Not Rewrite",
    author = "Omelianchuk, Kostiantyn  and
      Atrasevych, Vitaliy  and
      Chernodub, Artem  and
      Skurzhanskyi, Oleksandr",
    booktitle = "Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    month = jul,
    year = "2020",
    address = "Seattle, WA, USA â†’ Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bea-1.16",
    pages = "163--170",
    abstract = "In this paper, we present a simple and efficient GEC sequence tagger using a Transformer encoder. Our system is pre-trained on synthetic data and then fine-tuned in two stages: first on errorful corpora, and second on a combination of errorful and error-free parallel corpora. We design custom token-level transformations to map input tokens to target corrections. Our best single-model/ensemble GEC tagger achieves an F{\_}0.5 of 65.3/66.5 on CONLL-2014 (test) and F{\_}0.5 of 72.4/73.6 on BEA-2019 (test). Its inference speed is up to 10 times as fast as a Transformer-based seq2seq GEC system.",
}
```

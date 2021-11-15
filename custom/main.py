from custom.extract_edits import EditExtractor
from gector.gec_model import GecBERTModel


if __name__ == "__main__":
    texts = [
        "Because nowdays whereever you go there is big queue and I hate being waited ."
    ]
    model_path = "data/roberta_1_gector.th"
    vocab_path = "data/output_vocabulary"
    gector = GecBERTModel(vocab_path=vocab_path, model_paths=[model_path],
                          min_error_probability=0.5, special_tokens_fix=True)
    model = EditExtractor(gector=gector, remove_initial_lowercase=False, threshold=-5.0)
    answer = model(texts)
    for curr_answer, text in zip(answer, texts):
        print(text)
        for elem in curr_answer.items():
            print("{} {} {} {:.2f}".format(*elem[0], elem[1]))
        print("")
import json
from utils import viterbi
from tqdm import trange, tqdm


class Evaluator:
    def __init__(self, eval_file, args):
        self.eval_file = eval_file
        self.sentences = []
        self.pos_maps = []
        self.results = []
        self.init_vec_path = args.save_path + 'init-vector.json'
        self.transition_path = args.save_path + 'transition.json'
        self.emission_path = args.save_path + 'emission.json'
        self.pos = set()
        self.total = 0

        self.precision = None
        self.recall = None
        self.f1 = None

        print("---Loading Evaluation File---")
        with open(self.eval_file, encoding='utf-8') as f:
            self.eval_sents = json.load(f)
            for sent in self.eval_sents:
                sentence = []
                pos_map = []
                for w, p in sent:
                    sentence.append(w)
                    pos_map.append(p)
                    self.pos.add(p)
                self.sentences.append(sentence)
                self.total += len(pos_map)
                self.pos_maps.append(pos_map)
        self._predict()

    def _predict(self):
        print("--Start Prediction---")
        with open(self.init_vec_path, encoding='utf-8') as f:
            init_vec = json.load(f)
        with open(self.transition_path, encoding='utf-8') as f:
            transition = json.load(f)
        with open(self.emission_path, encoding='utf-8') as f:
            emission = json.load(f)

        for sentence in tqdm(self.sentences, desc="Predicting"):
            path = viterbi(init_vec, transition, emission, sentence)
            # print(sentence)
            # print(path)
            self.results.append(path)

    def _cal_precision(self):
        correct = 0
        for pos_map, result in zip(self.pos_maps, self.results):
            for p, p_pred in zip(pos_map, result):
                if p == p_pred:
                    correct += 1
        self.precision = correct / self.total

    def _cal_recall(self):
        recall = 0.0
        for p in self.pos:
            p_len = 0
            tp = 0
            fn = 0
            for pos_map, result in zip(self.pos_maps, self.results):
                for _p, p_pred in zip(pos_map, result):
                    if _p == p:
                        # p_len += 1
                        if _p == p_pred:
                            tp += 1
                        else:
                            fn += 1
            recall += ((tp / (tp + fn)) / len(self.pos))
        self.recall = recall

    def _cal_f1(self):
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

    def get_precision(self):
        self._cal_precision()
        return self.precision

    def get_recall(self):
        self._cal_recall()
        return self.recall

    def get_f1(self):
        self._cal_f1()
        return self.f1




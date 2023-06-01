import json
from collections import defaultdict


def _freq2prob(d: defaultdict) -> dict:
    """
    :param d: defaultdict(int)
    :return prob_dict: dict
    """
    prob_dict = {}
    sum_freq = sum(d.values())
    for p, freq in d.items():
        prob_dict[p] = freq / sum_freq
    return prob_dict


class HiddenMarkovModelTrainer():
    def __init__(self, train_file: str, save_path: str):
        # JSON file for training
        self.init_vector = {}
        self.transition = {}
        self.emission = {}
        self.train_file = train_file
        self.save_path = save_path
        self.trans_freq = {}
        self.emit_freq = {}

        # Vocabulary-Vocabulary Dictionary
        self.vocab = defaultdict(int)

        # Part Of Speech Set
        self.pos = set()

        # Word Frequency of Minimal Reserved Word
        self.MIN_COUNT = 1

        # Load train set
        print("---Loading Training Dataset---")
        with open(self.train_file, encoding='utf-8') as f:
            self.train_sents = json.load(f)
            for sent in self.train_sents:
                for w, p in sent:
                    self.vocab[w] += 1
                    self.pos.add(p)

    def _count_freq(self):
        self.vocab = [w[0] for w in self.vocab.items() if w[1] >= self.MIN_COUNT]

        self.init_freq = defaultdict(int)

        for p in self.pos:
            self.trans_freq[p] = defaultdict(int)

        for p in self.pos:
            self.emit_freq[p] = defaultdict(int)
        print("---File Loaded---")

    def train(self):
        self._count_freq()
        print("---Starting Training---")

        for sent in self.train_sents:
            # Learning the initial probability
            # sent[0][1] is the first vocabulary of the sentence
            # sent[][]: (vocab, pos)
            self.init_freq[sent[0][1]] += 1

            # states_transition is a list of state tuple
            # states_transition records the transition process of POS
            states_transition = [(p1[1], p2[1]) for p1, p2 in zip(sent, sent[1:])]

            # Learning transition probability
            for p1, p2 in states_transition:
                self.trans_freq[p1][p2] += 1

            # Learning emission probability
            for w, p in sent:
                self.emit_freq[p][w] += 1

        # Setting 0 to those states transition tuple and emission tuple that not appear
        for p1 in self.pos:
            for p2 in self.pos:
                if p2 not in self.trans_freq[p1]:
                    self.trans_freq[p1][p2] = 0

        for p in self.pos:
            for v in self.vocab:
                if v not in self.emit_freq[p]:
                    self.emit_freq[p][v] = 0

        self.init_vector = _freq2prob(self.init_freq)

        for p, freq_dict in self.trans_freq.items():
            self.transition[p] = _freq2prob(freq_dict)

        for p, freq_dict in self.emit_freq.items():
            self.emission[p] = _freq2prob(freq_dict)

        print("---Training Finished---")
        print("---Saving Model Parameters---")

        with open(self.save_path + 'init-vector.json', 'w', encoding='utf-8') as f:
            json.dump(self.init_vector, f, indent=2, ensure_ascii=False)

        with open(self.save_path + 'transition.json', 'w', encoding='utf-8') as f:
            json.dump(self.transition, f, indent=2, ensure_ascii=False)

        with open(self.save_path + 'emission.json', 'w', encoding='utf-8') as f:
            json.dump(self.emission, f, indent=2, ensure_ascii=False)

        print("---Saved---")
        return self.init_vector, self.transition, self.emission

from collections import defaultdict


def viterbi(init_vec, transition, emission, observers: list):
    viterbi_matrix = defaultdict(dict)
    backpointers_matrix = defaultdict(dict)

    N = len(transition)
    T = len(observers)
    states = transition.keys()

    # Initializing 2 matrix
    for s in states:
        if observers[0] in emission[s]:
            viterbi_matrix[0][s] = init_vec[s] * emission[s][observers[0]]
        else:
            viterbi_matrix[0][s] = 0
        backpointers_matrix[0][s] = '<s>'

    def argmax(t, s):
        """
        Calculate the viterbi variable of state s in time t
        :param t: int, time t
        :param s: state s
        :return: max_prob, argmax_pre_state
        """
        max_prob, argmax_pre_state = 0, 0
        for i in states:
            p = viterbi_matrix[t - 1][i] * transition[i][s] * emission[s][observers[t]]
            if p > max_prob:
                max_prob = p
                argmax_pre_state = i
        return max_prob, argmax_pre_state

    for t in range(1, T):
        for s in states:
            max_prob, arg_max_state = argmax(t, s)
            viterbi_matrix[t][s] = max_prob
            backpointers_matrix[t][s] = arg_max_state

    max_prob_final_state, max_prob = None, 0
    for s in states:
        if viterbi_matrix[T - 1][s] > max_prob:
            max_prob_final_state = s
            max_prob = viterbi_matrix[T - 1][s]

    # Best recurse path
    best_path = [max_prob_final_state]
    for t in range(T - 1, 0, -1):
        try:
            pre_state = backpointers_matrix[t][best_path[-1]]
            best_path.append(pre_state)
        except:
            best_path.append(None)
    best_path = list(reversed(best_path))
    return best_path


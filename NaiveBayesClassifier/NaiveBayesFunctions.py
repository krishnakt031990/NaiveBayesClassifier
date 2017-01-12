__author__ = 'krishnateja'

import re
import math
from collections import defaultdict


def tokenizer(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)
    return set(all_words)


def count_words(training_set):
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenizer(message):
            counts[word][0 if is_spam else 1] += 1
        return counts


def probabilities(counts, total_spams, total_non_spams, k=0.5):
    return [(w, (spam + k) / (total_spams + 2 * k), (non_spam + k) / (total_spams + 2 * k)) for w, (spam, non_spam) in
            counts.iteritems()]


def spam_probability(word_probabilities, message):
    message_words = tokenizer(message)
    log_prb_spam = log_prb_not_spam = 0.0

    for word, word_prob_spam, word_prob_not_spam in word_probabilities:
        if message in message_words:
            log_prb_spam += math.log(word_prob_spam)
            log_prb_not_spam += math.log(word_prob_not_spam)
        else:
            log_prb_spam += math.log(1.0 - word_prob_spam)
            log_prb_not_spam += math.log(1.0 - word_prob_not_spam)

    prob_if_spam = math.exp(log_prb_spam)
    prob_if_not_spam = math.exp(log_prb_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

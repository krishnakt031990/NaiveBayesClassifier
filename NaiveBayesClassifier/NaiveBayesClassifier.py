__author__ = 'krishnateja'

from NaiveBayes.NaiveBayesFunctions import count_words, probabilities, spam_probability


class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probabilities = []

    def train(self, training_set):
        num_spams = len([is_spam for message, is_spam in training_set if is_spam])
        num_non_spams = len(training_set) - num_spams

        word_counts = count_words(training_set)
        self.word_probabilities = probabilities(word_counts, num_spams, num_non_spams)

    def classify(self, message):
        return spam_probability(self.word_probabilities, message)

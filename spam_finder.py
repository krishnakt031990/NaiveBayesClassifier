__author__ = 'krishnateja'

import glob
import re
from collections import Counter

from sklearn.cross_validation import train_test_split

from NaiveBayes import NaiveBayesClassifier

path_to_files = r"data/*/*"
data = []

for filename in glob.glob(path_to_files):
    is_spam = "ham" not in filename
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("Subject:"):
                subject_line = re.sub(r"^Subject: ", " ", line).strip()
                data.append((subject_line, is_spam))

train_data, test_data = train_test_split(data, test_size=0.75)

classifier = NaiveBayesClassifier.NaiveBayesClassifier()
classifier.train(train_data)

spam_probabilities = []

for message, is_spam in test_data:
    probability = classifier.classify(message)
    spam_probabilities += [(message, (is_spam, probability))]

print(spam_probabilities)

spam_messages = []

for _, (is_spam, probability) in spam_probabilities:
    if probability > 0.5:
        spam_messages += [(is_spam, probability)]

print(spam_messages)

true_positives = 0
false_positives = 0

for spam in spam_messages:
    if spam[0] == True:
        true_positives += 1
    else:
        false_positives += 1

print true_positives
print false_positives

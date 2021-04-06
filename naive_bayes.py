import numpy as np


class NaiveBayesClassifier:
    def __init__(self, opt):
        self.opt = opt
        self.alpha = 1
        self.vocab = set()
        self.bow, class_occ = {}, {}
        self.log_prior, self.log_likelihood = {}, {}

    def calc_log_likelihood(self, word_occ, total_occ):
        return np.log((word_occ + self.alpha) / (total_occ + self.alpha * len(self.vocab)))  # TODO: vocab?

    def init_bow(self, X_train, y_train):
        self.classes, self.class_count = np.unique(y_train, return_counts=True)  # calculate review numbers of classes
        self.bow = {label: {} for label in self.classes}  # create empty dictionary for words belonging to classes

        for review, label in zip(X_train, y_train):
            words = review.split(' ')
            for i in range(len(words)):
                if self.opt.feature == 'unigram':
                    ngram = words[i].lower()
                else:  # bigram
                    if i == len(words) - 1:
                        break
                    ngram = words[i].lower() + ' ' + words[i + 1].lower()

                if ngram not in self.bow[label]:
                    self.bow[label][ngram] = 1  # first occurrence of current word in current class
                else:
                    self.bow[label][ngram] += 1  # increase the count of current word in current class

                # TODO: search or current word?
                if ngram not in self.vocab:
                    self.vocab.add(ngram)  # update vocabulary

        # calculate occurrences of words of classes
        self.class_occ = {label: sum(self.bow[label][word] for word in self.bow[label]) for label in self.classes}

    def train(self, X_train, y_train):
        self.init_bow(X_train, y_train)
        sample_num = len(X_train)
        self.log_likelihood = {label: {} for label in self.classes}  # create empty dictionary for likelihoods

        for label, count in zip(self.classes, self.class_count):
            self.log_prior[label] = np.log(count / sample_num)  # calculate log prior for current class - P(y)

            for word in self.bow[label]:
                word_occ = self.bow[label][word]  # calculate log likelihood of current word given current class
                self.log_likelihood[label][word] = self.calc_log_likelihood(word_occ, self.class_occ[label])  # P(wi|y)

    def predict(self, test_sample):
        words = test_sample.split(' ')
        votes = {label: 0 for label in self.classes}

        for label in self.classes:
            votes[label] = self.log_prior[label]
            for i in range(len(words)):
                if self.opt.feature == 'unigram':
                    ngram = words[i].lower()
                else:  # bigram
                    if i == len(words) - 1:
                        break
                    ngram = words[i].lower() + ' ' + words[i + 1].lower()

                if ngram in self.bow[label]:
                    votes[label] += self.log_likelihood[label][ngram]
                else:  # unseen ngram
                    votes[label] += self.calc_log_likelihood(0, self.class_occ[label])

        return max(votes, key=votes.get)  # return the class with highest vote

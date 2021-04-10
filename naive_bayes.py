from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from string import punctuation
import numpy as np


class NaiveBayesClassifier:
    def __init__(self, opt):
        self.opt = opt
        self.alpha = 1
        self.vocab = set()
        self.bow, class_occ = {}, {}
        self.log_prior, self.log_likelihood = {}, {}
        self.class_reviews, self.class_vocab, self.class_tfids = {}, {}, {}

    def calc_log_likelihood(self, word_occ, total_occ):  # calculate log likelihood probability
        return np.log((word_occ + self.alpha) / (total_occ + self.alpha * len(self.vocab)))

    def process_review(self, string):
        if self.opt.discard_punct:  # if specified remove punctuations
            for punct in punctuation:
                string = string.replace(punct, '')
        return string.split()

    def inverse_document_frequency(self):
        for label in self.classes:
            pipe = Pipeline([('count', CountVectorizer(vocabulary=self.class_vocab[label])),
                             ('tfid', TfidfTransformer())]).fit(self.class_reviews[label])
            pipe['count'].transform(self.class_reviews[label]).toarray()
            self.class_tfids[label] = pipe['tfid'].idf_  # calculate TF-IDF scores for each class

    def init_bow(self, X_train, y_train):
        self.classes, self.class_count = np.unique(y_train, return_counts=True)  # calculate review numbers of classes
        self.bow = {label: {} for label in self.classes}  # create empty dictionary for words belonging to classes
        self.class_reviews = {label: [] for label in self.classes}  # create empty list for words belonging to classes
        self.class_vocab = {label: [] for label in self.classes}  # create empty list for words belonging to classes

        for review, label in zip(X_train, y_train):
            words = self.process_review(review)
            if self.opt.tfidf:
                self.class_reviews[label].append(review)

            for i in range(len(words)):
                if self.opt.discard_sw and words[i].lower() in ENGLISH_STOP_WORDS:  # if specified discard stop words
                    continue

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

                if ngram not in self.vocab:
                    self.vocab.add(ngram)  # update vocabulary

                if self.opt.tfidf and words[i].lower() not in self.class_vocab[label]:
                    self.class_vocab[label].append(words[i].lower())

        # calculate occurrences of words of classes
        self.class_occ = {label: sum(self.bow[label][word] for word in self.bow[label]) for label in self.classes}

    def train(self, X_train, y_train):
        self.init_bow(X_train, y_train)
        if self.opt.tfidf:  # if specified calculate TF-IDF scores
            self.inverse_document_frequency()

        sample_num = len(X_train)
        self.log_likelihood = {label: {} for label in self.classes}  # create empty dictionary for likelihoods

        for label, count in zip(self.classes, self.class_count):
            self.log_prior[label] = np.log(count / sample_num)  # calculate log prior for current class - P(y)

            for word in self.bow[label]:
                # calculate log likelihood of current word given current class - P(wi|y)
                if self.opt.tfidf:
                    word_occ = self.class_tfids[label][self.class_vocab[label].index(word)]
                    self.log_likelihood[label][word] = self.calc_log_likelihood(word_occ, sum(self.class_tfids[label]))
                else:
                    word_occ = self.bow[label][word]
                    self.log_likelihood[label][word] = self.calc_log_likelihood(word_occ, self.class_occ[label])

    def predict(self, test_sample):
        words = self.process_review(test_sample)  # get words in given test sample
        votes = {label: 0 for label in self.classes}

        for label in self.classes:
            votes[label] = self.log_prior[label]  # initialize vote with log prior probability
            for i in range(len(words)):
                if self.opt.feature == 'unigram':
                    ngram = words[i].lower()
                else:  # bigram
                    if i == len(words) - 1:
                        break
                    ngram = words[i].lower() + ' ' + words[i + 1].lower()

                # add log likelihood probability because of log domain
                if ngram in self.bow[label]:
                    votes[label] += self.log_likelihood[label][ngram]
                else:  # unseen ngram
                    votes[label] += self.calc_log_likelihood(0, self.class_occ[label])

        return max(votes, key=votes.get)  # return the class with highest vote

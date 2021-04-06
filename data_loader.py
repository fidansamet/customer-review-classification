import os
import sys
from sklearn.model_selection import train_test_split

TOPIC_LABEL = ['books', 'camera', 'dvd', 'health', 'music', 'software']
SENTIMENT_LABEL = ['pos', 'neg']
RANDOM_SEED = 42
TEST_SPLIT = 0.2


class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.topic_labels, self.sentiment_labels, self.reviews = [], [], []
        self.X_train, self.X_test = [], []
        self.y_train, self.y_test = [], []
        self.get_corpus()
        self.split_train_test_data()

    def get_corpus(self):
        if not os.path.exists(self.opt.dataroot):
            print("No data found in " + self.opt.dataroot)
            sys.exit()
        else:
            corpus_file = open(self.opt.dataroot)
            lines = corpus_file.readlines()
            for line in lines:
                space_split = line.split(' ', 3)  # split 3 spaces
                self.topic_labels.append(TOPIC_LABEL.index(space_split[0]))
                self.sentiment_labels.append(SENTIMENT_LABEL.index(space_split[1]))
                self.reviews.append(space_split[3])

    def split_train_test_data(self):
        if self.opt.category == 'sentiment':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.reviews, self.sentiment_labels,
                                                                                    test_size=TEST_SPLIT,
                                                                                    random_state=RANDOM_SEED)

        else:  # topic category
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.reviews, self.topic_labels,
                                                                                    test_size=TEST_SPLIT,
                                                                                    random_state=RANDOM_SEED)

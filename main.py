from options import Options
from data_loader import DataLoader, TOPIC_LABEL, SENTIMENT_LABEL
from naive_bayes import NaiveBayesClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import time
import csv


def validate(classifier, X_test, y_test):
    print("Validation started")
    time_start = time.clock()
    correct_classified = 0
    trues, preds = [], []

    for review, label in zip(X_test, y_test):
        pred = classifier.predict(review)
        trues.append(label)
        preds.append(pred)

        if pred == label:  # correct classification
            correct_classified += 1

    acc = 100 * (correct_classified / len(X_test))  # calculate the accuracy

    print("%d/%d samples are correctly classified - Accuracy: %0.2f" % (correct_classified, len(y_test), acc))
    print("Computation time: %0.2f second(s)" % (time.clock() - time_start))
    print("-------------------")

    plot_conf_matrix(trues, preds)


def test(classifier, X_test):
    print("Test started")
    time_start = time.clock()
    row_list, preds = [], []

    for review in X_test:
        pred = classifier.predict(review)
        preds.append(pred)

        row_list.append([len(row_list), SENTIMENT_LABEL[pred]])

    print("Computation time: %0.2f second(s)" % (time.clock() - time_start))
    print("-------------------")

    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Category"])
        writer.writerows(row_list)


def plot_conf_matrix(true, pred):
    conf_mat = metrics.confusion_matrix(true, pred)

    if opt.category == 'sentiment':
        df_cm = pd.DataFrame(conf_mat, SENTIMENT_LABEL, SENTIMENT_LABEL)
    else:
        df_cm = pd.DataFrame(conf_mat, TOPIC_LABEL, TOPIC_LABEL)

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap=plt.get_cmap('jet'))  # font size
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    opt = Options().parse()
    data_loader = DataLoader(opt)  # load data
    nb_classifier = NaiveBayesClassifier(opt)  # create Naive Bayes classifier
    nb_classifier.train(data_loader.X_train, data_loader.y_train)  # train classifier
    if opt.phase == 'train':
        validate(nb_classifier, data_loader.X_test, data_loader.y_test)  # test classifier
    else:
        test(nb_classifier, data_loader.X_test)  # test classifier

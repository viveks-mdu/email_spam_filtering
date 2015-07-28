__author__ = 'vivek'
import codecs, math
import re
from os import listdir
from os.path import isfile, join

constant_nu = 1
num_of_iterations = 10
check_stop_words = 1

path = ["../train3/ham",
        "../train3/spam"]

test_data_path = ["../test3/ham",
                  "../test3/spam"]

num_tokens = 0
word_list = []
samples_word_fr = []
weight_vector = []
y = []
stop_words = []

voc = {}

def is_token_useful(token):
    global stop_words, check_stop_words

    match_found = re.search(r'[0-9!@#$\%\'\"^&\*-_\(\)]', token)
    if match_found:
        return 0

    if token in stop_words and check_stop_words:
        return 0
    else:
        return 1


def get_feature_vector(file):
    global num_tokens, word_list

    fh = codecs.open(file, "rU", "utf-8", errors="ignore")

    feature_vector = [0 for i in range(num_tokens+1)]
    feature_vector[0] = 1

    for line in fh:
        tokens = line.split()
        for j in range(len(tokens)):
            if tokens[j] in word_list:
                feature_vector[word_list.index(tokens[j])] += 1

    return feature_vector


def calc_dot_product(feature_vector):
    global weight_vector, num_tokens

    dot_pdt = 0
    for i in range(num_tokens+1):
        dot_pdt += (weight_vector[i] * feature_vector[i])

    return dot_pdt


def build_statistics_for_perceptron():
    global num_tokens, stop_words, samples_word_fr, y, word_list, path

    stop_words_file = "./stop_words.txt"
    fh1 = codecs.open(stop_words_file, "rU", "utf-8", errors="ignore")
    for line in fh1:
        tokens = line.split()
        for j in range(len(tokens)):
            stop_words.append(tokens[j])

    for num_folder in range(len(path)):
        files_list = [f for f in listdir(path[num_folder]) if isfile(join(path[num_folder], f))]
        for i in range(len(files_list)):
            fh = codecs.open(join(path[num_folder], files_list[i]), "rU", "utf-8", errors="ignore")
            for line in fh:
                tokens = line.split()
                for j in range(len(tokens)):
                    if is_token_useful(tokens[j]):
                        voc[tokens[j]] = 1

    num_tokens = len(voc)
    # print("Total tokens: %d" % num_tokens)

    word_list.append("--reserved--")
    for word_ind in voc.keys():
        word_list.append(word_ind)

    for num_folder in range(len(path)):
        files_list = [f for f in listdir(path[num_folder]) if isfile(join(path[num_folder], f))]
        for i in range(len(files_list)):
            samples_word_fr.append(get_feature_vector(join(path[num_folder], files_list[i])))
            if num_folder == 0:
                y.append(-1)
            else:
                y.append(1)

    # print("samples word frequency size: %d" % len(samples_word_fr))


def find_weights_for_perceptron():
    global weight_vector, constant_nu, y, num_of_iterations

    weight_vector = [0 for i in range(num_tokens+1)]

    for loop_counter in range(num_of_iterations):
        for sample_index in range(len(samples_word_fr)):
            dot_pdt = calc_dot_product(samples_word_fr[sample_index])
            if dot_pdt > 0:
                output = 1
            else:
                output = -1

            if output != y[sample_index]:
                # print("Iteration: %d, sample: %d, dot pdt: %f" % (loop_counter, sample_index, dot_pdt))
                for feature_index in range(len(samples_word_fr[sample_index])):
                    if samples_word_fr[sample_index][feature_index] > 0:
                        weight_vector[feature_index] += constant_nu * (y[sample_index] - output)



def report_perceptron_accuracy():
    global num_tokens

    correct_prediction = 0
    total_cases = 0
    num_ham = 0
    num_spam = 0
    for num_folder in range(len(test_data_path)):
        files_list = [f for f in listdir(test_data_path[num_folder]) if isfile(join(test_data_path[num_folder], f))]

        for i in range(len(files_list)):
            x_vector = get_feature_vector(join(test_data_path[num_folder], files_list[i]))
            dot_pdt = calc_dot_product(x_vector)
            if dot_pdt > 0:
                output = 1
                num_spam += 1
            else:
                output = -1
                num_ham += 1

            if num_folder == 0:
                t = -1
            else:
                t = 1

            if output == t:
                correct_prediction += 1

            total_cases += 1

    print("Perceptron: Correct prediction = %d; Total cases = %d" % (correct_prediction, total_cases))
    print("num ham = %d; num spam = %d" % (num_ham, num_spam))
    accuracy = (correct_prediction / total_cases) * 100
    return accuracy


print("Program execution started ...")

build_statistics_for_perceptron()
find_weights_for_perceptron()
accuracy = report_perceptron_accuracy()
print("Perceptron accuracy: %f" % accuracy)

print("Program execution completed.")

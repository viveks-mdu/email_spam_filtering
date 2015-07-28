__author__ = 'vivek'
# vxs135130
import codecs, math
import re
from os import listdir
from os.path import isfile, join

constant_nu = 0.01
constant_lambda = 2
constant_epsilon = 1
check_stop_words = 1
lr_iterations = 10

test_data_path = []
test_data_path.append("./test/ham")
test_data_path.append("./test/spam")

num_files = 0
num_tokens = 0
num_words = 0
num_criteria_words = [0, 0]
num_criteria_files = [0, 0]
ham_prior = 0
spam_prior = 0
word_list = []
samples_word_fr = []
weight_vector = []
y = []
stop_words = []


voc = {}
# voc['word1'] = [5, 4]

def is_token_useful(token):
    global stop_words, check_stop_words

    match_found = re.search(r'[0-9!@#$\%\'\"^&\*-_\(\)]', token)
    if match_found:
        return 0

    if token in stop_words and check_stop_words:
        return 0
    else:
        return 1


def build_statistics():
    global num_files, num_tokens, num_words, num_criteria_words, num_criteria_files, ham_prior, spam_prior, stop_words

    path = []
    path.append("./train/ham")
    path.append("./train/spam")

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
                        # update the statistics
                        if tokens[j] in voc:
                            voc[tokens[j]][num_folder] += 1
                        else:
                            voc[tokens[j]] = [0, 0]
                            voc[tokens[j]][num_folder] += 1

                        num_criteria_words[num_folder] += 1

            num_criteria_files[num_folder] += 1

    num_tokens = len(voc)
    num_words = num_criteria_words[0] + num_criteria_words[1]
    num_files = num_criteria_files[0] + num_criteria_files[1]
    ham_prior = ((num_criteria_files[0] * 1.0) / num_files)
    spam_prior = ((num_criteria_files[1] * 1.0) / num_files)


    # print("Total files: %d" % num_files)
    # print("#ham files: %d; #spam files: %d" % (num_criteria_files[0], num_criteria_files[1]))
    # print("Total words: %d" % num_words)
    # print("#ham words: %d; #spam words: %d" % (num_criteria_words[0], num_criteria_words[1]))
    # print("Total tokens: %d" % num_tokens)
    # print("ham prior: %f; spam prior: %f" % (ham_prior, spam_prior))


def NB_categorize_file(file):
    global num_files, num_tokens, num_words, num_criteria_words, num_criteria_files, ham_prior, spam_prior, test_data_path
    fh = codecs.open(file, "rU", "utf-8", errors="ignore")

    ham_likelihood = 0
    spam_likelihood = 0
    for line in fh:
        tokens = line.split()
        for j in range(len(tokens)):
            current_word = tokens[j].lower()
            if is_token_useful(current_word):
                if current_word in voc.keys():
                    ham_count = voc[current_word][0]
                    spam_count = voc[current_word][1]
                else:
                    ham_count = 0
                    spam_count = 0

                ham_likelihood += math.log10((ham_count+1)/(num_criteria_words[0]+num_tokens))
                spam_likelihood += math.log10((spam_count+1)/(num_criteria_words[1]+num_tokens))

    ham_posterior = math.log10(ham_prior) + ham_likelihood
    spam_posterior = math.log10(spam_prior) + spam_likelihood
    # print("ham posterior =  %f, spam posterior = %f" % (ham_posterior, spam_posterior))

    if ham_posterior >= spam_posterior:
        return 0
    else:
        return 1

def report_NB_accuracy():
    global num_files, num_tokens, num_words, num_criteria_words, num_criteria_files, ham_prior, spam_prior
    correct_prediction = 0
    total_cases = 0
    num_ham = 0
    num_spam = 0
    for num_folder in range(len(test_data_path)):
        files_list = [f for f in listdir(test_data_path[num_folder]) if isfile(join(test_data_path[num_folder], f))]

        for i in range(len(files_list)):
            category = NB_categorize_file(join(test_data_path[num_folder], files_list[i]))
            if category == num_folder:
                correct_prediction += 1

            total_cases += 1

    # print("Correct prediction = %d; Total cases = %d" % (correct_prediction, total_cases))
    # print("num ham = %d; num spam = %d" % (num_ham, num_spam))
    accuracy = (correct_prediction / total_cases) * 100
    return accuracy

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

def build_statistics_for_LR():
    global samples_word_fr, y, word_list

    word_list.append("--reserved--")
    for word_ind in voc.keys():
        word_list.append(word_ind)

    path = []
    path.append("./train/ham")
    path.append("./train/spam")

    file_no = 0
    for num_folder in range(len(path)):
        files_list = [f for f in listdir(path[num_folder]) if isfile(join(path[num_folder], f))]

        for i in range(len(files_list)):
            samples_word_fr.append(get_feature_vector(join(path[num_folder], files_list[i])))
            y.append(num_folder)

def calc_dot_product(feature_vector):
    global weight_vector, num_tokens

    dot_pdt = 0
    for i in range(num_tokens+1):
        dot_pdt += (weight_vector[i] * feature_vector[i])

    return dot_pdt

def predict_probability(feature_vector):
    dot_pdt = calc_dot_product(feature_vector)
    p = (math.exp(dot_pdt) / (1 + math.exp(dot_pdt)))

    return p


def find_weights():
    global weight_vector, num_tokens, constant_nu, constant_lambda, constant_epsilon, y

    weight_vector = [0.01 for i in range(num_tokens+1)]
    temp_weight_vector = [0 for i in range(num_tokens+1)]

    max_delta_weight = 0

    # while max_delta_weight > constant_epsilon:
    for loop_counter in range(lr_iterations):
        delta_weight = 0
        max_delta_weight = 0
        for feature_index in range(num_tokens):
            # print("Processing feature %d" % feature_index)
            sum = 0
            for sample_index in range(num_files):
                if samples_word_fr[sample_index][feature_index] > 0:
                    p = predict_probability(samples_word_fr[sample_index])
                    error_difference = y[sample_index] - p
                    sum += (samples_word_fr[sample_index][feature_index] * error_difference)
            temp_weight_vector[feature_index] = (weight_vector[feature_index] * (1 - constant_nu * constant_lambda)) + (constant_nu * sum)
            delta_weight = math.fabs(weight_vector[feature_index] - temp_weight_vector[feature_index])
            if max_delta_weight < delta_weight:
                max_delta_weight = delta_weight

        for i in range(len(weight_vector)):
            weight_vector[i] = temp_weight_vector[i]

        # print("max_delta", max_delta_weight)
        # print(weight_vector)


def report_LR_accuracy():
    global num_files, num_tokens, num_words, num_criteria_words, num_criteria_files, ham_prior, spam_prior

    correct_prediction = 0
    total_cases = 0
    num_ham = 0
    num_spam = 0
    for num_folder in range(len(test_data_path)):
        files_list = [f for f in listdir(test_data_path[num_folder]) if isfile(join(test_data_path[num_folder], f))]

        for i in range(len(files_list)):
            x_vector = get_feature_vector(join(test_data_path[num_folder], files_list[i]))
            dot_pdt = calc_dot_product(x_vector)
            if dot_pdt < 0:
                y_value = 0
                num_ham += 1
            else:
                y_value = 1
                num_spam += 1

            if y_value == num_folder:
                correct_prediction += 1

            total_cases += 1

    print("LR: Correct prediction = %d; Total cases = %d" % (correct_prediction, total_cases))
    print("num ham = %d; num spam = %d" % (num_ham, num_spam))
    accuracy = (correct_prediction / total_cases) * 100
    return accuracy


print("Program execution started ...")

# read training files and build the statistics required
build_statistics()
nb_accuracy = report_NB_accuracy()
print("NB accuracy = %f" % nb_accuracy)

build_statistics_for_LR()
find_weights()
lr_accuracy = report_LR_accuracy()
print("LR accuracy = %f" % lr_accuracy)

print("Program execution completed.")

import numpy as np
import re
import csv
import itertools
from collections import Counter

DEFAULT_PADDING_WORD="<PAD/>"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def load_data_and_labels_v2():
    ts_file="./data/cat_data/test_data_16feb.csv"
    tr_file="./data/cat_data/prdcat_16feb.csv"
    file_tr=open(tr_file,'rb')
    file_ts=open(ts_file,'rb')
    reader_tr = csv.reader(file_tr)
    reader_ts = csv.reader(file_ts)
    x_tr_text=[]
    y_tr_text=[]
    x_ts_text=[]
    y_ts_text=[]
    for i,row in enumerate(reader_tr):
        if i>0:
            x_tr_text.append(row[0])
            y_tr_text.append(row[1])
    x_tr_text = [clean_str(sent) for sent in x_tr_text]
    x_tr_text = [s.split(" ") for s in x_tr_text]
    for i,row in enumerate(reader_ts):
        if i>0:
            x_ts_text.append(row[0])
            y_ts_text.append(row[1])
    x_ts_text = [clean_str(sent) for sent in x_ts_text]
    x_ts_text = [s.split(" ") for s in x_ts_text]
    return [x_tr_text, y_tr_text, x_ts_text, y_ts_text]





def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def pad_sentences_v2(x_tr_text, x_ts_text, padding_word=DEFAULT_PADDING_WORD):
    sequence_length = max(len(x) for x in x_tr_text)
    x_tr_padded = []
    x_ts_padded = []
    for i in range(len(x_tr_text)):
        sentence = x_tr_text[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        x_tr_padded.append(new_sentence)
    for i in range(len(x_ts_text)):
        sentence = x_ts_text[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        x_ts_padded.append(new_sentence)
    return [x_tr_padded, x_ts_padded]


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def build_input_data_v2(x_tr_padded, y_tr_text, x_ts_padded, y_ts_text):
    word_counts_x = Counter(itertools.chain(*x_tr_padded))
    vocabulary_inv_x = [x[0] for x in word_counts_x.most_common()]
    vocabulary_x = {x: i for i, x in enumerate(vocabulary_inv_x)}
    word_counts_y = Counter(y_tr_text)
    vocabulary_inv_y = [x[0] for x in word_counts_y.most_common()]
    vocabulary_y = {x: i for i, x in enumerate(vocabulary_inv_y)}

    x_tr = np.array([[vocabulary_x[word] for word in sentence] for sentence in x_tr_padded])
    x_ts = np.array([[vocabulary_x.get(word) or vocabulary_x.get(DEFAULT_PADDING_WORD) for word in sentence] for sentence in x_ts_padded])
    y_tr=np.zeros((len(y_tr_text),len(vocabulary_inv_y)),dtype=np.int)
    y_ts=np.zeros((len(y_ts_text),len(vocabulary_inv_y)),dtype=np.int)
    for i,label in enumerate(y_tr_text):
        y_tr[i][vocabulary_y[label]]=1
    for i,label in enumerate(y_ts_text):
        y_ts[i][vocabulary_y[label]]=1
    return [x_tr, y_tr, x_ts, y_ts, vocabulary_x, vocabulary_inv_x, vocabulary_y, vocabulary_inv_y]
    

def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_cat_data():
    x_tr_text, y_tr_text, x_ts_text, y_ts_text=load_data_and_labels_v2()
    print "File reading done"
    x_tr_padded, x_ts_padded = pad_sentences_v2(x_tr_text, x_ts_text)
    print "Padding done"
    return build_input_data_v2(x_tr_padded, y_tr_text, x_ts_padded, y_ts_text)



def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



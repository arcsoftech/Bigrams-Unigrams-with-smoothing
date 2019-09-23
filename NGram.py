import sys
from collections import OrderedDict, Counter
import pandas as pd
import timeit
import os

# split word and tag
def __token_splitter__(x):
    return tuple(x.split("_"))

# Get sentence token
def __get_sentence_token(sentence):
    return sentence.split(" ")

# Generate tokens and required parameter from the dataset
def __get_corpus_tokens__(sentences):
    taggedTokens = [x.strip().split(" ") for x in sentences]
    sentenceTokens = []
    for x in taggedTokens:
        splittedToken = [__token_splitter__(y) for y in x]
        sentenceTokens.append(splittedToken)
    sentenceList = []
    for sent in sentenceTokens:
        sent_token = " ".join([y[0].lower() for y in sent])
        sentenceList.append(sent_token)
    tokens = []
    for x in sentenceTokens:
        for y in x:
            tokens.append(y[0].lower())
    print("N is ", tokens.__len__())
    print("V is ", set(tokens).__len__())
    return tokens, sentenceList


#Count Unigram
def __unigram_count_model__(tokens, unigramModel=None):
    unigram = {}
    if(unigramModel is None):
        unigram = Counter(tokens)
    else:
        unigram = OrderedDict({token: unigramModel.get(token)
                               for token in tokens})
    return unigram

#Count Bigram
def __bigram_count_model__(sentences, bModel=None):
    bigrams = []
    if(bModel is None):
        sentence_token = [x.split(" ") for x in sentences]
        flat_list = []
        for x in sentence_token:
            sequences = [x[i:] for i in range(2)]
            bigrams = zip(*sequences)
            flat_list += (bigrams)

        bigrams = Counter(flat_list)
    else:
        token_words = sentences.split(" ")
        sequences = [token_words[i:] for i in range(2)]
        sentbigrams = zip(*sequences)
        bigrams = OrderedDict({token: bModel.get(token) if bModel.get(
            token) is not None else 0 for token in sentbigrams})
    return bigrams

#Bigram probability
def __bigram_probability_model__(unigram, bigram):
    bigramsProb = OrderedDict({})
    for key in bigram.keys():
        if unigram[key[0]] == 0:
            bigramsProb[key] = - float('inf')
        else:
            bigramsProb[key] = bigram[key] / unigram[key[0]]
    return bigramsProb


# Get Bigram Count after add one smoothing i.e Cstar
def __bigram_count_add_one_smoothing__(unigram, bigram):
    bigramCountAfterAddOne = OrderedDict({})
    V = len(unigram.keys())
    for key in bigram.keys():
        bigramCountAfterAddOne[key] = (
            bigram[key] + 1) * (unigram[key[0]]) / (unigram[key[0]] + V)
    return bigramCountAfterAddOne

# Get Bigram probability after add one smoothing
def __bigram_probability_with_add_one_smoothing__(unigram, bigram):
    bigramsProb = OrderedDict({})
    V = len(unigram.keys())
    for key in bigram.keys():
        bigramsProb[key] = (bigram[key] + 1) / (unigram[key[0]]+V)
    return bigramsProb

# Get Bigram frequency
def __compute_bigram_compute_frequency__(unigram, bigram):
    bucketLength = max(bigram.values())
    bucket = [0] * (bucketLength + 2)
    for i in range(bucketLength + 2):
        for key in bigram.keys():
            if (bigram[key] == i):
                bucket[i] += 1
    return bucket

#Get Bigram counts after good turing i.e Cstar
def __bigram_good_turing_counts__(unigram, bigram):
    bigramsCounts = OrderedDict({})
    bucketFrequency = __compute_bigram_compute_frequency__(unigram, bigram)
    for key in bigram.keys():
        if (bucketFrequency[bigram[key]+1] == 0):
            bigramsCounts[key] = 0.0
        else:
            bigramsCounts[key] = (
                bigram[key] + 1)*bucketFrequency[bigram[key]+1] / bucketFrequency[bigram[key]]
    return bigramsCounts

# Get bigram probability after goo turing discount smoothing
def __bigram_good_turing_probability__(unigram, bigram):
    bigramsProbs = OrderedDict({})
    bigram_cstar = __bigram_good_turing_counts__(unigram, bigram)
    bucketFrequency = __compute_bigram_compute_frequency__(unigram, bigram)
    N = 0
    for i, k in enumerate(bucketFrequency):
        N += i*k
    for key in bigram_cstar.keys():
        bigramsProbs[key] = bigram_cstar[key] / N

    return bigram_cstar, bigramsProbs


def main():
    pd.set_option('expand_frame_repr', False)
    start = timeit.default_timer()
    if len(sys.argv) != 3:
        print(
            "Incorrect Arguments! Correct form as: python NGram.py <corpus.txt> <sentence>")
    else:
        corpus_file_name = sys.argv[1]
        testSentence = sys.argv[2].lower().strip()
        dataset = open(corpus_file_name, "r").read()
        sentences = dataset.strip().split("\n")
        corpusTokens, sentenceList = __get_corpus_tokens__(sentences)


####################################Training###################################################################
        """
        No Smoothing
        """
        bigram = __bigram_count_model__(sentenceList)
        unigram = __unigram_count_model__(corpusTokens)
        bigram_probability_model = __bigram_probability_model__(
            unigram, bigram)
        bigramColumns= ["P({}|{})".format(key[1],key[0]) for key in bigram_probability_model]
        unigramModelDataFrame = pd.DataFrame.from_dict({"Unigram": unigram.keys(), "Count": unigram.values()},
                                                       orient='index').transpose()
        bigramModelDataFrame = pd.DataFrame.from_dict({"Bigram": bigram.keys(), "Count": bigram.values()},
                                                      orient='index').transpose()
        bigramProbabilityModelDataFrame = pd.DataFrame.from_dict({"Bigram Probability(No Smoothing)": bigramColumns, "Probability": bigram_probability_model.values()},
                                                                 orient='index').transpose()

        """
        Add One Smoothing
        """

        bigramAddOneSmoothingProbability = __bigram_probability_with_add_one_smoothing__(
            unigram, bigram)
        bigramAddOneSmoothingProbabilityDataFrame = pd.DataFrame.from_dict({"Bigram Probability(Add One Smoothing)": bigramColumns, "Probability": bigramAddOneSmoothingProbability.values()},
                                                                           orient='index').transpose()
        bigramsWithAddOneSmoothingCounts = __bigram_count_add_one_smoothing__(
            unigram, bigram)
        bigramAddOneSmoothingDataFrame = pd.DataFrame.from_dict({"Bigram Cstar(Add One Smoothing)": bigramsWithAddOneSmoothingCounts.keys(), "Count": bigramsWithAddOneSmoothingCounts.values()},
                                                                orient='index').transpose()

        """
        Good Turing Discount Smoothing
        """

        bigramsWithGoodTuringSmoothingCounts, bigramGoodTuringSmoothingProbability = __bigram_good_turing_probability__(
            unigram, bigram)
        bigramGoodTuringeSmoothingDataFrame = pd.DataFrame.from_dict({"Bigram Cstar(Good Turing Smoothing)": bigramsWithGoodTuringSmoothingCounts.keys(), "Count": bigramsWithGoodTuringSmoothingCounts.values()},
                                                                     orient='index').transpose()
        bigramGoodTuringSmoothingProbabilityDataFrame = pd.DataFrame.from_dict({"Bigram Cstar(Good Turing Smoothing)": bigramColumns, "Probability": bigramGoodTuringSmoothingProbability.values()},
                                                                               orient='index').transpose()
####################################Testing#####################################################################
        """
        No Smoothing
        """
        
        print("Bigram  models are stored in the output folder in their respected folder.Zero probabilities & counts are ignored.\nTesting started.....\n")
        print("No Smoothing counts and probability for test sentence\n")
        unigramCounts = __unigram_count_model__(
            __get_sentence_token(testSentence), unigramModel=unigram)
        bigramCounts = __bigram_count_model__(testSentence, bModel=bigram)
        unigramDataFrame = pd.DataFrame.from_dict({"Unigram": unigramCounts.keys(), "Count": unigramCounts.values()},
                                                  orient='index')
        print(unigramDataFrame)
        bigramDataFrame = pd.DataFrame.from_dict({"Bigram": bigramCounts.keys(), "Count": bigramCounts.values()},
                                                 orient='index')
        print(bigramDataFrame)
        bigramProbabilityTestSent = __bigram_probability_model__(
            unigramCounts, bigramCounts)
        bigramProbabilityTestSentDataFrame = pd.DataFrame.from_dict({"Bigram Probability": bigramProbabilityTestSent.keys(), "Count": bigramProbabilityTestSent.values()},
                                                                    orient='index')
        print(bigramProbabilityTestSentDataFrame)
        print("\n")

        """
        Add One Smoothing
        """
        print("Add one smoothing counts and probability for test sentence\n")

        bigramAddOneSmoothingProbabilityTestSentence = __bigram_probability_with_add_one_smoothing__(
            unigramCounts, bigramCounts)
        bigramAddOneSmoothingProbabilityTestSentenceDataFrame = pd.DataFrame.from_dict({"Bigram Probability": bigramAddOneSmoothingProbabilityTestSentence.keys(), "Count": bigramAddOneSmoothingProbabilityTestSentence.values()},
                                                                                       orient='index')
        print(bigramAddOneSmoothingProbabilityTestSentenceDataFrame)

        bigramsWithAddOneSmoothingCountsTestSentence = __bigram_count_add_one_smoothing__(
            unigramCounts, bigramCounts)
        bigramAddOneSmoothingTestSentenceDataFrame = pd.DataFrame.from_dict({"Bigram Cstar": bigramsWithAddOneSmoothingCountsTestSentence.keys(), "Count": bigramsWithAddOneSmoothingCountsTestSentence.values()},
                                                                            orient='index')
        print(bigramAddOneSmoothingTestSentenceDataFrame)
        print("\n")
        print(bigramAddOneSmoothingProbabilityTestSentenceDataFrame)
        print("\n")

        """
        Good Turing Discount Smoothing
        """
        print("Good Turing counts and probability for test sentence\n")
        bigramsWithGoodTuringSmoothingCountsTestSent, bigramGoodTuringSmoothingProbabilityTestSent = __bigram_good_turing_probability__(
            unigramCounts, bigramCounts)
        bigramGoodTuringeSmoothingTestSentDataFrame = pd.DataFrame.from_dict({"Bigram Cstar": bigramsWithGoodTuringSmoothingCountsTestSent.keys(), "Count": bigramsWithGoodTuringSmoothingCountsTestSent.values()},
                                                                             orient='index')
        bigramGoodTuringSmoothingTestSentProbabilityDataFrame = pd.DataFrame.from_dict({"Bigram Proability": bigramGoodTuringSmoothingProbabilityTestSent.keys(), "Count": bigramGoodTuringSmoothingProbabilityTestSent.values()},
                                                                                       orient='index')
        print(bigramGoodTuringSmoothingTestSentProbabilityDataFrame)
        print("\n")
        print(bigramGoodTuringeSmoothingTestSentDataFrame)
        print("\n")
#####################################Output#####################################################################
        os.makedirs("output/NoSmoothing/", exist_ok=True)
        os.makedirs("output/AddOneSmoothing/", exist_ok=True)
        os.makedirs("output/GoodTuringSmoothing/", exist_ok=True)

        unigramModelDataFrame.to_csv(
            r'output/NoSmoothing/unigram_count_model.csv')
        bigramModelDataFrame.to_csv(
            r'output/NoSmoothing/bigram_count_model.csv')
        bigramProbabilityModelDataFrame.to_csv(
            r'output/NoSmoothing/bigram_probability_model.csv')
        bigramAddOneSmoothingProbabilityDataFrame.to_csv(
            "output/AddOneSmoothing/bigram_probability_model.csv")
        bigramAddOneSmoothingDataFrame.to_csv(
            "output/AddOneSmoothing/bigram_reconstituted_counts_model.csv")
        bigramGoodTuringeSmoothingDataFrame.to_csv(
            "output/GoodTuringSmoothing/bigram_reconstituted_counts_model.csv")
        bigramGoodTuringSmoothingProbabilityDataFrame.to_csv(
            "output/GoodTuringSmoothing/bigram_probability_model.csv")
    stop = timeit.default_timer()
    print('Execution Time: ', stop - start)


if __name__ == '__main__':
    main()

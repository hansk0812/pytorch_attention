import collections
import traceback

import string
from unidecode import unidecode
import codecs
import binascii

import torch
from torch.utils.data import Dataset, Sampler
from typing import Iterable, List

import os
import random
import re
import numpy as np
from collections import OrderedDict

from requests.exceptions import ConnectionError
import nltk
import stanza
try:
    nltk.download('punkt')
    stanza.download('en')
    #stanza.download('ta')
    en_nlp = stanza.Pipeline('en', processors='tokenize')
    #ta_nlp = stanza.Pipeline('ta', processors='tokenize')
except ConnectionError:
    en_nlp = stanza.Pipeline('en', processors='tokenize', download_method=None)
    #ta_nlp = stanza.Pipeline('ta', processors='tokenize', download_method=None)

from utils.dataset_visualization import visualize_dataset_for_bucketing_stats

from gensim.models import Word2Vec

from datasets.utils import return_unicode_hex_within_range, return_tamil_unicode_isalnum, check_unicode_block

from indicnlp.morph import unsupervised_morph 
from indicnlp import common
common.INDIC_RESOURCES_PATH="/home/hans/NMT_repetitions/indic_nlp_library/indic_nlp_resources/"

class EnTamV2Dataset(Dataset):

    SRC_LANGUAGE = 'en'
    TGT_LANGUAGE = 'ta'

    # Using START and END tokens in source and target vocabularies to enforce better relationships between x and y
    reserved_tokens = ["UNK", "PAD", "START", "END", "NUM", "ENG"]
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, NUM_IDX, ENG_IDX = 0, 1, 2, 3, 4, 5
    num_token_sentences = 500

    word_vector_size = 100
  
    # Pretrained word2vec models take too long to load (many hours) - training my own
    #en_wv = load_facebook_model('dataset/monolingual/cc.en.300.bin_fasttext.bin')
    #ta_wv = load_facebook_model('dataset/monolingual/cc.ta.300.bin_fasttext.gz')

    def __init__(self, 
                 split, 
                 morphemes=False,
                 symbols=False, 
                 buckets=[(5, 5), (10, 11), (12, 12), (14, 13), (18,15), (20,17), (24,25), (30,30), (45,50), (85,80)],
                 #buckets=[[12,10],[15,12],[18,14],[21,16],[25,18],[28,21],[32,23],[37,26],[41,30],[50,35],[70,45],[100,100]], 
                 verbose=False,
                 max_vocab_size=150000,
                 vocabularies=(None, None),
                 start_stop_tokens=True):
        
        # Max vocab size at 325000 makes Epoch time = 5016.199s at 30GB GPU usage
        # Max vocab size at 150000 makes Epoch time = 2815.011s (~47 mins) at 6GB GPU usage
        # Max vocab size at 150000 makes Epoch time = 478.569s (~8 mins) at 30GB GPU usage
        # Num sentences per bucket: [10886, 11017, 14521, 15839, 17838, 17348, 14903, 17230, 15119, 16668, 14758, 6503]
        # symbols is a choice based on sequence length increase, context and similar potentially similar word vectors in either languages
        # Number of buckets estimated from dataset stats
        
        self.buckets = buckets
        self.morphemes = morphemes
        self.split = split
        self.verbose = verbose
        self.max_vocab_size = max_vocab_size
        self.start_stop_tokens = start_stop_tokens
        
        if not split == "train":
            self.eng_vocabulary, self.tam_vocabulary = vocabularies

        self.tamil_morph_analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer('ta')

        tokenized_dirname = "tokenized" if not self.morphemes else "tokenized_morphs"
        tokenized_dirname = tokenized_dirname if symbols else tokenized_dirname + "_nosymbols"
        tokenized_dirname = tokenized_dirname if start_stop_tokens else tokenized_dirname + "_nostartstop"
        if not os.path.exists(self.get_dataset_filename(split, "en", tokenized_dirname)) \
                or not os.path.exists(self.get_dataset_filename(split, "ta", tokenized_dirname)):
            
            self.bilingual_pairs, eng_words = self.get_sentence_pairs(split, symbols=symbols)
            
            if split == "train":
                eng_words = list(eng_words)
                self.create_token_sentences_for_word2vec(eng_words)

            self.eng_vocabulary, self.eng_word_counts, tokenized_eng_sentences = self.create_vocabulary([
                                                                                    x[0] for x in self.bilingual_pairs], language="en")
            self.tam_vocabulary, self.tam_word_counts, tokenized_tam_sentences = self.create_vocabulary([
                                                                                    x[1] for x in self.bilingual_pairs], language="ta")
            
            if split == "train":
                if len(self.eng_vocabulary) > self.max_vocab_size:
                    self.eng_vocabulary = sorted(self.eng_word_counts, key=lambda y: self.eng_word_counts[y], reverse=True)[:self.max_vocab_size-len(self.reserved_tokens)]
                    self.eng_vocabulary = [x for x in self.eng_vocabulary]
                if len(self.tam_vocabulary) > self.max_vocab_size:
                    self.tam_vocabulary = sorted(self.tam_word_counts, key=lambda y: self.tam_word_counts[y], reverse=True)[:self.max_vocab_size-len(self.reserved_tokens)]
                    self.tam_vocabulary = [x for x in self.tam_vocabulary]
            self.eng_vocabulary = set(self.eng_vocabulary)
            self.eng_vocabulary.update(self.reserved_tokens)
            self.tam_vocabulary = set(self.tam_vocabulary)
            self.tam_vocabulary.update(self.reserved_tokens)

            if self.verbose:
                print ("Most Frequent 1000 English tokens:", sorted(self.eng_word_counts, key=lambda y: self.eng_word_counts[y], reverse=True)[:1000])
                print ("Most Frequent 1000 Tamil tokens:", sorted(self.tam_word_counts, key=lambda y: self.tam_word_counts[y], reverse=True)[:1000])

            # save tokenized sentences for faster loading
            with open(self.get_dataset_filename(split, "en", tokenized_dirname), 'w') as f:
                for line in tokenized_eng_sentences:
                    f.write("%s\n" % line)
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname), 'w') as f:
                for line in tokenized_tam_sentences:
                    f.write("%s\n" % line)
       
        else:
            
            with open(self.get_dataset_filename(split, "en", tokenized_dirname), 'r') as f:
                tokenized_eng_sentences = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname), 'r') as f:
                tokenized_tam_sentences = [x.strip() for x in f.readlines()]

        # Remove UNK UNK, NUM NUM and ENG ENG tokens
        tokenized_eng_sentences = [re.sub("(UNK )+", "UNK ",
                                   re.sub("(ENG )+", "ENG ",
                                   re.sub("(NUM ,*)+", "NUM ", x))) for x in tokenized_eng_sentences]
        tokenized_tam_sentences = [re.sub("(UNK )+", "UNK ",
                                   re.sub("(ENG )+", "ENG ",
                                   re.sub("(NUM ,*)+", "NUM ", x))) for x in tokenized_tam_sentences]

        self.bilingual_pairs = list(zip(tokenized_eng_sentences, tokenized_tam_sentences))
        
        if hasattr(self, "eng_vocabulary"):
            assert not "DEBUG" in self.eng_vocabulary, "Debug token found in final train dataset"

        if not os.path.exists('utils/Correlation.png') and split == "train":
            visualize_dataset_for_bucketing_stats(self.bilingual_pairs)
        
        # redefine bilingual_pairs to include bucketing based padding for word vectorization
        for idx in range(len(self.bilingual_pairs)):
            eng, tam = self.bilingual_pairs[idx]
            eng_tokens, tam_tokens = eng.split(' '), tam.split(' ')

            E, T = len(eng_tokens), len(tam_tokens)

            # clip all tokens after buckets[-1] words
            if E > buckets[-1][0]:
                eng_tokens = eng_tokens[:buckets[-1][0]]
            if T > buckets[-1][1]:
                tam_tokens = tam_tokens[:buckets[-1][1]]
                
            for bucket_idx in range(len(buckets)):
                if buckets[bucket_idx][1] < T:
                    continue
                if buckets[bucket_idx][0] >= E:
                    break

            eng_tokens = eng_tokens + [self.reserved_tokens[self.PAD_IDX]] * (buckets[bucket_idx][0] - E)
            tam_tokens = tam_tokens + [self.reserved_tokens[self.PAD_IDX]] * (buckets[bucket_idx][1] - T)

            tokenized_eng_sentences[idx] = " ".join(eng_tokens)
            tokenized_tam_sentences[idx] = " ".join(tam_tokens)
        
        self.bilingual_pairs = list(zip(tokenized_eng_sentences, tokenized_tam_sentences))
        
        self.bilingual_pairs = sorted(self.bilingual_pairs, key=lambda x: len(x[1].split(' ')))
        self.bilingual_pairs = [x for x in self.bilingual_pairs if x[0] != "" and x[1] != ""]
        
        self.bucketing_indices, b_idx, start_idx = [], 0, 0

        # remove reserved_token_sentences for bucketing and extend it for word2vec embedding dataset
        if split == "train":
            word2vec_reserved_token_sentences = self.bilingual_pairs[-self.num_token_sentences:]
            self.bilingual_pairs = self.bilingual_pairs[:-self.num_token_sentences]

        for idx in range(len(self.bilingual_pairs)):
            if buckets[b_idx][1] == len(self.bilingual_pairs[idx][1].split(' ')):
                continue
            else:
                b_idx += 1
                self.bucketing_indices.append((start_idx, idx-1))
                start_idx = idx
        self.bucketing_indices.append((start_idx, idx-1))
        
        # remove reserved_token_sentences for bucketing and extend it for word2vec embedding dataset
        if split == "train":
            self.bilingual_pairs.extend(word2vec_reserved_token_sentences)

        if not os.path.exists(self.get_dataset_filename(split, "en", tokenized_dirname, substr="vocab")):
            self.eng_vocabulary = list(self.eng_vocabulary)
            self.tam_vocabulary = list(self.tam_vocabulary)
            with open(self.get_dataset_filename(split, "en", tokenized_dirname, substr="vocab"), 'w') as f:
                for word in self.eng_vocabulary:
                    f.write("%s\n" % word)
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname, substr="vocab"), 'w') as f:
                for word in self.tam_vocabulary:
                    f.write("%s\n" % word)
        else:
            with open(self.get_dataset_filename(split, "en", tokenized_dirname, substr="vocab"), 'r') as f:
                self.eng_vocabulary = [x.strip() for x in f.readlines()]
            with open(self.get_dataset_filename(split, "ta", tokenized_dirname, substr="vocab"), 'r') as f:
                self.tam_vocabulary = [x.strip() for x in f.readlines()]
            if self.verbose:
                print ("Loading trained word2vec models")
            
        if os.path.exists("dataset/%s/word2vec_entam.en.model" % ("word2vec" if not self.morphemes else "word2vec_morphemes")) and \
                os.path.exists("dataset/%s/word2vec_entam.ta.model" % ("word2vec" if not self.morphemes else "word2vec_morphemes")):
            self.en_wv = Word2Vec.load("dataset/%s/word2vec_entam.en.model" % ("word2vec" if not self.morphemes else "word2vec_morphemes"))
            self.ta_wv = Word2Vec.load("dataset/%s/word2vec_entam.ta.model" % ("word2vec" if not self.morphemes else "word2vec_morphemes"))
        else:
        
            if not os.path.isdir("dataset/word2vec_morphemes") and self.morphemes:
                os.mkdir("dataset/word2vec_morphemes")

            if not os.path.exists("dataset/%s/word2vec_entam.en.model" % ("word2vec" if not self.morphemes else "word2vec_morphemes")) or not \
                    os.path.exists("dataset/%s/word2vec_entam.ta.model" % ("word2vec" if not self.morphemes else "word2vec_morphemes")):
                if split == "train":
                    self.train_word2vec_model_on_monolingual_and_mt_corpus(symbols, \
                            tokenized_eng_sentences, tokenized_tam_sentences)       
        
        print ("English vocabulary size for %s set: %d" % (split, len(self.eng_vocabulary)))
        print ("Tamil vocabulary size for %s set: %d" % (split, len(self.tam_vocabulary)))
        print ("Using %s set with %d sentence pairs" % (split, len(self.bilingual_pairs)))
        
        # Sanity check for word vectors OOV
        if self.verbose:
            for sentence in tokenized_eng_sentences:
                for token in sentence.split(' '):
                    self.get_word2vec_embedding_for_token(token, "en")
            for sentence in tokenized_tam_sentences:
                for token in sentence.split(' '):
                    self.get_word2vec_embedding_for_token(token, "ta")
        
        self.eng_embedding = np.array([self.get_word2vec_embedding_for_token(word, "en") for word in self.eng_vocabulary])
        self.tam_embedding = np.array([self.get_word2vec_embedding_for_token(word, "ta") for word in self.tam_vocabulary])
        
        if split == "train":
            self.eng_vocabulary = {word: idx for idx, word in enumerate(self.eng_vocabulary)}
            self.tam_vocabulary = {word: idx for idx, word in enumerate(self.tam_vocabulary)}
        else:
            self.eng_vocabulary, self.tam_vocabulary = vocabularies
        
        self.eng_vocabulary_reverse = {self.eng_vocabulary[key]: key for key in self.eng_vocabulary}
        self.tam_vocabulary_reverse = {self.tam_vocabulary[key]: key for key in self.tam_vocabulary}

        self.ignore_index = self.tam_vocabulary[self.reserved_tokens[self.PAD_IDX]]
        self.bos_idx = self.tam_vocabulary[self.reserved_tokens[self.BOS_IDX]]
        self.eos_idx = self.tam_vocabulary[self.reserved_tokens[self.EOS_IDX]]
 
        if split == "train":
           
            self.bilingual_pairs = self.bilingual_pairs[:-self.num_token_sentences]
            print ("Removed %d reserved sentences meant for word vectorization!" % (self.num_token_sentences))

    def __len__(self):
        return len(self.bilingual_pairs)

    def __getitem__(self, idx):
        
        eng, tam = self.bilingual_pairs[idx]
        eng, tam = eng.split(' '), tam.split(' ')

        np_src, np_tgt = np.zeros(len(eng)), np.zeros(len(tam))
        
        for idx in range(len(eng)):
            try:
                np_src[idx] = self.eng_vocabulary[eng[idx]]
            except KeyError: # token not in train vocabulary (val and test sets)
                np_src[idx] = self.eng_vocabulary[self.reserved_tokens[self.UNK_IDX]]
        for idx in range(len(tam)):
            try:
                np_tgt[idx] = self.tam_vocabulary[tam[idx]]
            except KeyError:
                np_tgt[idx] = self.tam_vocabulary[self.reserved_tokens[self.UNK_IDX]]
        
        src_mask = [0 if x!=self.reserved_tokens[self.PAD_IDX] else 1 for x in eng]
        src_mask = np.array(src_mask)

        tgt_mask = [0 if x!=self.reserved_tokens[self.PAD_IDX] else 1 for x in tam]
        tgt_mask = np.array(tgt_mask)

        return np.int32(np_src), np_tgt, src_mask, tgt_mask
    
    def vocab_indices_to_sentence(self, sentence, language):
        # tensor to sentence

        assert language in ["en", "ta"]
        
        return_sentence = ""
        for idx in sentence:
            if language=='en':
                return_sentence += self.eng_vocabulary_reverse[idx.item()] + " "
            else:
                return_sentence += self.tam_vocabulary_reverse[idx.item()] + " "

        return return_sentence

    def return_vocabularies(self):
        return self.eng_vocabulary, self.tam_vocabulary

    def get_sentence_given_src(self, src, gt=False):
        # num_tokens, vocab_size
        sentence = ""
        for cls in src:
            if not gt:
                token = self.eng_vocabulary_reverse[cls]
            else:
                token = self.tam_vocabulary_reverse[cls]
            sentence += token + " "
        return sentence.strip()

    def get_sentence_given_preds(self, preds):
        # num_tokens, vocab_size
        sentence = ""
        for cls in preds:
            token = self.tam_vocabulary_reverse[np.argmax(cls)]
            if token == self.reserved_tokens[self.EOS_IDX]:
                break
            sentence += token + " "
        return sentence.strip()

    def get_word2vec_embedding_for_token(self, token, lang):
        
        try:
            if lang == "en":
                return self.en_wv.wv[token]
            else:
                return self.ta_wv.wv[token]
        
        except (KeyError, AttributeError):
            
            #traceback.print_exc()
            if self.verbose:
                print ("Token not in %s %s word2vec vocabulary: %s" % (self.split, lang, token))
            # word vector not in vocabulary - possible for tokens in val and test sets
            if lang == "en":
                return self.en_wv.wv[self.reserved_tokens[self.UNK_IDX]]
            else:
                return self.ta_wv.wv[self.reserved_tokens[self.UNK_IDX]]

    def train_word2vec_model_on_monolingual_and_mt_corpus(self, symbols, en_train_set, ta_train_set):

        with open(self.get_dataset_filename("train", "en", subdir="word2vec", substr="word2vec"), 'r') as f:
            eng_word2vec = [x.strip() for x in f.readlines()]

        with open(self.get_dataset_filename("train", "ta", subdir="word2vec", substr="word2vec"), 'r') as f:
            tam_word2vec = [x.strip() for x in f.readlines()]
        
        if self.verbose:
            print ("Preprocessing word2vec datasets for English and Tamil")
        word2vec_sentences, word2vec_eng_words = self.get_sentence_pairs("train", symbols=symbols, dataset=[eng_word2vec, tam_word2vec])
        en_word2vec, ta_word2vec = word2vec_sentences
        
        if self.verbose:
            print ("Tokenizing word2vec English and Tamil monolingual corpora")
        _,_, en_word2vec = self.create_vocabulary(en_word2vec, language="en")
        _,_, ta_word2vec = self.create_vocabulary(ta_word2vec, language="ta")
        
        en_word2vec.extend(en_train_set)
        ta_word2vec.extend(ta_train_set)

        en_word2vec = [x.split(' ') for x in en_word2vec]
        ta_word2vec = [x.split(' ') for x in ta_word2vec]
        
        if self.verbose:
            print ("Training word2vec vocabulary for English")
        self.en_wv = Word2Vec(sentences=en_word2vec, vector_size=self.word_vector_size, window=5, min_count=1, workers=4)
        self.en_wv.build_vocab(en_word2vec)
        self.en_wv.train(en_word2vec, total_examples=len(en_word2vec), epochs=20)
        self.en_wv.save("dataset/%s/word2vec_entam.en.model" % ("word2vec" if not self.morphemes else "word2vec_morphemes"))

        if self.verbose:
            print ("Training word2vec vocabulary for Tamil")
        self.ta_wv = Word2Vec(sentences=ta_word2vec, vector_size=self.word_vector_size, window=5, min_count=1, workers=4)
        self.ta_wv.build_vocab(ta_word2vec)
        self.ta_wv.train(ta_word2vec, total_examples=len(ta_word2vec), epochs=20)
        self.ta_wv.save("dataset/%s/word2vec_entam.ta.model" % ("word2vec" if not self.morphemes else "word2vec_morphemes"))

    def get_dataset_filename(self, split, lang, subdir=None, substr=""): 
        assert split in ['train', 'dev', 'test', ''] and lang in ['en', 'ta', ''] # Using '' to get dirname because dataset was defined first here!
        assert substr in ["", "vocab", "word2vec"]
        if not subdir is None:
            if substr not in ["vocab", "word2vec"]:
                directory = os.path.join("dataset", subdir, "%s.bcn" % "corpus")
            else:
                directory = os.path.join("dataset", subdir, "%s.bcn" % substr)
        else:
            if substr not in ["vocab", "word2vec"]:
                directory = os.path.join("dataset", "%s.bcn" % "corpus")
            else:
                raise AssertionError

        full_path = "%s.%s.%s" % (directory, split, lang)
        
        save_dir = os.path.dirname(full_path)
        
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        
        return full_path
    
    def get_morphologically_analysed_tamil_sentence(self, sentence):
        
        if self.morphemes:
            tokens = self.tamil_morph_analyzer.morph_analyze_document(sentence.split(' '))
            return ' '.join(tokens)
        else:
            return sentence

    def get_sentence_pairs(self, split, symbols=False, dataset=None):
        # use symbols flag to keep/remove punctuation

        text_pairs = []
        translator = str.maketrans('', '', string.punctuation)
        
        unnecessary_symbols = ["‘", "¦", "¡", "¬", '“', '”', "’", '\u200c'] # negation symbol might not be in EnTamV2
        # Exclamation mark between words in train set
        
        if symbols:
            symbol_replacements = {unnecessary_symbols[0]: "'", unnecessary_symbols[4]: '"', unnecessary_symbols[5]: "\"", unnecessary_symbols[6]: "'"}
        else:
            symbol_replacements = {}
        
        if not dataset is None:
            eng_sentences = dataset[0]
        else:
            with open(self.get_dataset_filename(split, self.SRC_LANGUAGE), 'r') as l1:
                eng_sentences = [re.sub(
                                    '\d+', ' %s ' % self.reserved_tokens[self.NUM_IDX], x.lower() # replace all numbers with [NUM] token
                                    ).strip().replace("  ", " ")
                                            for x in l1.readlines()]
            
        for idx, sentence in enumerate(eng_sentences):
            
            for sym in symbol_replacements:
                eng_sentences[idx] = eng_sentences[idx].replace(sym, " " + symbol_replacements[sym] + " ")
            
            if not symbols:
                eng_sentences[idx] = re.sub(r'[^\w\s*]|_',r' ', sentence)
            else:
                for ch in string.punctuation:
                    eng_sentences[idx] = eng_sentences[idx].replace(ch, " "+ch+" ")

            for sym_idx, sym in enumerate(unnecessary_symbols):
                #return_in_bilingual_corpus = eng_sentences[idx]
                if not symbols or (symbols and not sym in symbol_replacements.keys()):
                    eng_sentences[idx] = eng_sentences[idx].replace(sym, "")

            eng_sentences[idx] = re.sub("\s+", " ", eng_sentences[idx]) # correct for number of spaces
            
            # manual corrections
            eng_sentences[idx] = eng_sentences[idx].replace("naa-ve", "naive")
            eng_sentences[idx] = re.sub(r"j ' (\w)", r"i \1", eng_sentences[idx])
            eng_sentences[idx] = eng_sentences[idx].replace(". . .", "...")
        
        if not dataset is None:
            tam_sentences_file = dataset[1]
        else:
            with open(self.get_dataset_filename(split, self.TGT_LANGUAGE), 'r') as l2:
                # 2-character and 3-character alphabets are not \w (words) in re, switching to string.punctuation

                tam_sentences_file = list(l2.readlines())
            
        eng_words, tam_sentences = set(), []
        for idx, sentence in enumerate(tam_sentences_file):
        
            line = re.sub('\d+', ' %s ' % self.reserved_tokens[self.NUM_IDX], sentence.lower()) # use NUM reserved token

            if not symbols:
                line = line.translate(translator) # remove punctuations
            else:
                for ch in string.punctuation:
                    line = line.replace(ch, " "+ch+" ")
                
                for sym in symbol_replacements:
                    line = line.replace(sym, " "+symbol_replacements[sym]+" ")

            for sym in unnecessary_symbols:
                if not symbols or (symbols and not sym in symbol_replacements.keys()):
                    line = line.replace(sym, "") 
            
            line = re.sub("\s+", " ", line) # correct for number of spaces
            
            if dataset is None:
                p = re.compile("([a-z]+)\s|([a-z]+)")
                search_results = p.search(line)
                if not search_results is None:
                    eng_tokens = [x for x in search_results.groups() if not x is None]
                    eng_words.update(eng_tokens)

                    with open(self.get_dataset_filename("train", "en", subdir="tamil_eng_vocab_untokenized"), 'a') as f:
                        f.write("%s\n" % (eng_sentences[idx]))
                    with open(self.get_dataset_filename("train", "ta", subdir="tamil_eng_vocab_untokenized"), 'a') as f:
                        f.write("%s\n" % (sentence))

            # some english words show up in tamil dataset (lower case)
            line = re.sub("[a-z]+\s*", " %s " % self.reserved_tokens[self.ENG_IDX], line) # use ENG reserved token
            line = re.sub("\s+", " ", line) # correct for number of spaces
            
            line = line.replace(". . .", "...")
            tam_sentences.append(line.strip())
        
        if dataset is None:
            for eng, tam in zip(eng_sentences, tam_sentences):
                text_pairs.append((eng, tam))
        
            random.shuffle(text_pairs)
        
            return text_pairs, eng_words
        
        else: #word2vec
            
            return [eng_sentences, tam_sentences], eng_words

    def create_token_sentences_for_word2vec(self, eng_words):
        
        # DEBUG
        # tamil sentence has no english words for transfer to english vocabulary
        if len(eng_words) == 0:
            eng_words = ["START"]

        if len(eng_words) < self.num_token_sentences:
            eng_words = list(np.tile(eng_words, self.num_token_sentences//len(eng_words) + 1)[:self.num_token_sentences])

        # instantiate for train set only
        self.eng_words = eng_words
        
        self.reserved_token_sentences = []
        for idx in range(len(eng_words)):
            if self.start_stop_tokens:
                string="%s " % self.reserved_tokens[self.BOS_IDX]
            else:
                string = ""
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.NUM_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            string += ("%s " % self.reserved_tokens[self.UNK_IDX]) * np.random.randint(0,3)
            string += "%s " % eng_words[idx] if np.random.randint(0,2) else ""
            if self.start_stop_tokens:
                string += "%s " % self.reserved_tokens[self.EOS_IDX]
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string += ("%s " % self.reserved_tokens[self.PAD_IDX]) * np.random.randint(0,3)
            string = string.strip()
            
            src_string = string.replace(self.reserved_tokens[self.UNK_IDX], eng_words[idx])
            trg_string = string.replace(eng_words[idx], self.reserved_tokens[self.ENG_IDX])
            self.reserved_token_sentences.append((src_string, trg_string))

    def create_vocabulary(self, sentences, language):
        
        assert language in ['en', 'ta']

        for idx in range(len(sentences)):
            if language == 'en':
                sentences[idx] = unidecode(sentences[idx]) # remove accents from english sentences
                # Refer FAQs here: https://pypi.org/project/Unidecode/
                sentences[idx] = sentences[idx].replace("a<<", "e") # umlaut letter
                sentences[idx] = sentences[idx].replace("a 1/4", "u") # u with diaeresis
                sentences[idx] = sentences[idx].replace("a3", "o") # ó: a3 --> o
                
                sentences[idx] = sentences[idx].replace("a(r)", "i") # î: a(r) --> i
                sentences[idx] = sentences[idx].replace("a-", "i") # ï: a- --> i [dataset seems to also use ocr: ïடக்கக்கூடியதுதான்  --> i(da)kka for padi[kka]]
                sentences[idx] = sentences[idx].replace("a$?", "a") # ä: a$? --> a
                sentences[idx] = sentences[idx].replace("a'", "o") # ô: a' --> o
                sentences[idx] = sentences[idx].replace("d1", "e") # econostrum - single token
                sentences[idx] = sentences[idx].replace("a+-", "n") # ñ: a+- --> n
                sentences[idx] = sentences[idx].replace("a1", "u") # ù: a1 --> u
            
                # manual change
                num_and_a_half = lambda x: "%s%s" % (self.reserved_tokens[self.NUM_IDX], x) # NUM a half --> NUM and a half
                sentences[idx] = sentences[idx].replace(num_and_a_half(" a 1/2"), num_and_a_half(" and a half"))
            
            if self.start_stop_tokens:
                sentences[idx] = self.reserved_tokens[self.BOS_IDX] + ' ' + sentences[idx] + ' ' + self.reserved_tokens[self.EOS_IDX]
            sentences[idx] = re.sub('\s+', ' ', sentences[idx])

        if hasattr(self, "reserved_token_sentences"):
            if language == 'en':
                sentences.extend([x[0] for x in self.reserved_token_sentences])
            elif language == 'ta':
                sentences.extend([x[1] for x in self.reserved_token_sentences])
        
        # English
        # 149309 before tokenization ; 75210 after
        # 70765 tokens without symbols
        # 67016 tokens without numbers
        
        # nltk vs stanza: 66952 vs 66942 tokens

        # Tamil
        # 271651 tokens with English words
        # 264429 tokens without English words (ENG tag)
        
        vocab = set()
        word_counts = {}
        
        virama_introduction_chars = {"ங": "ங்"}

        if hasattr(self, "eng_tokens") and language=="en":
            vocab.update(self.eng_tokens)

        for idx, sentence in enumerate(sentences):
            if idx == len(sentences) - self.num_token_sentences and hasattr(self, 'reserved_token_sentences'):
                if language == "en":
                    vocab.update(list(
                        set(self.reserved_tokens) - \
                            set([self.reserved_tokens[self.ENG_IDX], self.reserved_tokens[self.UNK_IDX]])))
                else:
                    vocab.update(self.reserved_tokens)
                break
            else:
                if language == 'en':
                    #tokens = nltk.word_tokenize(sentence)
                    doc = en_nlp(sentence)
                    if len(doc.sentences) > 1:
                        tokens = [x.text for x in doc.sentences[0].tokens]
                        for sent in doc.sentences[1:]:
                            tokens.extend([x.text for x in sent.tokens])
                    else:
                        try:
                            tokens = [x.text for x in doc.sentences[0].tokens]
                        except IndexError:
                            tokens = []
                elif language == 'ta':
                    # stanza gives tokens of single alphabets that don't make semantic sense and increases vocab size
                    # Because of data preprocessing and special character removal, stanza doesn't do much for tokenizing tamil
                    
                    # DEBUG
                    # sentence = get_en_unicode_tokenized_sentence(sentence, self.tamil_unicode_hex, self.reserved_tokens[self.ENG_IDX])

                    tokens = sentence.split(' ')

                    for token_index, token in enumerate(tokens):

                        if not token in string.punctuation:
                            
                            token_languages = self.get_entam_sequence(token)
                            token_replacement = self.tokenize_entam_combinations(token_languages, token)
                            
                            if token_replacement[0] == token:
                                continue
                            else:
                                new_sentence_tokens = tokens[:token_index] + token_replacement + tokens[token_index+1:]
                                sentences[idx] = " ".join(new_sentence_tokens)

                    # if self.morephemes inside function to indicate lesser abstract functionality
                    sentences[idx] = self.get_morphologically_analysed_tamil_sentence(sentences[idx])
                    sentences[idx] = re.sub(r"\s+", " ", sentences[idx])
                    tokens = sentences[idx].split(' ')
                
                remove_tokens = []
                for token_idx, token in enumerate(tokens):
                    
                    if len(token) == 0:
                        continue

                    # use stress character (virama from wikipedia) to end tokens that need them
                    if language == "ta" and token[-1] in virama_introduction_chars.keys():
                        token = token[:-1] + virama_introduction_chars[token[-1]]
                    
                    if language == "ta":
                        langs = self.get_entam_sequence(token)
                        if len(langs) == 2 and all(["en" in key for key in langs]) and not token in string.punctuation and \
                                not token in [self.reserved_tokens[self.ENG_IDX], \
                                              self.reserved_tokens[self.NUM_IDX], \
                                              self.reserved_tokens[self.BOS_IDX], \
                                              self.reserved_tokens[self.EOS_IDX]] \
                                and not token == "...": 
                            # single word, english and not ENG token
                            remove_tokens.append(token_idx)

                    if token in vocab:
                        word_counts[token] += 1
                    else:
                        word_counts[token] = 1
                
                for tok_id in reversed(remove_tokens):
                    del tokens[tok_id]

                vocab.update(tokens)
                sentences[idx] = " ".join(tokens)
                
        if hasattr(self, "eng_vocab"):
            if language == "en":
                tokens_in_eng_vocabulary = 4 # only UNK and ENG don't belong to en vocabulary
                assert len(word_counts) == len(vocab) - (len(self.reserved_tokens) - tokens_in_eng_vocabulary), \
                        "sentence %d: Vocab size: %d, Word Count dictionary size: %d" % (idx, len(vocab), len(word_counts)) # BOS, EOS, NUM, PAD already part of sentences
            else:
                assert len(word_counts) == len(vocab), \
                        "sentence %d: Vocab size: %d, Word Count dictionary size: %d" % (idx, len(vocab), len(word_counts)) # BOS, EOS, NUM, PAD, ENG already part of sentences

        return vocab, word_counts, sentences

    def tokenize_entam_combinations(self, token_languages, token):
        
        if token in self.reserved_tokens:
            return [token]

        tokens_split, tamil_part = [], ""
        keys = list(token_languages.keys())
        for idx, key in enumerate(reversed(keys[:-1])):
            lang = "en" if "en" in key else "ta"
            start_of_lang_block = token_languages[key]
            end_of_lang_block = token_languages[keys[len(keys)-1 - idx]]
            
            if lang == "en":
                if end_of_lang_block - start_of_lang_block >= 3:
                    if tamil_part == "":
                        tokens_split.append(self.reserved_tokens[self.ENG_IDX])
                    else:
                        tokens_split.extend([tamil_part, self.reserved_tokens[self.ENG_IDX]])
                        tamil_part = ""
            else:
                tamil_part = token[start_of_lang_block:end_of_lang_block] + tamil_part
        
        if tamil_part != "":
            tokens_split.append(tamil_part)
        else:
            # no tamil characters means <=2 character english token
            tokens_split.append(self.reserved_tokens[self.ENG_IDX])

        tokens_split = list(reversed(tokens_split))

        return tokens_split
    
    def get_entam_sequence(self, token):
        
        if not hasattr(self, "tamil_characters_hex"):
            self.tamil_characters_hex = return_tamil_unicode_isalnum()
        
        sequence = OrderedDict()
        num_eng, num_tam = 0, 0
        get_count = lambda lang: str(num_eng) if lang=='en' else str(num_tam)

        if check_unicode_block(token[0], self.tamil_characters_hex):
            lang = 'ta'
            num_tam += 1
        else:
            lang = 'en'
            num_eng += 1

        sequence[lang+"0"] = 0

        for idx, character in enumerate(list(token)[1:]):

            if check_unicode_block(character, self.tamil_characters_hex):
                if lang == 'en':
                    lang = 'ta'
                    sequence[lang+get_count(lang)] = idx + 1
                    num_tam += 1
            else:
                if lang == 'ta':
                    lang = 'en'
                    sequence[lang+get_count(lang)] = idx + 1
                    num_eng += 1

        sequence[lang+get_count(lang)] = len(token)
        return sequence    

#TODO: Batch sequence from lowest to biggest bucket
class BucketingBatchSampler(Sampler):
    def __init__(self, bucketing_indices, batch_size):
        self.bucketing_indices = bucketing_indices
        self.batch_size = batch_size
    
    def __len__(self):

        necessary_bucketing_indices = sum([int(x[1]-x[0] < self.batch_size) for x in self.bucketing_indices])
        
        if necessary_bucketing_indices > 0: # val and test sets potentially
            factor = len(self.bucketing_indices) # ensure all examples are sampled by increasing dataloader length by 2x num_buckets
        else:
            factor = 1

        return (self.bucketing_indices[-1][1] + (self.batch_size * factor) - 1) // self.batch_size
    
    def __iter__(self):
        for _ in range(len(self)):
            bucket_sample = torch.randint(low=0, high=len(self.bucketing_indices), size=(1,))
            start, end = self.bucketing_indices[bucket_sample]

            if end - start < self.batch_size:
                ret = list(range(start, end)) * ((self.batch_size // (end - start)) + 1)
                ret = ret[:self.batch_size]
                yield ret
            else:
                start_idx = torch.randint(low=start, high=end+1-self.batch_size, size=(1,))
                yield range(start_idx, start_idx+self.batch_size)

if __name__ == "__main__":
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", help="Verbose flag for dataset stats", action="store_true")
    ap.add_argument("--nosymbols", "-ns", help="Symbols flag for eliminating symbols from dataset", action="store_true")
    ap.add_argument("--no_start_stop", "-nss", help="Remove START and STOP tokens", action="store_true")
    ap.add_argument("--morphemes", "-m", help="Morphemes flag for morphological analysis", action="store_true")
    ap.add_argument("--batch_size", "-b", help="Batch size (int)", type=int, default=64)
    args = ap.parse_args()

    train_dataset = EnTamV2Dataset("train", symbols=not args.nosymbols, verbose=args.verbose, morphemes=args.morphemes, start_stop_tokens=not args.no_start_stop)
    eng_vocab, tam_vocab = train_dataset.return_vocabularies()
    val_dataset = EnTamV2Dataset("dev", symbols=not args.nosymbols, verbose=args.verbose, morphemes=args.morphemes, 
                                  vocabularies=(eng_vocab, tam_vocab), start_stop_tokens=not args.no_start_stop)
    #test_dataset = EnTamV2Dataset("test", symbols=not args.nosymbols, verbose=args.verbose, morphemes=args.morphemes, 
    #                              vocabularies=(eng_vocab, tam_vocab), start_stop_tokens=not args.no_start_stop)
    
    from torch.utils.data import DataLoader

    bucketing_batch_sampler = BucketingBatchSampler(val_dataset.bucketing_indices, batch_size=args.batch_size)
    dataloader = DataLoader(val_dataset, batch_sampler=bucketing_batch_sampler)
    
    #for idx, (src, tgt) in enumerate(train_dataset):
    #    print (idx, src.shape, tgt.shape, src.min(), src.max(), tgt.min(), tgt.max())
    #for idx, (src, tgt) in enumerate(dataloader):
    #    print (idx, src.shape, tgt.shape, src.min(), src.max(), tgt.min(), tgt.max())
    
    # Display all data before training
    for x, y in dataloader:
        for x_i, y_i in zip(x,y):
            print (train_dataset.vocab_indices_to_sentence(x_i, "en"))
            print (train_dataset.vocab_indices_to_sentence(y_i, "ta"))

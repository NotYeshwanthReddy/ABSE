from ABSE.utils.utils import get_domain_sentiment_words_set, read_data, merge_results, normalize_results
from ABSE.src.rules import Rules
import spacy
import pandas as pd
from tqdm import tqdm
import string
import nltk

class ABSE:
    def __init__(self, file_name:str,
                 id_column='sid',
                 text_column='text',
                 normalize_results=True):
        # loading rules
        self.rules = Rules()
        # loading sentiment words from vader lexicon
        self.domain_sentiment_words_set = get_domain_sentiment_words_set()
        self.domain_aspect_words_set = set()
        # declaring class variables
        self.id_column = id_column
        self.text_column = text_column
        # loading the language model for dependency parsing
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('punkt')
        # creating dataframe with final results
        data = read_data(file_name, id_column, text_column)
        # loading data (a list of comments/paragraphs)
        self.documents = data[text_column].tolist()
        # saving sid and text into results for human understandability.
        self.results = data
        self.normal_results = pd.DataFrame()
    
    def save_domain_sentiment_words_set(self, file_name="sentiments_set.txt"):
        with open(file_name, 'w') as file:
            sentiment_words = "\n".join(list(self.domain_sentiment_words_set))
            file.write(sentiment_words)
        print("Saved to {}".format(file_name))
        return True
    
    def save_domain_aspect_words_set(self, file_name="aspects_set.txt"):
        with open(file_name, 'w') as file:
            aspect_words = "\n".join(list(self.domain_aspect_words_set))
            file.write(aspect_words)
        print("Saved to {}".format(file_name))
        return True
    
    def run_rule_1(self):
        """
        Extracts aspect words using the sentiment words set.
        """
        aspect_words = []
        # content level aspects list
        r11,r12 = [],[]
        for content in tqdm(self.documents, desc="Rule1"):
            sentences = nltk.tokenize.sent_tokenize(str(content))
            # paragraph level aspects list
            r11_,r12_ = [],[]
            # divide paragraph at sentence level
            for sentence in sentences:
                # remove punctuations
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                # dependency parsing
                parsed_sentence = self.nlp(sentence)
                asp_sent_pairs, new_aspect_words = self.rules.rule1_1(parsed_sentence, 
                                                                 self.domain_sentiment_words_set)
                # adding aspects to the list
                r11_.extend(asp_sent_pairs)
                aspect_words.extend(new_aspect_words)
                asp_sent_pairs, new_aspect_words = self.rules.rule1_2(parsed_sentence, 
                                                                 self.domain_sentiment_words_set)
                # adding sentence level results to the list.
                r12_.extend(asp_sent_pairs)
                aspect_words.extend(new_aspect_words)
            # adding row level results to the list
            r11.append(r11_);r12.append(r12_)
        self.results['r11']=r11; self.results['r12']=r12;
        # adding aspect words to the existing list.
        self.domain_aspect_words_set =  self.domain_aspect_words_set.union(set(aspect_words))
    
    def run_rule_2(self):
        """
        Extracts sentiment words using the aspect words set.
        """
        sentiment_words = list(self.domain_sentiment_words_set)
        # content level sentiment list
        r21,r22 = [],[]
        for content in tqdm(self.documents, desc="Rule2"):
            sentences = nltk.tokenize.sent_tokenize(str(content))
            # paragraph level sentiment list
            r21_,r22_ =  [],[]
            # divide paragraph at sentence level
            for sentence in sentences:
                # remove punctuations
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                parsed_sentence = self.nlp(sentence)
                # dependency parsing
                asp_sent_pairs, new_sentiment_words = self.rules.rule2_1(parsed_sentence, 
                                                                         self.domain_aspect_words_set)
                # adding sentiments to the list
                r21_.extend(asp_sent_pairs)
                sentiment_words.extend(new_sentiment_words)
                asp_sent_pairs, new_sentiment_words = self.rules.rule2_2(parsed_sentence, 
                                                                         self.domain_aspect_words_set)
                # adding sentence level results to the list. 
                r22_.extend(asp_sent_pairs)
                sentiment_words.extend(new_sentiment_words)
            # adding row level results to list
            r21.append(r21_);r22.append(r22_)
        self.results['r21']=r21; self.results['r22']=r22;
        # adding sentence words to the existing list.
        self.domain_sentiment_words_set = self.domain_sentiment_words_set.union(set(sentiment_words))
    
    def run_rule_3(self):
        """
        Extracts more aspect words using existing aspects words set.
        """
        # creating empty lists to contain results
        r31,r32 = [],[]
        # iterating row wise data
        for content in tqdm(self.documents, desc="Rule3"):
            # tokenize paragraphs
            sentences = nltk.tokenize.sent_tokenize(str(content))
            r31_,r32_ =  [],[]
            # breaking paragraphs to sentences
            for sentence in sentences:
                # remove punctuations
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                parsed_sentence = self.nlp(sentence)
                # running rule R_31
                new_words = self.rules.rule3_1(parsed_sentence, self.domain_aspect_words_set)
                r31_.extend(new_words)
                # running rule R_32
                new_words = self.rules.rule3_2(parsed_sentence, self.domain_aspect_words_set)
                # saving results of one sentence
                r32_.extend(new_words)
            # saving results of one row
            r31.append(r31_); r32.append(r32_)

        self.results['r31']=r31; self.results['r32']=r32;
        new_aspect_words = [aspect for aspects in r31 for aspect in aspects]
        new_aspect_words.extend([aspect for aspects in r32 for aspect in aspects])
        self.domain_aspect_words_set = self.domain_aspect_words_set.union(set(new_aspect_words))
    
    def run_rule_4(self):
        """
        Extracts more sentiment words using existing sentiment words set.
        """
        r41,r42 = [],[]

        for content in tqdm(self.documents, desc="Rule4"):
            sentences = nltk.tokenize.sent_tokenize(str(content))
            r41_,r42_ = [],[]
            for sentence in sentences:
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                parsed_sentence = self.nlp(sentence)

                new_words = self.rules.rule4_1(parsed_sentence, self.domain_sentiment_words_set)
                r41_.extend(new_words)
                new_words = self.rules.rule4_2(parsed_sentence, self.domain_sentiment_words_set)
                r42_.extend(new_words)
            r41.append(r41_); r42.append(r42_)

        self.results['r41']=r41; self.results['r42']=r42;
        new_sentiment_words = [sentiment for sentiments in r41 for sentiment in sentiments]
        new_sentiment_words.extend([sentiment for sentiments in r42 for sentiment in sentiments])
        self.domain_sentiment_words_set = self.domain_sentiment_words_set.union(set(new_sentiment_words))
    
    def run_syntactic_relations(self):
        """
        Developed by Varsha
        """
        all_aspects = []
        for content in tqdm(self.documents, desc="Syntactic Relations 1"):
            document = self.nlp(str(content))
            aspects = self.rules.get_syntactic_relations1(document, self.nlp)
            all_aspects.append(aspects)
        self.results['syntactic relations1'] = all_aspects
        
        all_aspects = []
        for content in tqdm(self.documents, desc="Syntactic Relations 2"):
            document = self.nlp(str(content))
            aspects = self.rules.get_syntactic_relations2(document, self.nlp)
            all_aspects.append(aspects)
        self.results['syntactic relations2'] = all_aspects

    def get_noun_phrases(self):
        """
        Developed by Varsha
        """
        all_phrases =[]
        for content in tqdm(self.documents, desc="noun_phrases_rule_based"):
            document = self.nlp(str(content))
            all_phrases.append(self.rules.rule_based_noun_phrase_extraction(document))
        self.results['noun phrase_rule base']= all_phrases

        all_phrases =[]
        for content in tqdm(self.documents, desc="noun_phrases_spacy_based"):
            document = self.nlp(str(content))
            all_phrases.append(self.rules.spacy_noun_phrase_extraction(document))
        self.results['noun phrase_spacy']= all_phrases
        
    def run_all_rules(self, 
                      expand_word_sets=False, 
                      expand_sentiments=False, 
                      expand_aspects=False):
        """
        Runs all the rules on the given dataset.
        R4 is executed first to expand domain based sentiment words.
        R1 is executed to find aspects based on sentiments.
        R3 is executed to expand domain based aspect words.
        R2 is executed to find sentiment words based on aspects.
        
        run_syntactic_relations and noun_phrases are extracted using rules shared by Varsha
        """
        if expand_word_sets or expand_sentiments:
            self.run_rule_4()
        self.run_rule_1()
        if expand_word_sets or expand_aspects:
            self.run_rule_3()
        self.run_rule_2()
        self.run_syntactic_relations()
        self.get_noun_phrases()

        # merge all output columns 
        if expand_word_sets or expand_aspects:
            self.results['abse_pairs'], all_aspects = merge_results(self.results,
                                                                    ["r11","r12",
                                                                    "r21","r22",
                                                                    "r31","r32",
                                                                    "syntactic relations1",
                                                                    "syntactic relations2",
                                                                    "noun phrase_spacy"])
        # excluding r31 and r32 
        else:
            self.results['abse_pairs'], all_aspects = merge_results(self.results,
                                                                   ["r11","r12",
                                                                    "r21","r22",
                                                                    "syntactic relations1",
                                                                    "syntactic relations2",
                                                                    "noun phrase_spacy"])
        # convert data to normal form. 
        self.normal_results = normalize_results(self.results, 
                                                all_aspects, 
                                                self.domain_sentiment_words_set)
        
from ABSE.utils.utils import get_MR
from nltk.corpus import stopwords


class Rules:
    def __init__(self):
        self.MR = get_MR()
        self.stopwords = set(stopwords.words('english'))

    def rule1_1(self, parsed_sentence, domain_sentiment_words_set):
        """
        O -> O-Dep -> A
        O : {O} 
        O-Dep : {MR}
        POS(A) : {'NN'}
        """
        sentiment_aspect_pairs_ = []
        new_aspect_words = []
        for O in parsed_sentence:
            if (str(O).lower() in domain_sentiment_words_set) and \
                (O.dep_ in self.MR) and \
                (O.head.pos_ == "NOUN"):
    #             print("{} -> {} -> {}".format(O, 
    #                                           O.dep_, 
    #                                           O.head))
                sentiment_aspect_pairs_.append((O.head.text, O.text))
                new_aspect_words.append(O.head.text.lower())
        return sentiment_aspect_pairs_, new_aspect_words

    def rule1_2(self, parsed_sentence, domain_sentiment_words_set):
        """
        O -> O-Dep -> H <- A-Dep <- A
        O : {O}
        O/A-Det : {MR}
        POS(A) : {'NOUN'}
        """
        sentiment_aspect_pairs_ = []
        new_aspect_words = []
        for O in parsed_sentence:
            for A in parsed_sentence:
                if O==A:
                    continue
                elif (O.head == A.head) and \
                    (str(O).lower() in domain_sentiment_words_set) and \
                    ((O.dep_ in self.MR) and \
                    (A.dep_ in self.MR)) and \
                    (A.pos_=="NOUN"):
    #                 print("{} -> {} <- {}".format(O,O.head,A))
                    sentiment_aspect_pairs_.append((A.text, O.text))
                    new_aspect_words.append(A.text.lower())
        return sentiment_aspect_pairs_, new_aspect_words
    
    def rule2_1(self, parsed_sentence, domain_aspect_words_set):
        """
        O -> O-Dep -> A
        A : {A}
        O-Dep : {MR}
        POS(A) : {NN}
        """
        new_sentiment_words = []
        sentiment_aspect_pairs_ = []
        for O in parsed_sentence:
            if (str(O.head).lower() in domain_aspect_words_set) and \
                (O.dep_ in self.MR) and \
                (O.pos_ == "ADJ"):
    #             print("{} -> {} -> {}".format(O, 
    #                                           O.dep_, 
    #                                           O.head))
                sentiment_aspect_pairs_.append((O.head.text, O.text))
                new_sentiment_words.append(O.text.lower())
        return sentiment_aspect_pairs_, new_sentiment_words

    def rule2_2(self, parsed_sentence, domain_aspect_words_set):
        """
        O->O-Dep->H<-A-Dep<-A
        A : {A}
        O-Dep: {MR}
        A-Dep: {MR}
        POS(O): JJ
        """
        new_sentiment_words = []
        sentiment_aspect_pairs_ = []
        for O in parsed_sentence:
            for A in parsed_sentence:
                if O==A:
                    continue
                elif (O.head.i == A.head.i) and \
                    (str(A).lower() in domain_aspect_words_set) and \
                    (O.dep_ in self.MR) and \
                    (A.dep_ in self.MR) and \
                    (O.pos_ == "ADJ"):
    #                 print("{} -> {} <- {}".format(O, 
    #                                               O.dep_, 
    #                                               A))
                    sentiment_aspect_pairs_.append((O.head.text, O.text))
                    new_sentiment_words.append(O.text.lower())
        return sentiment_aspect_pairs_, new_sentiment_words

    def rule3_1(self, parsed_sentence, domain_aspect_words_set):
        """
        Ai(j)->Ai(j)-Dep->Aj(i)
        Aj(i): {A}
        Ai(j)-Dep: {CONJ},
        POS(Ai(j)):{NN}
        """
        new_aspect_words = []
        for Aij in parsed_sentence:
            for Aji in parsed_sentence:
                if Aij==Aji:
                    continue
                elif (Aij.head == Aji) and \
                    (str(Aji).lower() in domain_aspect_words_set) and \
                    (Aij.dep_=='conj') and \
                    (Aij.pos_=="NOUN"):
    #                 print("{} -> {} -> {}".format(Aij, 
    #                                               Aij.dep_, 
    #                                               Aij.head))
                    new_aspect_words.append((Aij.text.lower()))
                    # domain_aspect_words_set.add(str(Aij).lower())
        return new_aspect_words
    
    def rule3_2(self, parsed_sentence, domain_aspect_words_set):
        """
        Ai->Ai-Dep->H<-Aj-Dep<-Aj
        Ai:{A}, 
        (Ai-Dep == Aj-Dep) OR (Ai-Dep: subj AND Aj-Dep:'obj')
        POS(Aj): {NN}
        """
        new_aspect_words = []
        for Ai in parsed_sentence:
            for Aj in parsed_sentence:
                if Ai==Aj:
                    continue
                elif (Ai.head==Aj.head) and \
                    (str(Ai).lower() in domain_aspect_words_set) and \
                    ((Ai.dep_ == Aj.dep_) or (Ai.dep_=='subj' and Aj.dep_=='obj')) and \
                    (Aj.pos_=="NOUN"):
    #                 print("{} -> {} -> {}".format(Ai, 
    #                                           Ai.dep_, 
    #                                           Ai.head))
                    new_aspect_words.append((Aj.text.lower()))
                    # domain_aspect_words_set.add(str(Aj).lower())
        return new_aspect_words
    
    def rule4_1(self, parsed_sentence, domain_sentiment_words_set):
        """
        Oi(j)->Oi(j)-Dep->Oj(i)
        Oj(i):{O}
        Oi(j)-Dep:{CONJ}
        POS(Oi(j)):{JJ}
        """
        new_sentiment_words = []
        for Oij in parsed_sentence:
            for Oji in parsed_sentence:
                if Oij==Oji:
                    continue
                elif (Oij.head == Oji) and \
                    (str(Oji).lower() in domain_sentiment_words_set) and \
                    (Oij.dep_=='conj') and \
                    (Oij.pos_=="ADJ"):
    #                 print("{} -> {} -> {}".format(Oij, 
    #                                               Oij.dep_, 
    #                                               Oij.head))
                    new_sentiment_words.append((Oij.text.lower()))
        return new_sentiment_words
    
    def rule4_2(self, parsed_sentence, domain_sentiment_words_set):
        """
        Oi->Oi-Dep->H<-Oj-Dep<-Oj
        Oi:{O}
        (Oi-Dep==Oj-Dep) OR (Oi/Oj-Dep: {pnmod, mod})
        POS(Oj):{JJ}
        """
        new_sentiment_words = []
        for Oi in parsed_sentence:
            for Oj in parsed_sentence:
                if Oi==Oj:
                    continue
                elif (Oi.head==Oj.head) and \
                    ((Oi.dep_ == Oj.dep_) or (Oi.dep_ in ['pnmod','mod'] and Oj.dep_ in ['pnmod','mod'])) and \
                    (Oj.pos_=="ADJ"):
                    
    #                 print("{} -> {} -> {}".format(Oi, 
    #                                           Oi.dep_, 
    #                                           Oi.head))
                    new_sentiment_words.append((Oj.text.lower()))
        return new_sentiment_words

    def get_syntactic_relations1(self, parsed_document, parser):
        #This rule is based on noun-adjective syntatic relation. 
        sentiment_aspect_pairs_ = []
        #parse the sentences
        for sent in parsed_document.sents:
            parsed_sentence = parser(sent.text)
            sentiment_term = ''
            target = ''
            #parse the token from sentence
            for token in parsed_sentence:
                #If subject is noun and subj and its child id adjective
                if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
                    target = token.text
                for child in token.children:
                    if token.pos_ == 'ADJ':
                        sentiment_term = child.text
            if(len(target)>1):
                sentiment_aspect_pairs_.append((target, sentiment_term))
        return sentiment_aspect_pairs_

    def get_syntactic_relations2(self, parsed_document, parser):
        #This rule is based on noun-adjective syntatic relation. 
        sentiment_aspect_pairs_ = []
         #parse the sentences
        for sent in parsed_document.sents:
            #print(sent.text)
            sent = parser(sent.text)
            sentiment_term = ''
            target = ''
            #parse the token from sentence
            for token in sent:
                #If object is noun and obj and its child id adjective
                if token.dep_ == 'pobj' and token.pos_ == 'NOUN':
                    target = token.text
                    for child in token.children:
                        if child.pos_ == 'ADJ':
                            sentiment_term = child.text
                            sentiment_aspect_pairs_.append((target, sentiment_term))
                            #print(target, " : ", descriptive_term)
        return sentiment_aspect_pairs_

    def rule_based_noun_phrase_extraction(self, parsed_document):
        #This rule identify the aspect based on noun chunks
        noun_pharses=set()   
        #parse the noun chunks
        for nc in parsed_document.noun_chunks:
            #check if left and right edge is noun , then add to the aspect term
            for np in [nc, parsed_document[nc.root.left_edge.i:nc.root.right_edge.i+1]]:
                if ((np not in (self.stopwords)) & (len(np) > 2)):
                        noun_pharses.add(np)
        return noun_pharses
    
    def spacy_noun_phrase_extraction(self, parsed_document):
        #This rule identify the aspect based on noun chunks
        noun_pharses=set()
        #parse the noun chunks
        for noun_chunk in parsed_document.noun_chunks:
            #check length of chunk and it should not be stopword
            if ((noun_chunk.text not in (self.stopwords)) & (len(noun_chunk.text) > 2)):
                noun_pharses.add(noun_chunk.text)
        return noun_pharses
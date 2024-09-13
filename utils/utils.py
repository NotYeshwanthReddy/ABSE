import pandas as pd
import pathlib
from tqdm import tqdm


def get_MR():
    return ('amod', 'mod', 'nsubj', 'subj', 'obj', 'dobj', 'desc', 'pnmod', 'conj')

def read_data(file_name = 'sample_data/data.xlsx', 
              id_column = 'sid', 
              text_column = 'text'):
    # reading data
    file_suffix = pathlib.PurePosixPath(file_name).suffix
    if file_suffix=='.xlsx':
        data = pd.read_excel(file_name)
    elif file_suffix=='.csv':
        data = pd.read_csv(file_name)
    else:
        print("Unsupported file format.")
        return 0
    # picking required columns
    data = data[[id_column, text_column]]
    # drop empty text rows
    data.dropna(subset = [text_column], inplace=True)
    return data

def get_domain_sentiment_words_set():
    """
    Extract sentiment words from vader lexicon text file.
    """
    # empty set
    domain_sentiment_words_set = set()

    # reading file
    vader_lexicon = ''
    with open("ABSE/utils/vader_lexicon.txt", "r") as f:
        vader_lexicon = f.read()

    # splitting into lines
    lines = vader_lexicon.split("\n")

    sentiment_words = {}
    other_words = {}

    for i in range(len(lines[:-1])):
        items = lines[i].split("\t")
    #     print(i,len(items), items)
        word, avg_score = items[0],items[1]
    #     print(word, avg_score)
        if word.isalpha():
            sentiment_words[word] = avg_score
        else:
            other_words[word] = avg_score

    for word in sentiment_words.keys():
        domain_sentiment_words_set.add(str(word))

    return domain_sentiment_words_set

def merge_results(data:pd.DataFrame(), 
                  columns_to_merge=["r11","r12","r21","r22","r31","r32","syntactic relations1","syntactic relations2","noun phrase_spacy"]):
    """
    Takes in all rule results and creates a single list with aspect-sentiment pairs.
    [('aspect1', 'sentiment1'), ('aspect2', 'sentiment2'),...]
    """
    req = data.loc[:,columns_to_merge]
    all_results = []
    all_aspects = []
    for row in req.iterrows():
        row = row[1]
        row_results = []

        for i in range(len(row)):
            # filtering empty values
            if len(row[i])==0:
                continue
            elif type(row[i])==list:
                # adding results of r31, r32
                if len(row[i][0])==1:
                    for item in row[i]:
                        row_results.append((item[0], ''))
                        all_aspects.append(item[0])
                # adding results of r11, r12, r21, r22, syntactic_relations1, syntactic_relations2
                elif len(row[i][0])==2:
                    row_results.extend(row[i])
                    all_aspects.extend([i[0] for i in row[i]])
            # adding results from column 'noun_phrase_spacy'
            elif type(row[i])==set:
                for item in row[i]:
                    row_results.append((item, ''))
                    all_aspects.append(item)
            else:
                pass
        all_results.append(row_results)
    return all_results, set(all_aspects)

def normalize_results(data: pd.DataFrame(),
                      all_aspects:set(),
                      all_sentiments:set(),
                      id_column='sid',
                      results_column='abse_pairs'):
    ndf = pd.DataFrame(columns=['abse_id', 'sid', 'aspect_id', 'aspect', 'sentiment_id', 'sentiment'])
    
    aspects_id_map = {}
    for i,aspect in enumerate(all_aspects):
        aspects_id_map[aspect] = i
    sentiment_id_map = {}
    for i, sentiment in enumerate(all_sentiments):
        sentiment_id_map[sentiment] = i
    
    abse_id = 0
    
    for i in tqdm(data.index):
        for as_pair in data.iloc[i][results_column]:
            aspect = as_pair[0]
            sentiment = as_pair[1]
            try:
                sid = sentiment_id_map[sentiment.lower()]
            except:
                sid = -1
            ndf.loc[len(ndf.index)] = [abse_id, 
                                       data.iloc[i][id_column],
                                       aspects_id_map[aspect],
                                       aspect,
                                       sid,
                                       sentiment]
            abse_id+=1

    return ndf
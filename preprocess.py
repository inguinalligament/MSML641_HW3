################################################################################################
####        TITLE: MSML HW3                                                                 ####
####        DESCRIPTION: SENTIMENT ANALYSIS - PREPROCESS.PY                                 ####
####        AUTHOR: BRADLEY SCOTT                                                           ####
####        UMD ID: 119 775 028                                                             ####
####        DATE: 26OCT2025                                                                 ####
####        REFERENCES USED (see paper for full details):                                   ####
####            ChatGPT 5                                                                   ####
################################################################################################

'''
[BS10262025] pre3_641_000001
[BS10262025] import all necessary modules
'''
import re
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

'''
[BS10262025] pre3_641_000005
[BS10262025] Build a function to clean the text
'''
def clean_text(s):
    s = s.lower() # lowercase all text
    s = re.sub(r"<br\s*/?>", " ", s) # replace line breaks with space
    s = re.sub(r"<.*?>", " ", s) # replace any other HTML tags with space
    s = re.sub(r"http\S+", " ", s) # replace URLs and hyperlinks with space
    s = re.sub(r"[^a-z0-9\s]", " ", s) # replace any character that is not lowercase letter
                                       # a digit or a whitespace with a space
    s = re.sub(r"\s+", " ", s).strip() # remove any extra whitespace
    return s

'''
[BS10262025] pre3_641_000010
[BS10262025] Build a function to load the data
'''
def load_and_preprocess(path, vocab_size=10000, test_size=0.5, random_state=42):
    df = pd.read_csv(path)
    df['review'] = df['review'].astype(str).apply(clean_text)
    df['label'] = (df['sentiment'].str.lower() == 'positive').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['label'],
        test_size=test_size, stratify=df['label'], random_state=random_state
    )

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    return X_train, X_test, y_train, y_test, tokenizer
        
'''
[BS10262025] pre3_641_000015
[BS10262025] Build a function to pad/truncate data to the sequence length 
'''
def pad_data(X_train, X_test, seq_len):
    X_train = pad_sequences(X_train, maxlen=seq_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=seq_len, padding='post', truncating='post')
    return X_train, X_test
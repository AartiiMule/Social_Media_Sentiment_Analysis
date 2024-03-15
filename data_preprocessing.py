
import contractions
from unidecode import unidecode
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# preprocessing
# 1. remove spaces, newlines

def remove_spaces(data):
    clean_text = data.replace('\\n',' ').replace('\t',' ').replace('\\',' ')
    return clean_text

#2. Contraction mapping
def expand_text(data):
    expanded_text = contractions.fix(data)
    return expanded_text

#3.handling accented characters
def handling_accented(data):
    fixed_text = unidecode(data)
    return fixed_text

#4.Cleaning (tokenization,normalization,removing stopwords and special characters)
stopword_list = stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('nor')
stopword_list.remove('not')


def clean_data(data):
    tokens = word_tokenize(data)                      
    clean_text = [word.lower() for word in tokens if (word not in punctuation) and (word.lower() not in stopword_list) and (len(word)>2) and (word.isalpha())]
    return clean_text

# 5. Autocorrect
# def autocorrection(data):
#     spell = Speller(lang='en')
#     corrected_text = spell(data)
#     return corrected_text

#6. Lemmatization
def lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    final_data = []
    for word in data:
        lemmatized_word = lemmatizer.lemmatize(word)
        final_data.append(lemmatized_word)
    return final_data


def join_list(data):
    return " ".join(data)
import nltk.tokenize
import string
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
porter_stemmer = PorterStemmer()


def lemmatizer(text):  #hadi bonus man3ndi brk ida ma habbitouch tdiro radicalisation aka stemming
    a = ""
    res = re.findall(r"[a-z]+", text)
    for i in res:
        a += " " + wordnet_lemmatizer.lemmatize(i)
    return a


def replace_dollar(text):
    pattern = r"\$"
    text = re.sub(pattern, " dollar ", text)
    return(text)


def stemming(text):
    stem_text = ""
    res = re.findall(r"\b[a-z]+\b", text)
    for word in res:
        stem_text += " " + porter_stemmer.stem(word)
    return stem_text

def replace_url(text):
    pattern = r"https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]" \
              r"\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|" \
              r"https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}"
    text= re.sub(pattern, "httpaddr", text)
    return(text)


def replace_email(text):
    pattern = r"([a-z]|[0-9]+)+\@[a-z]+\.[a-z]+"
    text = re.sub(pattern, "emailaddr", text)
    return(text)


def make_lower(text):
    text = text.lower()
    return(text)

def remove_stopwords(text):  #meme hadi bonus brk
    for i in stopwords:
        pattern = r"\b" + i + r"\b"
        text = re.sub(pattern, '', text)
    return (text)

def removeBallise(text):
    pattern = r"\<.*\>"
    text = re.sub(pattern, "", text)
    return(text)


def removeblank(text):
    pattern = r"(\b|\s|\n)"
    text = re.sub(pattern, " ", text)
    return(text)


def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation]).lower()
    return punctuationfree


def remove_num(text):
    pattern = r"[0-9]"
    text = re.sub(pattern, "nombre ", text)
    return text


def preprocess(txt):
    txt = removeBallise(txt)
    txt = replace_email(txt)
    txt = replace_dollar(txt)
    txt = replace_url(txt)
    txt = remove_punctuation(txt)
    txt = removeblank(txt)
    txt = remove_num(txt)
    txt = stemming(txt)
    txt = make_lower(txt)
    return txt

def preprocess_email(emails):
    processed_emails = []
    for txt in emails:
        txt = removeBallise(txt)
        txt = replace_email(txt)
        txt = replace_dollar(txt)
        txt = replace_url(txt)
        txt = remove_punctuation(txt)
        txt = removeblank(txt)
        txt = remove_num(txt)
        txt = stemming(txt)
        txt = make_lower(txt)
        processed_emails.append(txt)
    return processed_emails

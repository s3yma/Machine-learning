words=["cared", "university", "fairly", "easiliy", "singing", "sings", 
       "sung", "singer"]

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)

from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language="english")
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words) # lanchester gives best result so far, comperatively


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print("rocks:", lemmatizer.lemmatize("rocks"))
print("corpora:", lemmatizer.lemmatize("corpora", pos="n"))
print("better:", lemmatizer.lemmatize("better", pos="a")) #pos adjective, noun etc., without this parameter it displays wrong output. a little help.
print("larger:", lemmatizer.lemmatize("larger", pos="a"))
print("worst:", lemmatizer.lemmatize("worst", pos="a"))

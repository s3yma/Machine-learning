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

#bag of the words

from sklearn.feature_extraction.text import CountVectorizer
Sentences = ["We are using the Bag of the Words model.", 
             "Bag of the Words model is used for extracting the features."]
vectorizer=CountVectorizer()
features_text = vectorizer.fit_transform(Sentences).todense()
print(vectorizer.vocabulary_)
print(features_text)




import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nlkt.corpus import games

def extract_features(word, N=2):
    last_n_letters = word[-N:]
    return{"feature":last_n_letters.lower()}

male_list = [(name,"male") for name in names.words("male.txt")]
female_list = [(name,"female") for name in names.words("female.txt")]
data=(male_list + female_list)
random.seed(5)
random.shuffle(data)
input_names=["David", "Jacob", "Swati", "Shubha"]
train_sample = int(0.8*len(data))

for i in range(1,6):
    print("\n Number of letters:", i)
    features = [(extract_features(nn,i), gender) for (nn, gender) in data]
    train_data, test_data = features[:train_sample], features[train_sample:]
    classifier = NaiveBayesClassifier.train(train_data)
    accuracy_classifier = round(100*nltk_accuracy(classifier,test_data),2)
    print("Accuracy = " + str(accuracy_classifier) + "%")

for name in input_names:
    print(name, "-->", classifier.classify(extract_features(name, i)))

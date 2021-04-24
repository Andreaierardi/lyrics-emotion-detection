import nltk
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
nltk.download('stopwords')

# Custom Transformer (Inheriting from classes)
class CleanText( BaseEstimator, TransformerMixin ):
    
    # Class Constructor
    # The class constructor is formed by a function with double underscore __ :
    # these are called 'special functions' as they have special meaning.
    # In particular the '__init__' gets called whenever
    # a new object of that class is instantiated,
    # and are used to initialize all the necessary variables.
    # In this example we initialize the language variable 'lang' with 'English'
    # and pick the SnowballStemmer as the default stemmer.
    def __init__( self, lang = "english"):
        self.lang = lang
        self.stemmer = SnowballStemmer(self.lang)
    
    # The 'fit' method here is used to instantiate the class on the 'self' variable
    # and return the object itself
    def fit( self, X, y = None ):
        return self
    
    # Custom function: this applies the stemmer just created in the '__init__'
    # part to the 'self' variable
    def clean( self, x ):
        words   = [self.stemmer.stem(word) for word in word_tokenize(x.lower()) if word.isalpha() and word not in stopwords.words("english")]
        return " ".join(words)
    
    # Method that describes what we need this transformer to do i.e. cleaning the text
    # in the 'text' column in the data frame.
    # This will be used later on in the usage of the custom transformer
    # within the pipeline.
    def transform( self, X, y = None ):
        return X["text"].apply(self.clean)

# Custom Transformer: same parts as the previous custom transformer
# This one will be used for feature extraction

class CustomFeatures( BaseEstimator, TransformerMixin ):
    
    # Class Constructor
    def __init__( self ):
        return
    
    # Return self nothing else to do here
    def fit( self, X, y = None ):
        return self
        
    # Method that describes what we need this transformer to do i.e.
    # returning length, digits and punctuations in the 'text' column in data frame
    def transform( self, X, y = None ):
        import pandas
        f           = pandas.DataFrame()
        f['len']    = X['text'].str.len()
        f['digits'] = X['text'].str.findall(r'\d').str.len()
        f['punct']  = X['text'].str.findall(r'[^a-zA-Z\d\s:]').str.len()
        return f[['len','digits','punct']]

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
nltk.download('punkt')
# FeatureUnion combines two or more pipelines or transformers
# and is very fast!
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
# Our first pipeline called 'pipe' will be formed by three 'steps' or parts:
# 1)"extract" which in turns is formed through FeatureUnion which put together two parts:
# "terms" (formed by a pipeline with the CleanText() transformer we created above
# and the TfidVectorize text vectorizing transformer from scikit-learn) and "custom"
# (formed by the CustomFeatures transformer we created above);
# 2) "select", formed by the scikit-learn transformer method "SelectKBest" for feature
# selection with a chi squared score function;
# 3) "scale", same as 2) using the StandardScaler method from scikit-learn.
# The whole pipeline will be used as pre-processing task in classifying pipelines.
# extract features
pipe_bi = Pipeline([("extract", FeatureUnion([("terms", Pipeline([('clean', CleanText()),
                                                               ('tfidf', TfidfVectorizer(ngram_range = (1,2)))])),
                                           ("custom", CustomFeatures())])),
                 ("select", SelectKBest(score_func = chi2,k = 500)),
                 ("scale", StandardScaler(with_mean = False))])


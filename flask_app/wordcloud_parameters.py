import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import warnings

warnings.filterwarnings("ignore")

"""
purpose of the code is to generate and analyze word clouds, 
which visually represent the frequency of words in a given text datase
"""
def worldcloud_generator(sentences, background_color="black", max_words=200):

    # cleaning data
    sentences = sentences.str.lower()
    # print(sentences)
    # Remove punctuation
    sentences = sentences.apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation))
    )

    # Tokenizing
    # Load stopwords
    stopwords = set(STOPWORDS)
    # join all words
    txt = " ".join(sentences)
    # Tokenizing
    tokens = word_tokenize(txt)
    # Removing stopwords
    text_clean = [word for word in tokens if word not in stopwords]

    # Generating the Word Cloud
    # Generate the text
    text = " ".join(text_clean)
    # Create a WordCloud object
    wordcloud = WordCloud(
        background_color=background_color,
        max_words=max_words,
        width=700,
        height=500,
        max_font_size=100,
        collocations=False, # Disables multi-word phrases (collocations) from being considered as a single word
    )
    # Generate the word cloud
    wordcloud.generate(text)

    return wordcloud

"""
This function extracts parameters (positions, sizes, frequencies, etc.) of words from a word cloud object.
"""
def wordcloud_params(wc):
    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # Get x and y positions
    position_x_list = []
    position_y_list = []
    for i in position_list:
        position_x_list.append(i[0])
        position_y_list.append(i[1])

    # Get the relative occurence frequencies --> word size
    size_list = []
    for i in freq_list:
        size_list.append(i * 100)

    # Return wordcloud parametres (positions, frequencies, colors...)
    return position_x_list, position_y_list, freq_list, size_list, color_list, word_list
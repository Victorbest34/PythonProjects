#!/usr/bin/env python
# coding: utf-8

# # Task 1
# 
# ---
# 
# ## Web scraping and analysis
# 
# This Jupyter notebook includes some code to get you started with web scraping. We will use a package called `BeautifulSoup` to collect the data from the web. Once you've collected your data and saved it into a local `.csv` file you should start with your analysis.
# 
# ### Scraping data from Skytrax
# 
# If you visit [https://www.airlinequality.com] you can see that there is a lot of data there. For this task, we are only interested in reviews related to British Airways and the Airline itself.
# 
# If you navigate to this link: [https://www.airlinequality.com/airline-reviews/british-airways] you will see this data. Now, we can use `Python` and `BeautifulSoup` to collect all the links to the reviews and then to collect the text data on each of the individual review links.

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[2]:


base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

reviews = []

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")


# In[3]:


df = pd.DataFrame()
df["reviews"] = reviews
df.head()


# Congratulations! Now you have your dataset for this task! The loops above collected 1000 reviews by iterating through the paginated pages on the website. However, if you want to collect more data, try increasing the number of pages!
# 
#  The next thing that you should do is clean this data to remove any unnecessary text from each of the rows. For example, "✅ Trip Verified" can be removed from each row if it exists, as it's not relevant to what we want to investigate.

# In[4]:


df


# Removing the parts before | in the reviews column

# In[5]:


df.reviews= df.reviews.str.split('|',expand=True)[1]



# In[6]:


df


# **Rule-based approach**
# 
# This is a practical approach to analyzing text without training or using machine learning models. The result of this approach is a set of rules based on which the text is labeled as positive/negative/neutral. These rules are also known as lexicons. Hence, the Rule-based approach is called Lexicon based approach.
# 
# Widely used lexicon-based approaches are TextBlob, VADER, SentiWordNet.
# 
# **Data preprocessing steps:**
# 
# Cleaning the text
# 
# Tokenization
# 
# Enrichment – POS tagging
# 
# Stopwords removal
# 
# Obtaining the stem words

# # Step 1: Cleaning the text

# In[7]:


import re

# Define a function to clean the text
def clean(text):
# Removes all special characters and numericals leaving the alphabets
    text = re.sub('[^A-Za-z]+', ' ', str(text))
    return text

# Cleaning the text in the review column
df['Cleaned Reviews'] = df['reviews'].apply(clean)
df.head()


# # Step 2: Tokenization
# 
# Tokenization is the process of breaking the text into smaller pieces called Tokens. It can be performed at sentences(sentence tokenization) or word level(word tokenization).
# 
# # Step 3: Enrichment – POS tagging
# 
# Parts of Speech (POS) tagging is a process of converting each token into a tuple having the form (word, tag). POS tagging essential to preserve the context of the word and is essential for Lemmatization.
# 
# # Step 4: Stopwords removal
# Stopwords in English are words that carry very little useful information. We need to remove them as part of text preprocessing. nltk has a list of stopwords of every language. 
# 
# # Step 5: Obtaining the stem words
# A stem is a part of a word responsible for its lexical meaning. The two popular techniques of obtaining the root/stem words are Stemming and Lemmatization.
# 
# The key difference is Stemming often gives some meaningless root words as it simply chops off some characters in the end. Lemmatization gives meaningful root words, however, it requires POS tags of the words.
# 

#  
# ```
# NLTK is a leading platform for building Python programs to work with human language data. 
# 
# ```
# 
# ```
# It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along 
# with a suite of text processing libraries for classification, tokenization, stemming, tagging, 
# parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries
# ```
# 
# 
# 

# In[8]:


import nltk

"""This punkt tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, 
collocations, and words that start sentences. """

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet



# In[9]:


#The nltk.corpus package defines a collection of corpus reader classes, which can be used to access the contents of a diverse set of corpora.


nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# POS tagger dictionary
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    #print(tags)
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
          newlist.append(tuple([word, pos_dict.get(tag[0])]))
          #print(tag[0])
          #print(pos_dict.get(tag[0]))
    return newlist 

df['POS tagged'] = df['Cleaned Reviews'].apply(token_stop_pos)
df.head()


# In[10]:


# Obtaining the stem words – Lemmatization

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
     if not pos:
        lemma = word
        lemma_rew = lemma_rew + " " + lemma
     else:
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

df['Lemma'] = df['POS tagged'].apply(lemmatize)
df.head()


# In[11]:


df[['reviews','Lemma']]


# # Sentiment Analysis using VADER
# 
# VADER stands for Valence Aware Dictionary and Sentiment Reasoner.
# 
# Vader sentiment not only tells if the statement is positive or negative along with the intensity of emotion.
# 
# 

# In[12]:


get_ipython().system('pip install vaderSentiment')


# In[13]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# function to calculate vader sentiment
def vadersentimentanalysis(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']

df['Sentiment'] = df['Lemma'].apply(vadersentimentanalysis)

# function to analyse
def vader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound < 0 :
        return 'Negative'
    else:
        return 'Neutral'
df['Analysis'] = df['Sentiment'].apply(vader_analysis)
df.head()


# In[14]:


vader_counts = df['Analysis'].value_counts()
vader_counts


# # Visual Representation

# In[15]:


import matplotlib.pyplot as plt

# Sample data (replace this with your actual data)
vader_counts = {
    'Positive': 60,
    'Neutral': 30,
    'Negative': 10,
}

# Define colors for the pie chart
colors = ['green', 'gold', 'red']

# Create a figure and subplot
plt.figure(figsize=(10, 6))

# Create the pie chart
plt.subplot(1, 3, 2)
plt.title("Reviews Analysis")
plt.pie(vader_counts.values(), labels=vader_counts.keys(), colors=colors, explode=(0, 0, 0.1), autopct='%1.1f%%', shadow=True, startangle=140)

# Add a legend
plt.legend(vader_counts.keys(), loc='best')

plt.axis('equal')  # Equal aspect ratio ensures that the pie is circular
plt.tight_layout()

# Show the plot
plt.show()


# In[16]:


df.to_csv("BA_reviews.csv")


# # Wordcloud
# 
# Word Cloud or Tag Clouds is a visualization technique for texts that are natively used for visualizing the tags or keywords from the websites

# In[17]:


get_ipython().system('pip install wordcloud')


# In[18]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        max_font_size=60,  # Increase the max font size for more impact
        scale=3,
        random_state=1,
        colormap='viridis'  # Use a different colormap for color variation
    )

    wordcloud = wordcloud.generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')

    plt.imshow(wordcloud, interpolation='bilinear')  # Use a better interpolation method for smoother appearance
    plt.title("Word Cloud", fontsize=24)  # Add a title with a larger font size

    plt.show()

show_wordcloud(df.Lemma)


# In[ ]:





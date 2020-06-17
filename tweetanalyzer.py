#!/usr/bin/env python
# coding: utf-8

# In[ ]:




#__Description__:
#This script will use twitter api
#The necessary data will be returned to the server for parsing
#The script will just accept the Query String
from __future__ import print_function
import tweepy
import numpy as np
import configurations
import pandas as pd
import time
import sys
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
#from langdetect import detect

#We will be using TextBlob for Sentiment Analysis
#TextBlob is also used for text translation
from textblob import TextBlob
import googlemaps


'''
This function will help us escape the rate-limit-error we may recieve
'''
def limit_handled(cursor):

    while True:
        try:
        	#time.sleep(1)
        	yield cursor.next()

        except tweepy.RateLimitError:
            time.sleep(60*15)
            continue

        except StopIteration:
        	break



def make_maps(tweetsDataframe):
	#This function will return the data as required by google maps
	doughnut = []
	sentiment_map = []
	sources_plot = []
	sentiment_pie = []
	retweet_table = []

	########################################
	#for the language plot :: dougnut chart
	doughnut.append(["Language","Tweets"])
	lang_count = tweetsDataframe["language"].value_counts()
	lang_count= lang_count.to_dict()
	for key,value in lang_count.items():
		temp = [key,value]
		doughnut.append(temp)
	########################################


	########################################
	#for the Sentiment plot :: pie chart
	sentiment_pie.append(["Sentiment","Tweets"])
	sentiment_count = tweetsDataframe["sentiments_group"].value_counts()
	sentiment_count= sentiment_count.to_dict()
	for key,value in sentiment_count.items():
		temp = [key,value]
		sentiment_pie.append(temp)
	########################################



	########################################
	#for the sources plot ::
	sources_plot.append(["Twitter Client","Users"])
	source_count = tweetsDataframe["source"].value_counts()[:5][::-1]
	source_count= source_count.to_dict()
	for key,value in source_count.items():
		temp = [key,value]
		sources_plot.append(temp)
	########################################





	########################################
	#for the sentiment_map plot :: geochart
	#sentiment_map.append(['Lat', 'Long', 'Language'])
	for i in range(0,len(tweetsDataframe)):

		temp= []
		latitude = tweetsDataframe['latitude'][i]
		longitude = tweetsDataframe['longitude'][i]
		language = tweetsDataframe["translate"][i]
		#language = language.encode('utf-8')
		sentiment = tweetsDataframe['sentiments'][i]
		if latitude == "":
			continue
		else:

		#if sentiment >=-1 and sentiment <=1:
			temp = [latitude,longitude,language,sentiment,"tooltip"]
			sentiment_map.append(temp)


	########################################



	########################################
	#Most Famous Tweet Table :: Table_Chart
	retweet_table.append(["Tweet Text","ReTweets"])
	df = tweetsDataframe[['translate','retweet_count']].drop_duplicates().sort_values(['retweet_count'],ascending=False)[:10]
	for key,value in zip(df['translate'],df['retweet_count']):
		temp = [key,value]
		retweet_table.append(temp)
	########################################











	return (doughnut,sentiment_map,sources_plot,sentiment_pie,retweet_table)







def QueryTwitter(search_string):

	#Fetching the Configuration Settings
	key = "CRn9TI5ITd3OLA548dzNfzRq2"
	secret = "A129CIOeLb6Z8FxXO1aFWOWOyERfHvnEN8oD5uxg5ED6nXfRBF"
	access_token = "1181121350183706626-5b5mWWAvrh9DLJGEIt93ht7KCjeWdg"
	access_secret = "vMI1tZNrHWjWPDgver7U6Oj9Fo4C5rKOLxdiTR4ddhtxT"

	#Authenticating ::
	#Receiving Access Tokens
	auth = tweepy.OAuthHandler(consumer_key=key,consumer_secret=secret)
	auth.set_access_token(access_token, access_secret)

	#Instantiating the API with our Access Token
	api = tweepy.API(auth)

	tweet_list = []
	for tweet in limit_handled(tweepy.Cursor(api.search,q=search_string).items(50)):
		tweet_list.append(tweet)

	#We now extract details from the tweet and get the resultant DataFrame
	tweet_Data = filter_tweets(tweet_list)
#spcae for additional code
    #Here we can see that at many places we have '@names', which is of no use, since it don't have any meaning, So needs to be removed.
	def remove_pattern(text, pattern_regex):
		r = re.findall(pattern_regex, text)
		for i in r:
			text = re.sub(i, '', text)
		return text 
    # We are keeping cleaned tweets in a new column called 'tidy_tweets'
	tweet_Data['tidy_tweets'] = np.vectorize(remove_pattern)(tweet_Data['translate'], "@[\w]*: | *RT*")
	cleaned_tweets = []

	for index, row in tweet_Data.iterrows():
    # Here we are filtering out all the words that contains link
		words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]
		cleaned_tweets.append(' '.join(words_without_links))
	tweet_Data['tidy_tweets'] = cleaned_tweets

	tweet_Data['absolute_tidy_tweets'] = tweet_Data['tidy_tweets'].str.replace("[^a-zA-Z# ]", "")
      
	stopwords_set = set(stopwords.words('english'))
	cleaned_tweets = []

	for index, row in tweet_Data.iterrows():
    
        # filerting out all the stopwords 
		words_without_stopwords = [word for word in row.absolute_tidy_tweets.split() if not word in stopwords_set and '#' not in word.lower()]
    
        # finally creating tweets list of tuples containing stopwords(list) and sentimentType 
		cleaned_tweets.append(' '.join(words_without_stopwords))
    
	tweet_Data['absolute_tidy_tweets'] = cleaned_tweets
	tokenized_tweet = tweet_Data['absolute_tidy_tweets'].apply(lambda x: x.split())
	for i, tokens in enumerate(tokenized_tweet):
		tokenized_tweet[i] = ' '.join(tokens)

	tweet_Data['absolute_tidy_tweets'] = tokenized_tweet
    
    
	def generate_wordcloud_1(all_words):
		wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5, colormap='Dark2').generate(all_words)

		plt.figure(figsize=(14, 10))
		plt.imshow(wordcloud, interpolation="bilinear")
		plt.axis('off')
		#plt.title("POSITIVE WORD CLOUD")
		#plt.show()
		plt.savefig('C:\\Users\\Admin\\Desktop\\twitter\\static\\poswc.png')

	def generate_wordcloud_2(all_words):
		wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5, colormap='Dark2').generate(all_words)

		plt.figure(figsize=(14, 10))
		plt.imshow(wordcloud, interpolation="bilinear")
		plt.axis('off')
		#plt.title("NEGATIVE WORD CLOUD")
		#plt.show()
		plt.savefig('C:\\Users\\Admin\\Desktop\\twitter\\static\\negwc.png')
        
	all_words = ' '.join([text for text in tweet_Data['absolute_tidy_tweets'][tweet_Data.sentiments_group == 'positive']])
	generate_wordcloud_1(all_words)
	all_words = ' '.join([text for text in tweet_Data['absolute_tidy_tweets'][tweet_Data.sentiments_group == 'negative']])
	generate_wordcloud_2(all_words)
    
    # function to collect hashtags
	def hashtag_extract(text_list):
		hashtags = []
		# Loop over the words in the tweet
		for text in text_list:
			ht = re.findall(r"#(\w+)", text)
			hashtags.append(ht)

		return hashtags

	def generate_hashtag_freqdist(hashtags):
		a = nltk.FreqDist(hashtags)
		d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
		# selecting top 15 most frequent hashtags     
		d = d.nlargest(columns="Count", n = 25)
		plt.figure(figsize=(16,7))
		ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
		plt.xticks(rotation=80)
		ax.set(ylabel = 'Count')
    #plt.show()
		plt.savefig('C:\\Users\\Admin\\Desktop\\twitter\\static\\hashtag.png')
        
	hashtags = hashtag_extract(tweet_Data['tidy_tweets'])
	hashtags = sum(hashtags, [])
	generate_hashtag_freqdist(hashtags)
    
	from nrclex import NRCLex
	words = ' '.join([text for text in tweet_Data['absolute_tidy_tweets']])
	text_object = NRCLex(words)
	nrc=(text_object.affect_frequencies)
	e = pd.DataFrame({'Emotion': list(nrc.keys()),
                      'Frequency': list(nrc.values())})
	plt.figure(figsize=(16,7))
	ax = sns.barplot(data=e, x= "Emotion", y = "Frequency")
	plt.xticks(rotation=80)
	ax.set(ylabel = 'Frequency')
    #plt.show()
	plt.savefig('C:\\Users\\Admin\\Desktop\\twitter\\static\\nrc_lexicon.png')    


	(doughnut,sentiment_map,sources_plot,sentiment_pie,retweet_table) = make_maps(tweet_Data)
	#return tweet_Data
	return (doughnut,sentiment_map,sources_plot,sentiment_pie,retweet_table)

    
    









# Will be creating the dataframes in this function
# Snetiment Analysis
def filter_tweets(tweets):


	id_list = [tweet.id for tweet in tweets]
	#Will contain a single column table containing all the tweet ids
	tweet_Data = pd.DataFrame(id_list,columns=['id'])
	tweet_Data["text"] = [tweet.text for tweet in tweets]
	tweet_Data["retweet_count"]= [tweet.retweet_count for tweet in tweets]
	#tweet_Data["favourite_count"] = [tweet.favourite_count for tweet in tweets]
	# Location
	#tweet_Data["location"] = [tweet.author.location for tweet in tweets]



	Sentiments_list = []
	Sentiments_group = []

	Subjectivity_list = []
	Subjectivity_group = []

	tweet_text_list = []
	tweet_location_list = []

	tweet_language = []
	tweet_latitude = []
	tweet_longitude =[]
	tweet_country = []
	tweet_source = []
	tweet_translation= []



	for tweet in tweets:
		raw_tweet_text = tweet.text
		message = TextBlob(tweet.text)
		location = tweet.author.location
		source = tweet.source
		#source = source.encode('utf-8')
		tweet_source.append(source)
		# location can be null :: We have to handle that too
		if len(location) !=0:
			(latitude,longitude,country) = geocode_location(location)
			tweet_latitude.append(latitude)
			tweet_longitude.append(longitude)
			tweet_country.append(country)

		else:
			tweet_latitude.append("")
			tweet_longitude.append("")
			tweet_country.append("")




		#Detecting and Changing the language to english for sentiment analysis
		lang = message.detect_language()
		tweet_language.append(str(lang))
		try:
			if str(lang) != "en":
				message = message.translate(to="en") #Problem Here
		except:
			pass

		#### Special Character removal #####
		message = str(message)
		new_message = ""
		for letter in range(0,len(message)):
			current_read =message[letter]
			if ord(current_read) > 126:
				#this is a special character & hence will be skipped
				continue
			else:
				new_message =new_message+current_read

		message = new_message ### Change here on :: Added the Translated Text to Database
		tweet_translation.append(message[:120])
		message = TextBlob(message)
		######################################

		#Changing the Language is important
		#Since it will help in sentiment analysis using TextBlob
		#When language is english remove special characters :: heavily affects analysis
		sentiment = message.sentiment.polarity
		if (sentiment > 0):
			#postive
			Sentiments_group.append('positive')
		elif (sentiment < 0):
			#Negative
			Sentiments_group.append('negative')
		else:
			Sentiments_group.append('neutral')



		subjectivity = message.sentiment.subjectivity
		if (subjectivity > 0.4):
			#subjective ::: Long tweet
			Subjectivity_group.append('subjective')
		else:
			Subjectivity_group.append('objective')

		Sentiments_list.append(sentiment)
		Subjectivity_list.append(subjectivity)
		tweet_text_list.append(raw_tweet_text)
		tweet_location_list.append(location)


	tweet_Data["sentiments"] = Sentiments_list
	tweet_Data["sentiments_group"] = Sentiments_group

	tweet_Data["subjectivity"]= Subjectivity_list
	tweet_Data["subjectivity_group"] = Subjectivity_group

	tweet_Data["location"] = tweet_location_list
	tweet_Data["text"] = tweet_text_list

	tweet_Data["language"] = tweet_language
	tweet_Data["latitude"] = tweet_latitude
	tweet_Data["longitude"]= tweet_longitude
	tweet_Data["country"] = tweet_country
	tweet_Data["source"] = tweet_source
	tweet_Data["translate"] = tweet_translation



	#Let us calculate the sentiment scores

	return tweet_Data

def geocode_location(loc):
	#Importing the API key for Google Geocode
	gmaps_api = 'AIzaSyDpXgJFT98eHszi3fqHJK5oi9w59GE0hGs'

	#Registering our app by sending the API key
	gm = googlemaps.Client(key=gmaps_api)

	##################################################################
	#We need to geocode this location and store it as lat and longtitude
	location_result = gm.geocode(loc)
	if len(location_result) > 0:
		#means that atleast something was returned
		latitude = location_result[0]['geometry']['location']['lat']
		longitude= location_result[0]['geometry']['location']['lng']
		country =location_result[0]['formatted_address'].split(",")
		country = country[len(country)-1]		# there arises a problem here
		return (latitude,longitude,country)


	else:
		#store null
		return ("","","")

	return
	##################################################################


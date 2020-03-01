# CSE 40437/60437 - Social Sensing and Cyber-Physical Systems - Spring 2020
# Project
# Chris Foley, Catherine Markley, Ale Lopez

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# The filename of the dataset to be parsed
TWEET_DATASET = "./smalltweets.json"

# This dictionary will contain only the data that is needed for this project
TWEETS = dict()

# This is a list of keywords that will be used to filter out tweets that do not contain any of them
KEYWORDS = ["earthquake", "quake", "shaking", "shake"]

# Imports tweets from files.
def import_tweets(filename):

    # Import data from file
    with open(filename) as file:
        data = file.read()

    # Loads each tweet within file, assuming each tweet is separated by a newline
    # ignores final line of file - blank line
    rawTweets = [json.loads(tweet) for tweet in data.split('\n')[:-1]]

    for tweet in rawTweets:
        if shouldInclude(tweet):
            # print(tweet["text"])
            # print("\n\n")

            TWEETS[tweet["id"]] = dict()
            TWEETS[tweet["id"]]["coordinates"] = tweet["coordinates"]
            TWEETS[tweet["id"]]["created_at"] = tweet["created_at"]
            TWEETS[tweet["id"]]["text"] = tweet["text"]

    # for tweet in TWEETS.items():
    #     print(tweet)
    print(len(TWEETS))

# Returns the geolocation of the tweet
def getLocation(tweet):
    pass

# Takes in a tweet and returns in that tweet should be considered in the dataset
# Returns false for tweets with no geolocation, tweets not in English, and for retweets
# TODO: ensure func description remains accurate
def shouldInclude(tweet):
    # Remove tweet if it's a retweet
    if "RT" in tweet["text"][0:4]:
        return False
    # Remove tweet if it doesn't have a geolocation
    if tweet["coordinates"] is None or tweet["coordinates"]["coordinates"] == [0, 0]:
        return False

    # Keep only tweets in English
    if tweet["lang"] != "en":
        return False

    # TODO: Ensure content is about earthquake
    return any(keyword in tweet["text"] for keyword in KEYWORDS)

# Plot the tweet locations onto a visual map (using a dataframe)
def plotLocation(tweetDF):
    # downloadingthe shape file
    #street_map = gpd.read_file('map.shp')

    # the lat/long coordinate reference system
    crs = {'init': 'epsg:4326'}

    # creating points from lat/long
    geometry = [Point(xy) for xy in zip(tweetDF["Longitude"], tweetDF["Latitude"])]

    # creating a new dataframe for the plot
    geo_df1 = gpd.GeoDataFrame(tweetDF[:250], crs = crs, geometry = geometry[:250])
    geo_df2 = gpd.GeoDataFrame(tweetDF[250:500], crs = crs, geometry = geometry[250:500])
    geo_df3 = gpd.GeoDataFrame(tweetDF[500:], crs = crs, geometry = geometry[500:])

    # plot the map and points
    fig, ax = plt.subplots(figsize = (15, 15))
    #street_map.plot(ax = ax, alpha = 0.4, color = 'grey')
    geo_df1.plot(ax = ax, markersize = 20, color = 'blue', marker = 'o')
    geo_df2.plot(ax = ax, markersize = 20, color = 'green', marker = 'o')
    geo_df3.plot(ax = ax, markersize = 20, color = 'pink', marker = 'o')

    # plot true epicenter
    #center = pd.DataFrame([(28.230, 84.731)])
    geometry4 = [Point(28.230, 84.731)]
    geo_df4 = gpd.GeoDataFrame(geometry4, crs = crs, geometry = geometry4)
    geo_df4.plot(ax = ax, markersize = 20, color = 'blue', marker = 'x')

    plt.show()
    pass

# Creates DataFrame that contains all tweets' coordinates
def createLocationDataframe():
    all_coordinates = list()
    for id, tweet in TWEETS.items():
        tweet_coordinates = tweet["coordinates"]["coordinates"]
        all_coordinates.append(tweet_coordinates)

    df = pd.DataFrame(np.array(all_coordinates), columns=["Latitude", "Longitude"])
    plotLocation(df)

# Places tweets onto a map of the region, creating different frames based upon
# the time stamps. Saves locally
def exportRippleMap():
    # Separate tweets based on time stamp
    # times = [tweet["created_at"] for tweet in TWEETS.items()]
    # times = set(times)
    pass

# Main execution
if __name__ == '__main__':
    import_tweets(TWEET_DATASET)
    exportRippleMap()
    createLocationDataframe()
    print("yay ive made it to the end")




  #   /Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/pyproj/crs/crs.py:55:
  # FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method.
  # When making the change, be mindful of axis order changes: https://pyproj4.github.io/pyproj/stable/gotchas.html#axis-order-changes-in-proj-6
  # return _prepare_from_string(" ".join(pjargs))

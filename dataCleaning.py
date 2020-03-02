# CSE 40437/60437 - Social Sensing and Cyber-Physical Systems - Spring 2020
# Project
# Chris Foley, Catherine Markley, Ale Lopez

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd
from shapely.geometry import Point
import datetime
import pytz
import gmplot
from pandas.api.types import is_numeric_dtype
# from mpl_toolkits.basemap import Basemap


# The filename of the dataset to be parsed
TWEET_DATASET = "./smalltweets.json"

# This dictionary will contain only the data that is needed for this project
TWEETS = dict()

# This is a list of keywords that will be used to filter out tweets that do not contain any of them
KEYWORDS = ["earthquake", "quake", "shaking", "shake"]

# Imports tweets from files.
def import_tweets(filename):
    global TWEETS

    # Import data from file
    with open(filename) as file:
        data = file.read()

    # Loads each tweet within file, assuming each tweet is separated by a newline
    # ignores final line of file - blank line
    rawTweets = [json.loads(tweet) for tweet in data.split('\n')[:-1]]

    times = []

    # Create a datetime to hold the minimum datetime from the included tweets
    # Initialize to the maximum datetime available to ensure it is overwritten
    minDatetime = datetime.datetime(datetime.MAXYEAR, 12, 31, 23, 59, 59, 999999, tzinfo=pytz.UTC)

    # Create a datetime to hold the maximum datetime from the included tweets
    # Initialize to the minimum datetime available to ensure it is overwritten
    maxDatetime = datetime.datetime(datetime.MINYEAR, 1, 1, 1, 1, 1, 1, tzinfo=pytz.UTC)

    for tweet in rawTweets:
        if shouldInclude(tweet):
            # Convert time stamp string into a datetime object
            timestamp = datetime.datetime.strptime(tweet["created_at"], '%a %b %d %H:%M:%S %z %Y')

            # Extract useful information
            TWEETS[tweet["id"]] = dict()
            TWEETS[tweet["id"]]["coordinates"] = tweet["coordinates"]["coordinates"]
            TWEETS[tweet["id"]]["created_at"] = timestamp
            TWEETS[tweet["id"]]["text"] = tweet["text"]

            # print(tweet["coordinates"]["coordinates"], "\n\t", tweet["text"], "\n", timestamp, "\n\n" )

            # Check if new minimum
            if timestamp < minDatetime:
                minDatetime = timestamp

            if timestamp > maxDatetime:
                maxDatetime = timestamp

            times.append(timestamp)

    # print("MINIMUM DATE:")
    # print(minDatetime)

    print(len(TWEETS))
    removeOutliers()

    # first step is the min date time rounded down to 10 minute mark (floor style)
    step = minDatetime  - datetime.timedelta(minutes=minDatetime.minute % 10,
                             seconds=minDatetime.second,
                             microseconds=minDatetime.microsecond)

    # create dictionary of time categories at 5 minute intervals
    timecats = {}
    while step < (minDatetime + datetime.timedelta(hours=1, minutes=5)):
        timecats[step] = []
        step += datetime.timedelta(minutes=5)

    # place tweets in appropriate category
    for id, tweet in TWEETS.items():
        tweettime = tweet["created_at"]

        # round down to find the category
        cat = tweettime - datetime.timedelta(minutes=tweettime.minute % 5,
                                 seconds=tweettime.second,
                                 microseconds=tweettime.microsecond)

        if cat in timecats.keys():
            timecats[cat].append(id)


    # for cat, ts in timecats.items():
    #     print(cat)
    #     for t in ts:
    #         print("\t", t)

    # for tweet in TWEETS.items():
    #     print(tweet)
    print(len(TWEETS))
    epicenter_prediction = find_epicenter(timecats)
    df = createLocationDataframe()
    plotLocation(df, epicenter_prediction)

def removeOutliers():
    # Using Pandas
    global TWEETS
    df = createLocationDataframe()
    df = df[(np.abs(stats.zscore(df)) < 1.5).all(axis=1)]

    TWEETS = {key: val for key, val in TWEETS.items() if df.isin([key]).any().any()}

def find_epicenter(timecats):
    total_sum_lat = 0
    total_sum_long = 0
    total_tweets = 0
    times = len(timecats.keys()) + 1
    weight = 10*(0.5**times)

    for timecat, tweetids in timecats.items():
        total_sum_lat  = total_sum_lat  + (weight * sum([TWEETS[id]["coordinates"][1] for id in tweetids]))
        total_sum_long = total_sum_long + (weight * sum([TWEETS[id]["coordinates"][0] for id in tweetids]))
        total_tweets += weight * len(tweetids)
        times -=1
        weight = 10*(0.5**times)

    lat_prediction = total_sum_lat/total_tweets
    long_prediction = total_sum_long/total_tweets
    print("Prediction: ", long_prediction, ", ", lat_prediction)
    print("Actual: 84.731, 28.230")
    return Point(long_prediction, lat_prediction)


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
def plotLocation(tweetDF, epicenter_prediction):
    # downloading the shape file
    street_map = gpd.read_file('world_map/ne_50m_admin_0_countries.shp')

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
    street_map.plot(ax = ax)
    #street_map.plot(ax = ax, alpha = 0.4, color = 'grey')
    geo_df1.plot(ax = ax, markersize = 20, color = 'blue', marker = 'o')
    geo_df2.plot(ax = ax, markersize = 20, color = 'green', marker = 'o')
    geo_df3.plot(ax = ax, markersize = 20, color = 'pink', marker = 'o')

    # plot true epicenter
    #center = pd.DataFrame([(28.230, 84.731)])
    geometry4 = [Point(84.731, 28.230)]
    geo_df4 = gpd.GeoDataFrame(geometry4, crs = crs, geometry = geometry4)
    geo_df4.plot(ax = ax, markersize = 20, color = 'red', marker = 'x')

    # plot epicenter prediction
    geometry5 = [epicenter_prediction]
    geo_df5 = gpd.GeoDataFrame(geometry5, crs = crs, geometry = geometry5)
    geo_df5.plot(ax = ax, markersize = 20, color = 'black', marker = '*')

    plt.show()
    pass

def test_googlemap(tweetDF):
    long_list = tweetDF["Longitude"]
    lat_list = tweetDF["Latitude"]

    gmap = gmplot.GoogleMapPlotter(28.1348, 84.4352, 10)

    gmap.scatter(lat_list, long_list, '#FF5555', size = 40, marker = True)

    gmap.plot(lat_list, long_list, 'cornflowerblue', edge_width = 3.0)

    gmap.apikey = 'AIzaSyBWohY_btQ0Gat3hMF5p-KTTUGKOZ4xQvU'

    gmap.draw("./test.html")

    pass

def test_basemap(tweetDF):
    fig, ax = plt.subplots(figsize = (15, 15))

    long_list = tweetDF["Longitude"]
    lat_list = tweetDF["Latitude"]

    m = Basemap(projection='merc', resolution='h',
            lat_0=28.1348, lon_0=84.4352,
            width=1E6, height=1.2E6)
    m.drawcountries(color='gray')

    m.scatter(long_list, lat_list, marker='o', color='r')

    plt.show()

    pass

# Creates DataFrame that contains all tweets' coordinates
def createLocationDataframe():
    all_info = list()
    for id, tweet in TWEETS.items():
        tweet_coordinates = tweet["coordinates"]
        tweet_info = [tweet_coordinates[0], tweet_coordinates[1], id]
        all_info.append(tweet_info)

    df = pd.DataFrame(np.array(all_info), columns=["Longitude", "Latitude", "id"])
    return df

    #test_googlemap(df)
    #test_basemap(df)

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
    # exportRippleMap()

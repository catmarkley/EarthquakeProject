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
from pandas.api.types import is_numeric_dtype
import copy
import random
from random import random
import sys

# The filename of the dataset to be parsed
TWEET_DATASET = "./smalltweets.json"

# This dictionary will contain only the data that is needed for this project
TWEETS = dict()

# This is a list of keywords that will be used to filter out tweets that do not contain any of them
KEYWORDS = ["earthquake", "quake", "shaking", "shake"]


# Imports tweets from file into a dictionary
# returns the first/last tweets times and an array of all times
def import_tweets(filename):
    global TWEETS

    # Import data from file
    with open(filename) as file:
        data = file.read()

    # Loads each tweet within file, assuming each tweet is separated by a newline
    # ignores final line of file - blank line
    rawTweets = [json.loads(tweet) for tweet in data.split('\n')[:-1]]

    for tweet in rawTweets:
        if shouldInclude(tweet):
            # Convert time stamp string into a datetime object
            timestamp = datetime.datetime.strptime(tweet["created_at"], '%a %b %d %H:%M:%S %z %Y')

            # Extract useful information
            TWEETS[tweet["id"]] = dict()
            TWEETS[tweet["id"]]["coordinates"] = tweet["coordinates"]["coordinates"]
            TWEETS[tweet["id"]]["created_at"] = timestamp
            TWEETS[tweet["id"]]["text"] = tweet["text"]

    #print("Before outliers removed size: ", len(TWEETS))
    removeOutliers()
    #print("After outliers removed size: ", len(TWEETS))

    times = []

    # Create a datetime to hold the minimum datetime from the included tweets
    # Initialize to the maximum datetime available to ensure it is overwritten
    minDatetime = datetime.datetime(datetime.MAXYEAR, 12, 31, 23, 59, 59, 999999, tzinfo=pytz.UTC)

    # Create a datetime to hold the maximum datetime from the included tweets
    # Initialize to the minimum datetime available to ensure it is overwritten
    maxDatetime = datetime.datetime(datetime.MINYEAR, 1, 1, 1, 1, 1, 1, tzinfo=pytz.UTC)

    for id, tweet in TWEETS.items():
        timestamp = tweet["created_at"]
        # print(tweet["coordinates"]["coordinates"], "\n\t", tweet["text"], "\n", timestamp, "\n\n" )

        # Check if new minimum
        if timestamp < minDatetime:
            minDatetime = timestamp

        if timestamp > maxDatetime:
            maxDatetime = timestamp

        times.append(timestamp)
    
    return minDatetime, maxDatetime, times


# Function to create time categories (of when tweets come in)
# and place tweets in these time categories
def timeDivideTweets(minDatetime):
    # first step is the min date time rounded down to 10 minute mark (floor style)
    step = minDatetime  - datetime.timedelta(minutes=minDatetime.minute % 10,
                             seconds=minDatetime.second,
                             microseconds=minDatetime.microsecond)

    # create dictionary of time categories at 5 minute intervals
    timecats = {}
    while step < (minDatetime + datetime.timedelta(hours=2, minutes=5)):
        timecats[step] = []
        step += datetime.timedelta(minutes=5)

    # place tweets in appropriate category
    #print("TWEETS length: ", len(TWEETS))
    num = 0
    for id, tweet in TWEETS.items():
        tweettime = tweet["created_at"]

        # round down to find the category
        cat = tweettime - datetime.timedelta(minutes=tweettime.minute % 5,
                                 seconds=tweettime.second,
                                 microseconds=tweettime.microsecond)

        if cat in timecats.keys():
            num += 1
            timecats[cat].append(id)

    #print("LEN OF TIMECATS VALS: ", num)
    return timecats


# Function to remove outlier tweets
def removeOutliers():
    # Using Pandas
    global TWEETS
    df = createLocationDataframe()
    df = df[(np.abs(stats.zscore(df)) < 1.5).all(axis=1)]

    TWEETS = {key: val for key, val in TWEETS.items() if df.isin([key]).any().any()}


# Takes in all tweets separated by time categories
# Predict the epicenter based on an averaging algorithm and returns this point
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
def plotLocation(tweetDF, epicenter_pred_avg, epicenter_pred_particle):
    # downloading the shape file
    street_map = gpd.read_file('world_map/ne_50m_admin_0_countries.shp')

    # the lat/long coordinate reference system
    crs = {'init': 'epsg:4326'}

    # creating points from lat/long
    geometry = [Point(xy) for xy in zip(tweetDF["Longitude"], tweetDF["Latitude"])]

    # creating a new dataframe for the plot
    geo_df1 = gpd.GeoDataFrame(tweetDF[:60], crs = crs, geometry = geometry[:60])
    geo_df2 = gpd.GeoDataFrame(tweetDF[60:160], crs = crs, geometry = geometry[60:160])
    geo_df3 = gpd.GeoDataFrame(tweetDF[160:500], crs = crs, geometry = geometry[160:500])
    geo_df4 = gpd.GeoDataFrame(tweetDF[500:], crs = crs, geometry = geometry[500:])

    # plot the map and points
    fig, ax = plt.subplots(figsize = (15, 15))
    street_map.plot(ax = ax)
    #street_map.plot(ax = ax, alpha = 0.4, color = 'grey')
    geo_df1.plot(ax = ax, markersize = 20, color = 'blue', marker = 'o')
    geo_df2.plot(ax = ax, markersize = 20, color = 'green', marker = 'o')
    geo_df3.plot(ax = ax, markersize = 20, color = 'pink', marker = 'o')
    geo_df4.plot(ax = ax, markersize = 20, color = 'orange', marker = 'o')

    # plot true epicenter
    #center = pd.DataFrame([(28.230, 84.731)])
    geometry5 = [Point(84.731, 28.230)]
    geo_df5 = gpd.GeoDataFrame(geometry5, crs = crs, geometry = geometry5)
    geo_df5.plot(ax = ax, markersize = 20, color = 'red', marker = 'x')

    # plot epicenter prediction from average
    geometry6 = [epicenter_pred_avg]
    geo_df6 = gpd.GeoDataFrame(geometry6, crs = crs, geometry = geometry6)
    geo_df6.plot(ax = ax, markersize = 20, color = 'black', marker = '*')

    # plot epicenter prediction from particle filtering
    geometry7 = [epicenter_pred_particle]
    geo_df7 = gpd.GeoDataFrame(geometry7, crs = crs, geometry = geometry7)
    geo_df7.plot(ax = ax, markersize = 20, color = 'brown', marker = '*')

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


def generateParticleSet(spread, start_long, start_lat):
    long_min = start_long-10
    long_max = start_long+10
    lat_min = start_lat-10
    lat_max = start_lat+10
    #print("Starting max/min long, max/min lat: ", long_max,  ' ', long_min,  ' ', lat_max,  ' ', lat_min)
    orig_S = {}
    S = []
    lat_step = (lat_max - lat_min)/spread
    long_step = (long_max - long_min)/spread
    lats = []
    longs = []
    for i in range(0, spread):
        lats.append(i*lat_step + lat_min)
        longs.append(i*long_step + long_min)

    avg_long = sum(longs)/len(longs)
    avg_lats = sum(lats)/len(lats)
    total = 0
    part_id = 0
    for i in range(0, spread):
        for j in range(0, spread):
            # Set the weight of all particles to 1
            # (longitude, latitude, weight, particle ID, velocity_x, velocity_y)
            particle = [longs[j], lats[i], 1, part_id, 0, 0]
            S.append(particle)
            orig_S[part_id] = [longs[j], lats[i]]
            total += (longs[j]-avg_long)**2 + (lats[i]-avg_lats)**2
            part_id += 1

    # calculate the standard deviation of this set:
    std_dev = (total/len(S))**0.5

    return orig_S, S, spread**2, std_dev


def averageTweets(timecat):
    avg_lat = 0
    avg_long = 0
    for tweetID in timecat:
        avg_long += TWEETS[tweetID]['coordinates'][0]
        avg_lat += TWEETS[tweetID]['coordinates'][1]

    avg_long = avg_long/len(timecat)
    avg_lat = avg_lat/len(timecat)

    return avg_long, avg_lat


def reweightParticles(S, m_x, m_y, variance):
    for i in range(0, len(S)):
        x = S[i][0]
        y = S[i][1]
        d_x = m_x - x
        d_y = m_y - y
        exp = -(d_x**2 + d_y**2)/(2*variance)
        sqrt = (2*3.14159*variance)**(1/2)
        if(exp < -100):
            w = 100000
        elif(-exp < 0.000001):
            w = 0.0000000001
        else:
            w = (1/sqrt)**exp
        
        # weight is the inverse because we want weight to be smaller as d_x and d_y get bigger
        S[i][2] = 1/w
        #print("WEIGHT: ", w, "\tDIST: ", d_x, ', ', d_y, '\tExp: ', exp, ' Sqrt: ', sqrt, ' 1/sqrt: ', 1/sqrt, ' Calc: ', temp2**exp)


def resample(particleSet, N):
    weights = []
    new_set = []
    for particle in particleSet:
        weights.append(particle[2])

    for _ in range(N):
        new_set.append(weighted_choice(particleSet, weights))

    return new_set


def weighted_choice(objects, weights):
    """ returns randomly an element from the sequence of 'objects', 
        the likelihood of the objects is weighted according 
        to the sequence of 'weights', i.e. percentages."""

    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]


def repositionParticles(particleSet, vel_max, N):
    velocities = list(np.random.normal(0, vel_max, 2*N))
    for i in range(0, len(particleSet)):
        vel_x = velocities[i]
        vel_y = velocities[i+N]
        # update the particle's velocity
        particleSet[i][4] = vel_x
        particleSet[i][5] = vel_y
        # update the particle's position based on its velocity
        particleSet[i][0] = particleSet[i][0] + vel_x
        particleSet[i][1] = particleSet[i][1] + vel_y


def avgOrigPartPositions(particleSet, originalParticleSet, N):
    long_total = 0
    lat_total = 0
    for particle in particleSet:
        ID = particle[3]
        long_total += originalParticleSet[ID][0]
        lat_total += originalParticleSet[ID][1]
        #print("ID, Original position: ", ID, ', ', originalParticleSet[ID][0], ', ', originalParticleSet[ID][1])

    return long_total/N, lat_total/N


def findSeedLoc(timecats, x):
    tweetIDFirst = list(timecats.values())[1][0]
    return TWEETS[tweetIDFirst]['coordinates'][0], TWEETS[tweetIDFirst]['coordinates'][1]
    '''collected = 0
    avg_long = 0
    avg_lat = 0
    while(collected < x):
        time = 1
        timecat = list(timecats.values())[time]
        for tweetID in timecat:
            collected += 1
            avg_long += TWEETS[tweetID]['coordinates'][0]
            avg_lat += TWEETS[tweetID]['coordinates'][1]
        time += 1

    return avg_long/collected, avg_lat/collected'''


def particleFiltering(timecats):

    # STEP 1: Generate a particle set S_0 by allocating N particles evenly on a map area
    # Set map area coordinates, set N
    # The particles are allocated in a square
    
    # Find the average loc of the first x tweets and generate the particle set based on that area
    start_long, start_lat = findSeedLoc(timecats, 5)
    #print("Starting long and lat: ", start_long, start_lat)

    # Generate the particle set and create a copy of this original set
    # Create a copy of this initial set to remember the initial positions of all particles
    originalParticleSet, particleSet, N, std_dev = generateParticleSet(60, start_long, start_lat)
    #print("N: ", N)
    #print("Std deviation: ", std_dev)

    # Start my for loop
    time = 0
    while(time < len(timecats)):

        # Make sure the current time category is not empty
        found = False
        while(not found):
            if(len(list(timecats.values())[time]) == 0):
                time += 1
            else:
                found = True

        # STEP 2: Average the location from all tweets in the current timecat
        av_long, av_lat = averageTweets(list(timecats.values())[time])
        #print("Av long and lat from timecat ", time, ": ", av_long, av_lat)

        # STEP 3: Reweight the particles based on their distance from the averaged timecat's location
        # The closer to this point, the higher the weight of the particle
        reweightParticles(particleSet, av_long, av_lat, std_dev**2)

        # STEP 4: Create a new particle set by resampling N particles from S_0 (can pick particles more than once)
        # Sampling based on weight, so higher weight means more likely to sample
        particleSet = resample(particleSet, N)

        pred_long, pred_lat = avgOrigPartPositions(particleSet, originalParticleSet, N)
        #print("Predicted long, lat after reweighting and resampling: ", pred_long, pred_lat)

        # STEP 5: Predict particle movement to the next timecat by using Newton's laws of motion
            # Each particle is randomly moved a little differently
            # Create a normal distribution of velocities to use, and randomly choose one for each particle
            # This particle movement models the earthquake information spread
            # The original earthquake location will be determined by the original location of particles

        repositionParticles(particleSet, 0.5, N)

        # STEP 6: Repeat steps 2-6 until we have gone through all of the time categories
        time += 1

    # STEP 7: We will end with a set of particles - retrieve their initial location and average it: this is the predicted epicenter
    pred_long, pred_lat = avgOrigPartPositions(particleSet, originalParticleSet, N)
    #print("Final predicted long, lat: ", pred_long, pred_lat)
    return Point(pred_long, pred_lat)


# Main execution
if __name__ == '__main__':
    
    if(len(sys.argv) < 2):
        print("Usage: earthquakePredictor.py <iterations>")
        exit(1)
    else:
        iterations = int(sys.argv[1])

    # Outliers removed in this function
    minDatetime, maxDatetime, times = import_tweets(TWEET_DATASET)
    
    print("Number of tweets after outlier detection: ", len(TWEETS))

    # Create time categories
    timecats = timeDivideTweets(minDatetime)

    # Predict the epicenter by particle filtering
    # Since there is randomness involved, we will do this multiple times
    avg_x = 0
    avg_y = 0
    for i in range(0, iterations):
        result = particleFiltering(timecats)
        avg_x += result.x
        avg_y += result.y
        print("Particle filtering iteration ", i+1, " complete")

    epicenter_pred_particle = Point(avg_x/iterations, avg_y/iterations)

    print("\nParticle filtering prediction: ", epicenter_pred_particle)

    # Predict the epicenter by an averaging method
    epicenter_pred_avg = find_epicenter(timecats)

    print("Averaging method prediction: ", epicenter_pred_avg)

    print("Actual: 84.731, 28.230\n\n\n")

    # Plot points
    df = createLocationDataframe()
    plotLocation(df, epicenter_pred_avg, epicenter_pred_particle)

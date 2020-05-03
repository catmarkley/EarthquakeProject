# Earthquake Epicenter Predictor


## Created by

- Catherine Markley (cmarkley)
- Chris Foley (cfoley)
- Ale Lopez (alopez10)

## Usage

Run this statement in the terminal: 

```bash
python3.6 earthquakePredictor.py <iterations>
```

The iterations argument represents the number of iterations of particle filtering (since particle filtering uses a bit of randomness), but since each iteration takes a while a lower number such as 2 can be used for testing. 10 is the number we used for our final paper since a larger number of iterations will produce more accurate results, but it may take some time to run.

Once run, after performing both our averaging and particle filtering methods, a map will pop up showing the tweets and prediction data points. This is an interactive map that can be zoomed and moved around.


## Other Files

"smalltweets.json" - our data file of tweets; contains the first 50,000 tweets (in order of time) from the extremely large original data file containing 2 million tweets. This could not be uploaded to github but it is a dependency for this program to run.

"world_map" - a folder containing the necessary files to generate our map image


## Link to github
https://github.com/catmarkley/EarthquakeProject
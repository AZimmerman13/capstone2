# capstone2
## Motivation

Energy production is the cause of massive quantities of greenhouse gasses flowing into our atmosphere.  For many years, solar panels and wind turbines have been hailed as the harbinger of a renewable energy revolution that has only just very recently begun to take shape on a global scale.  The chief complaint from those hesitant to adopt or incorporate renewable energy sources was almost always their effect on energy prices.  At the time of writing, the world is engulfed in a novel Coronavirus pandemic that has shut down industry and social interaction accross the board.  This pandemic, and the countless videos of wildlife returning to abandoned street and uncluttered harbors, has sparked many conversations about what type of world we want to return to once it is safe to do so.

If you ask yourself this question and the answer is either "a renewable world" **or** you just think data science is really cool, continue reading.  One of the best tools an advocate for the incorporation of renewables can have is an understanding of how they affect energy prices.  In this project I endeavor to build a model that can predict energy prices based on generation and weather data and, hopefully, provide some insight as to how renewables affect those prices.

## Data
This publicly available dataset came in two seperate .csv files (weather and energy) posted on [Kaggle](https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather#weather_features.csv) in mid 2019.  Some previous work had been done to understand the effect of time on energy prices, but I was more interested in determining the effect of different energy generation mixtures and weather.  As such, the following analysis does not consider the effects of the time-series on price.

The weather dataset contained hourly data for the 5 largest cities in Spain: Madrid, Barcelona, Valencia, Sevilla, Bilbao.

![image of those 5 cities in spain](images/map-of-spain.jpg)

credit: https://www.alicante-spain.com/images/map-of-spain.jpg

This dataset was relatively clean save for the ' ' in front of 'Barcelona' in every row, as well as what appeared to be a broken pressure gauge for about a month in Barcelona.

The energy dataset contained similarly timestamped data, mostly concerning generation in MW for various energy sources throughout the country.  This dataset was incomplete in a few areas, namely that the 'generation fossil coal-derived gas', 'generation fossil oil shale', 'generation fossil peat', 'generation geothermal', and 'generation marine' contained only zeros, and 'generation hydro pumped storage aggregated' contained all null values. 

Both datasets covered a three-year period from January 2015 to December 2018.


## Pipeline and Workflow

I gave myself the challenge of working with the AWS suite on this project, taking the opportunity to gain familiarity with these widely used tools.  I stored my data and wrote results remotely into an S3 bucket, and did all model training and manipulation on the full dataset in and ubuntu t2.small EC2 instance with anaconda3 and git.  I wrote code mostly on my local machine, making small adjustments in vim on the EC2 instance when necessary.  I followed a basic git workflow, essentially treating my local and virtual machines as if they were partners working on the same project.

I created a Pipeline class to load data in from S3 (using the s3fs library) and apply the necessary cleaning and transformations.  I also worked a bit with SKlearn's built-in Pipeline class.  The biggest speed-bump at this stage was turning the 'city_name' feature into a series of features that represented weather data for each city.  While this solved the probem of having duplicate indices (one for each city at each timestamp), it sent my dimensionality skyward very quickly.


## EDA

EDA turned out to be very useful for narrowing down my long list of features into something a bit less computationally expensive and more interpretable.

I had a suspicion that weather_description and weather_id were redundant as they contained a similar number of categories.  Some simple exploration confirmed this and found additional collinearity/sub categorization in the weather_main column, clear upon moderately close inspection:


| Weather Description          | Weather ID | Weather 'main' |
|------------------------------|------------|----------------|
| sky is clear                 | 800        | clear          |
| few clouds                   | 801        | clouds         |
| scattered clouds             | 802        | clouds         |
| broken clouds                | 803        | clouds         |
| overcast clouds              | 804        | clouds         |
| light rain                   | 500        | rain           |
| moderate rain                | 501        | rain           |
| heavy intensity rain         | 502        | rain           |
| light intensity shower rain  | 520        | rain           |
| heavy intensity shower rain  | 522        | rain           |
| shower rain                  | 521        | rain           |
| very heavy rain              | 503        | rain           |
| thunderstorm with heavy rain | 202        | thunderstorm   |
| thunderstorm with light rain |  200       | thunderstorm   |
| proximity thunderstorm       | 211        | thunderstorm   |
| thunderstorm                 | 211        | thunderstorm   |
| light intensity drizzle      | 300        | drizzle        |
| mist                         | 701        | mist           |
| fog                          | 741        | fog            |

weather_description and weather_id match nearly 1:1, and weather_main contains faily intuitive groupings of weather types.  I opted to one-hot encode weather_main and discard the other two to minimize dimensionality.


### Correlation Matrices

To avoid making a single, massive, unreadable correlation matrix with all of my features, I decided to add price to the weather DataFrame and make a separate, moderately-readable one for each subset.  When it comes to weather, it appears that wind speed and temperature are the only features which are routinely correlated with energy price (bottom row).

![](images/clean_weather_corr_sparse.png)

Energy

The energy dataset provides a much more visually interesting (and analytically helpful) matrix.  Lignite, gas, coal, and oil generation , along with total load, all appear positively correlated with price.  Meanwhile, onshore wind and pumped storage consumption appear to be negatively correllated with price.

![](images/clean_energy_corr.png)


### VIF
perhaps unsurprisingly, temp, max_temp, and min_temp were highly correlated.  The final straw in removing these columns was seeing them with variance inflation factors over 500.

maybe dump this section

### PCA: 
first 8 priciple componants make up 40% of the variance, not great.  There is definitely some signal, but it might take more featurization than I originally planned to get a model working well.



### Model Selection

#### Random Forest
From the outset, I was planning on using a random forest regressor on this data.  
![num estimators plot](images/rf_num_estimator_plot.png)

It appears that a case can be made that the best num_estimators here is just about 10.  Running my RandomForest with 10 estimators produced surprisingly high r^2 scores for my train and test data, 0.97 and 0.82 respectively.  I came away from these results concerned that I had introduced some leakage that was causing my model to overfit.

#### SKlearn Pipeline
I used SKlearn's pipeline class to compare my random forest with 3 other models.  The similar results from the random forest via the sklearn pipeline reassured me that I had not caused any leakage with my treatment of the standardization and train-test-split in my custom pipeline


| Model                 | Train R^2 | Test R^2 | Holdout R^2 |
|-----------------------|-----------|----------|-------------|
| RandomForestRegressor | 0.97      | 0.85     | 0.85        |
| Lasso(alpha=0.03)     | 0.44      | 0.43     |             |
| Ridge(alpha=0.03)     | 0.44      | 0.43     |             |
| LinearRegression      | 0.44      | 0.43     |             |

These results indicate that the relationships at play between the features and the target are not linear, and that in order to get highly interpretable results from a Linear Model, significant feature engineering would be required.
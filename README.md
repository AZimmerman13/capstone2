# capstone2

## EDA
There were several columns that did not appear useful, such as 'weather_icon' which, according to the data description, contained only information about the "Weather icon code for website".  I removed this column and sought to further reduce dimensionality among the categorical features.
array(['01n', '01d', '01', '02n', '02d', '02', '03', '04n', '04', '10n',
       '03n', '10', '04d', '03d', '10d', '50d', '09n', '11d', '11n',
       '09d', '50n'], dtype=object)


I had a suspicion that weather_description and weather_id were redundant as they contain a similar number of categories.  Some simple exploration confirmed my suspicion, and found additional collinearity/sub categorization in the weather_main column, clear upon moderately close inspection:


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

weather_description and weather_id match nearly 1:1, and weather_main contains faily intuitive groupings of weather types.  I opted to use weather_description and discard the other two.


The weather data was for the 5 largest cities in Spain:

Madrid, Barcelona, Valencia, Sevilla, Bilbao

This dataset was relatively clean save for the ' ' in front of 'Barcelona' in every row, which made for a nice little trouble-shooting session.

Barcelona pressure contained ~45 rows with reading between 1 and 2 orders of magnitude above the maximum for all other cities.

no snow in seville or barcelona

perhaps unsurprisingly, temp, max_temp, and min_temp were highly correlated


The energy dataset was incomplete in a few areas, namely that the 'generation fossil coal-derived gas', 'generation fossil oil shale', 'generation fossil peat', 'generation geothermal', and 'generation marine' contained only zeros, and 'generation hydro pumped storage aggregated' contained all null values. 
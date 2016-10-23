---
layout: post
title: "Assessing Accuracy of Google Maps Cycling Estimates"
excerpt: "I compare cycling times from real trips of Bay Area Bike Share users to Google Maps cycling time estimates."
categories: [R, Viz]
comments: true
---

I like riding my bike for transportation whenever possible, and like many others, I often use Google Maps to find a route that is both safe and direct. However, their cycling directions sometimes seem a bit off, and I have always wondered how accurate the time estimates actually are for the average cyclist. This skepticism has been echoed by [others](http://www.betterbybicycle.com/2014/09/how-accurate-are-google-maps-cycling.html) as well.

Recently, I have been experimenting with data from the Bay Area Bike Share (BABS) system -- a network of shared bikes docked at stations scattered around the Bay Area. In this analysis, I compare Google Maps cycling time estimates with actual trip times from the BABS to investigate their accuracy.



## Data

### BABS

The BABS makes trip data publically available [here](http://www.bayareabikeshare.com/open-data). It consists of individual records for 669,959 trips made between 8/31/2013 and 8/31/2015 complete with information like origin, destination, duration, start time, and more.



There are five different cities served by the system: San Francisco, Redwood City, Palo Alto, Mountain View, and San Jose. While it is possible to take a bike between cities, I filter those trips out because of their rarity and the likelihood that a trip from San Francisco to Palo Alto, for example, includes a stint on the Caltrain.

Below is a histogram of the actual duration of these trips. It is cut off at 1,800 seconds (30 minutes), but there are some trips that take significantly longer. These outliers are dealt with later. The distribution is skewed off to the right as we would expect since a cyclist can take an arbitrarily long amount of time to complete their trip.

![plot of chunk unnamed-chunk-3](/assets/Rfig/unnamed-chunk-3-1.svg)

### Google Maps Estimates

Through the Google Maps Directions API, I pull cycling estimates for every pair of stations in the same city. See [here](https://github.com/llefebure/bike-sharing/blob/master/R/google-api-trip-estimates.R) for the full code that generates those numbers. In addition, some stations changed location, so certain routes affected by this have multiple estimates with corresponding date ranges.



The relationship between the cycling distance (in meters) of a trip and its expected duration (in seconds) as reported by Google Maps is plotted below for each route. There is clearly a strong linear trend with some heteroskedasticity. As trips increase in distance, the variance of their expected duration increases. Trips range in distance from about 50 meters to 6,000 meters and in duration from about 10 seconds to 23 minutes.

![plot of chunk unnamed-chunk-5](/assets/Rfig/unnamed-chunk-5-1.svg)

From this data, I infer the expected average cycling speed for each route by scaling the ratio of duration and time. This distribution is shown below. The average expected speed of a route is between 8 and 9 miles per hour.

![plot of chunk unnamed-chunk-6](/assets/Rfig/unnamed-chunk-6-1.svg)



## Removing Major Outliers

My ultimate goal is to compare actual trip times with Google Maps estimates, so I need to filter out trips that were not continuous point to point journeys. For example, it is possible that a tourist stops several times to take photos before reaching their destination. Trips such as these could very easily skew the analysis. A quick look at the distribution of trip times reveals at least one obvious outlier.


| Min| First Quartile| Median|  Mean| Third Quartile|      Max|
|---:|--------------:|------:|-----:|--------------:|--------:|
|  60|            341|    508| 902.5|            731| 17270000|

Clearly nobody made a continuous trip of over 17,000,000 seconds, which equates to approximately 200 days. However, finding the not so obvious discontinuous trips is more challenging. The BABS imposes overage charges on any trip that lasts longer than 30 minutes, so the system is setup to discourage those longer, discontinuous journeys. As a first step, I will filter out all trips longer than 30 minutes.



There is an additional piece of important information attached to each trip -- the subscriber type. This tells us more about the rider. Subscribers are those with annual or 30 day memberships, and customers are those with 24 hour or 3 day passes. I expect that subscribers are those that use the system for mostly commuting and transportation (the trips we are interested in), while customers can include tourists who use the system for exploring the area. Plotted below is the distribution of the difference between a trip's actual time and its estimated time stacked by subscriber type. As expected, customers ride slower than subscribers. I filter out these trips as well.

![plot of chunk unnamed-chunk-10](/assets/Rfig/unnamed-chunk-10-1.svg)



## Analysis

The remaining dataset now has all trips shorter than 30 minutes made by subscribers. In the analysis that follows, I work with this dataset and explore descriptive statistics, particularly the median. Because the distribution of trip times is heavily skewed to the right and some outliers inevitably remain, the median is a more meaningful gauge of the "average" or "typical" trip than the mean itself.

The distribution of the difference (in seconds) between the actual and estimated time of each trip is shown below. The median is just 8 seconds, suggesting that the Google Maps estimates are quite good. Despite the long right tail, the quartiles are perfectly symmetric as well -- the middle 50% of trips were within +/- 81 seconds of the median.


|  Min| First Quartile| Median|  Mean| Third Quartile|  Max|
|----:|--------------:|------:|-----:|--------------:|----:|
| -826|            -73|      8| 21.96|             89| 1739|

![plot of chunk unnamed-chunk-13](/assets/Rfig/unnamed-chunk-13-1.svg)

However, trips vary in length significantly. A one minute difference for a five minute trip is much different than a one minute difference for a twenty minute trip. Instead of looking at the difference between the actual and estimated times, I look at this difference scaled by the expected time below. Some outliers were missed, but the distribution is still remarkably symmetric. The median trip is just 2% longer than estimated.




|     Min| First Quartile|  Median|   Mean| Third Quartile|   Max|
|-------:|--------------:|-------:|------:|--------------:|-----:|
| -0.8618|        -0.1494| 0.01944| 0.1195|         0.2314| 42.41|

![plot of chunk unnamed-chunk-16](/assets/Rfig/unnamed-chunk-16-1.svg)

Next, I segment the trips by city. Note that the vast majority of trips were in San Francisco and that the median trip there was even more precise at just under 1% or 3 seconds slower than the Google Maps estimate. In the other cities, the accuracy is not nearly as good, but there are far fewer trips to make conclusions from.


|City          | Count of Trips| Proportion of Total| Median Difference From Estimated Duration (proportion)| Median Difference From Estimated Duration (s)|
|:-------------|--------------:|-------------------:|------------------------------------------------------:|---------------------------------------------:|
|Mountain View |          14841|           0.0267041|                                              0.1701031|                                            41|
|Palo Alto     |           3285|           0.0059109|                                              0.4472574|                                           121|
|Redwood City  |           2534|           0.0045595|                                              0.3103448|                                            82|
|San Francisco |         504580|           0.9079148|                                              0.0079051|                                             3|
|San Jose      |          30517|           0.0549107|                                              0.0754017|                                            27|



![plot of chunk unnamed-chunk-19](/assets/Rfig/unnamed-chunk-19-1.svg)


Finally, I segment the trips by their expected distance to determine whether the accuracy of estimates differs between shorter and longer trips. The 0%-25% bucket contains, for example, all trips for which the Google Maps estimate was among the shortest quarter. The plots below show that the estimates for shorter trips are slightly too fast and those for longer trips are a bit slow. Two factors that could account for this difference are:

* Only better and faster cyclists attempt longer journeys on the bike, making those estimates seem too slow.
* The overhead amount of time that it takes to check out a bike, possibly adjust the seat, and check in the bike at the other end impacts shorter journeys more, making those estimates seem too fast.




|Quartile Bucket | Median Difference From Estimated Duration (proportion)| Median Difference From Estimated Duration (s)|
|:---------------|------------------------------------------------------:|---------------------------------------------:|
|0%-25%          |                                              0.1666667|                                            36|
|25%-50%         |                                              0.0516796|                                            20|
|50%-75%         |                                             -0.0503472|                                           -28|
|75%-100%        |                                             -0.0493179|                                           -38|



![plot of chunk unnamed-chunk-23](/assets/Rfig/unnamed-chunk-23-1.svg)

## Conclusion

Google Maps estimates seem to be very accurate for the median rider in San Francisco. When segmented out by trip distance, the estimates were slightly too fast for shorter routes and too slow for longer routes. This analysis gives some quantitative intuition about urban cycling estimates in the Bay Area, but I cannot make broader conclusions because the data was restricted to subscribers in the BABS on very specific and short, urban routes. It is quite possible that Google even uses this data to calibrate their models. They probably want their estimates to be as accurate as possible in their backyard!

One factor that could have skewed these results is missed outliers (trips that were not continuous point to point journeys). As discussed previously, I removed some of these trips because of their unrealistically long length, but there are definitely some that were missed. It is possible that there exists outliers on the other end of the spectrum too -- trips that were unrealistically short. The BABS employs rebalancers that drive around and redistribute bikes to different stations in the system. I could not figure out whether these types of "trips" are included in the dataset or not, but their presence would certainly be probelmatic. Despite all of these issues, I think that the median still gives a realistic picture because it is relatively resistant to outliers.

Now when I use Google Maps for cycling directions to get somewhere, I have some quantitative intuition as to their time accuracy. I will put aside ego, honestly assess how much better (or worse) than the median rider I think I am on that day, and extrapolate accordingly!

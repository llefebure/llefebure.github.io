---
layout: post
title: "Coronavirus Earnings Season"
excerpt: "A look at the frequency of coronavirus mentions in earnings calls during Q1 2020"
categories: [Viz, Text]
comments: true
image:
  feature: /assets/Pyfig/coronavirus-earnings-graphic_1_0.svg
---
Axios [reported](https://www.axios.com/coronavirus-global-economic-risk-stock-market-china-24da74a5-c824-4c79-a8c7-220213d28824.html) several weeks back on growing concern from economists about the understated risks to the global economy posed by coronavirus. The report cites an interesting statistic about the attention paid to coronavirus on recent earnings calls.

> Of the 364 companies that have held Q4 earnings calls, 138 cited the term "coronavirus" during the call, and about 25% of those included some impact from the coronavirus or modified guidance due to the virus, according to FactSet.

Additionally, the author cites a [note](https://www.guggenheiminvestments.com/perspectives/global-cio-outlook/coronavirus-impact-on-the-global-economy) in which Scott Minerd, global CIO of Guggenheim Investments, compares the "cognitive dissonance" of market participants to the assurances given by Neville Chamberlain to Britain on the eve of World War II.

One day after the Axios report, the S&P 500 index hit an all time high before fears materialized and markets crashed. My mind came back to that statistic, so I pulled the transcripts for a quick analysis.

## Data Acquisition

The Motley Fool provides earnings call transcripts for the companies that they cover in a [centralized place](https://www.fool.com/earnings-call-transcripts/), making it quite easy to scrape and parse. Global media outlets picked up the story in early January, so I pulled transcripts from all earnings calls occurring in 2020 (as of March 28th). This leaves us with data on 2,788 unique earnings calls.

## Analysis

1,284 (46%) of the transcripts in this dataset include a mention of "covid" or "coronavirus". However, there is a strong temporal trend which can be seen in the graphic below as recent earnings calls are far more likely to mention it than earlier ones.

The first recorded mention was on 1/23 during the Q&A section of the American Airlines call.

> Hi. Hey, how are you? My question has to do with the coronavirus but are kind of on three fronts. I know it's early, but are you guys seeing any booking impact at all? Have you contemplated any travel waivers for the region? And lastly, maybe this one is for Robert. What measures is American taking to protect passengers and crew? -<i>Dawn Gilbertson, Consumer Travel Reporter at USA Today</i>

> Thanks, Dawn. So first off, we haven't -- it's too soon to see any impact. Our network isn't that extensive in Asia. But we're on top of it. We're working with CBP, the CDC and public health officials, as well as our medical resources here to make sure that we're following all best practices. We're doing that with an intent to make sure that we take care of our customers and team members. We've seen viruses in the past that we've had to make accommodations for and to be prepared for. We're doing all those same things right now. And we're going to watch it and make sure that we take aggressive action if there is a need to. -<i>Robert Isom, President</i>

The following week, several additional companies took note of the growing outbreak. In Apple's call on 1/28, for example, Tim Cook addressed the issue in the context of Apple's support for the local community.

> This quarter, we also announced a $2.5 billion plan to help address the housing availability and affording crisis in our home state of California. We feel a great responsibility to help the region we have always called home, stay vibrant and to ensure that it remains a great place for everyone to live and raise a family, including those who do so much to serve the community like firefighters and teachers. In much more recent news, we are closely following the development of the coronavirus. We are donating to groups that are working to contain the outbreak. We are working closely with our Apple team members and partners in the affected areas, and our thoughts are with all of those affected across the region. -<i>Tim Cook, CEO</i>

Throughout February, around half of calls included a mention, and that number has climbed to nearly 100% in March as state and local governments around the US began to lock down businesses.


![svg](/assets/Pyfig/coronavirus-earnings-graphic_1_0.svg)


------------------

The code for this analysis (including the script for scraping the Motley Fool) can be found [here](https://github.com/llefebure/misc/tree/master/coronavirus-earnings-calls).

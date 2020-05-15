<p align="center">
  <img src="main/img/pit-swamp-of-despair.jpeg" width = 900 ><sup>(2)</sup>
</p>

*Capstone II Project for Galvanize Data Science Immersive, Week 8*

*by Marc Russell*

# Dude, What's That Car?
### Identify and classify automobiles using images 


## Table of Contents
- [Introduction](#introduction)
  - [Motivation](#motivation)
  - [The Data](#the-data)
  - [Questions](#questions)
- [Image Pipeline](#image-pipeline)
  - []
  - [Transformations](#transformations)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Dataset Overview](#dataset-overview)
  - [Feature Categories](#feature-categories)
  - [Exploration](#exploration)
    - [Weather](#weather)

- [Model](#model)
- [Conclusion](#conclusion)
- [Citation](#citation)


# Introduction

## Motivation 

Law enforcement are commonly tasked with confirming that license plates match
the vehicle that they are attached to. 

This job is greatly assisted by automatic license plate reader/recognition technology (ALPR) which relays the licence plate number to the officer. After the ALPR sends the plate number to the officer’s mobile computer, there is one tedious task remaining - confirming that the vehicle matches the description linked to the plate number. 

<p align="center">
  <img src="main/img/mobile-alpr.png" width = 600 ><sup>(2)</sup>
</p>

I haven’t seen one (public) attempt to apply machine learning and image recognition to this tedious job.

According to a 2011 study, 71% of police agencies reported using ALPR and 85% plan to acquire or increase their use of the technology over the next five years.<sup>2</sup>


### The Job and ALPR

Owners of motorized vehicles driven on public thoroughfares are required by law to annually register their vehicles and to attach license plates that are publicly and legibly displayed.

<p align="center">
  <img src="main/img/alpr-purpose.png" width = 500 ><sup>(2)</sup>
</p>

Law enforcement practitioners are often searching for vehicles that have been reported stolen, are suspected of being involved in criminal or terrorist activities, are owned by persons who are wanted by authorities, have failed to pay parking violations or maintain current vehicle license registration or insurance, or any of a number of other legitimate reasons. To aid in this search ALPR Technology can relay theh following information:

    a. Character and/or plate color
    b. Plate design factors (logos, stacked characters, etc.)
    c. State of origin (i.e., the state which issued the plate)
    d. Plate covers or other obstructions (e.g., bent, dirty, trailer hitch obstruction, etc.)
    e. Plate location on the vehicle
    f. Interval between vehicles
    g. Vehicle speed
    h. Lighting conditions (e.g., day vs. night)
    i. Weather conditions (e.g., snow, rain, fog)
    j. ALPR equipment (e.g., age and/or ability of the ALPR camera)
    k. ALPR implementation (e.g., camera angle) 
  
Issues arrise when 

alpr-purpose.png

## The Data

The dataset comes from vmmrdb.cecsresearch.org and consists of vehicle make/model/year labels assigned to images scraped from sites like Craigslist and Ebay. 

Containing 9,170 classes consisting of 291,752 images, covering models manufactured between 1950 to 2016 it is far from complete. 

Images were taken by different users, on different imaging devices, and cover multiple perspectives, ensuring a wide range of variations to account for various scenarios that could be encountered in a real-life scenario. The cars are not well aligned, and some images contain irrelevant backgrounds.

<p align="center" style="font-size:1.5em;">
Example Images
</p>
<figure class="image" align="center">
  <img src="main/img/f150-full.jpg" width = 450 >
  <figcaption>Above: Originial Image (~600x500)
</figure>
<p align="center">
  <img src="main/img/f150-g64.jpg" width = 150 >
  <img src="main/img/f150-g32.jpg" width = 150 >
  <img src="main/img/f150-g16.jpg" width = 150 >
</p>
<p align="center">
Left: 64x64 &nbsp; &nbsp;  &nbsp; | &nbsp;  &nbsp;  &nbsp; Middle: 32x32 &nbsp; &nbsp;  &nbsp; | &nbsp; &nbsp;  &nbsp; Right: 16x16</figcaption>
</p>
<p align="center">
  <img src="main/img/civic-g64.jpg" width = 150 >
  <img src="main/img/civic-g32.jpg" width = 150 >
  <img src="main/img/civic-g16.jpg" width = 150 >
</p>
<figure class="image" align="center">
  <img src="main/img/civic-full.jpg" width = 450 >
  <figcaption>Above: Originial Image (~600x500)</figcaption>
</figure>


<p align="center">
  <img src="main/img/civic-g16.jpg" width = 650 >
</p>


MVP (Minimum Viable Product)
1. fdf
2. (unsupervised learning) car recommenders, price predictors, (more supervised)
satellite car-image identification, (NLP) car description/rating analysis, and
making this into a deployable application for mobile phone users and for police
mobile computers.

[Back to Top](#Table-of-Contents)

# Exploratory Data Analysis

## Dataset Overview

Accident Dataset:
 - 49 columns and 3 million rows
 - Columns describe the accident event
 - Each row is a unique accident event
 
IRS Dataset:
 - 153 columns and 1 million rows
 - Each Column is an amount for each tax variable
 - Each Zipcode has 6 rows (one for each income division)

## Feature Categories


* Accident Results:

       Accident Counts (rows), Severity, Length of Road Effected, Description

* Weather:

      Temperature, Weather Condition, Precipitation, Visibility

* Road Attributes:

      Nearby Traffic Signs (Yield, Stop, etc.), Railroads, Speed Bumps, Merges
      
* Geo-positional:

      Latitude, Longitude, State, County, City, Zipcode


## Exploration



[Back to Top](#Table-of-Contents)


# Conclusion

I initially wanted to focus my study on how weather conditions effect accident rates. It turned out that adjusting my data to account for the fact that is not independent and identically distributed (IID) proved difficult. To remedy this, I would source a dataset of locational-hourly-weather and merged with the accident data set. Instead I decided to shift focus to exploring how accident counts change with location.

I found the relationships of state-to-state accident rates very interesting. Despite the data not being IID I was able to merge a population dataset with the accident counts to achieve a rate - the number of accidents per person per year in each state. This subtle change allows us to compare states between each other despite variable populations. It turns out that South Carolina has the highest accident rate while North and South Dakota have the lowest accident rates.

Lastly I looked at the relationship between time and accident counts. It quickly became apparent that the accident counts over time were also not IID. I decided to explore the relationship anyway, being careful not to draw strict conclusions. Over the course of the day it was expected to see the double rush hour peak. On the other hand, I found it surprising that the seasonal rise in accident counts happened during fall and not winter. 

I hope to return to this dataset soon and explore the remaining corners. It will also be a fun challenge to merge the aforementioned dataset which will allow me to draw conclusions where I was not able to before.

[Back to Top](#Table-of-Contents)

# Citation

<sup>(1)</sup>


*<sup>2</sup>Automated License Plate Reader Systems: Policy and Operational Guidance for Law Enforcement, https://www.ncjrs.gov/pdffiles1/nij/grants/239604.pdf *

[Back to Top](#Table-of-Contents)

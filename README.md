<img src="https://r3.whistleout.com/public/images/articles/2017/05/CO.png">

# Forest-Type-Prediction
This is my first Kaggle competition. It is about predicting the different forest type given with various geographic features of Colorado. 


<div id="introduction" class="section level2">

<h2>Introduction</h2>

<p>There are different forest types around the world and their differences usually involve with ecological factors, geographical locations, habitants, and weather. It is quite interesting how unique each forest type is. In this competition, we are given data relating to the forest type and the objective is to predict the forest type as accurately as possible. By diving with my knowledge in machine learning and performing two model testing, I have decided to use random forest since it generated a higher accuracy rate than the multinomial logistic regression. In my submission at Kaggle, I got a higher score using random forest than the multinomial logistic regression.</p>

</div>

<div id="objective" class="section level2">

<h2>Objective</h2>

<p>The objective of this challenge is to predict the forest type based on various categorial and quantitative variables.</p>

</div>

<div id="background" class="section level2">

<h2>Background</h2>

<p>Since the objective is to create a model that will predict the cover type, it is best to describe the qualitative variable, forest cover and wilderness area.</p>

<div id="cover-type" class="section level3">

<h3>Cover Type</h3>

<div id="sprucefir" class="section level4">

<h4>Spruce/Fir</h4>

<p>Their spatial patterns are dependent on the temperature. Because of the thicker trunks, they are able to survive fires.</p>

</div>

<div id="lodgepole-pine" class="section level4">

<h4>Lodgepole Pine</h4>

<p>It is commonly located near the ocean shore and in dry montane forests to the subalpine. It is an evergreen pine.</p>

</div>

<div id="ponderosa-pine" class="section level4">

<h4>Ponderosa Pine</h4>

<p>Pinus ponderosa is a large coniferous pine (evergreen) tree. The bark helps to distinguish it from other species. Like most western pines, the ponderosa generally is associated with mountainous topography. <i> From Wikipedia </i></p>

</div>

<div id="cottonwoodwillow" class="section level4">

<h4>Cottonwood/Willow</h4>

<p>Poplars of the cottonwood section are often wetlands or riparian trees. The aspens are among the most important boreal broadleaf trees.</p>

</div>

<div id="aspen" class="section level4">

<h4>Aspen</h4>

<p>Aspen trees are all native to cold regions with cool summers, in the north of the Northern Hemisphere, extending south at high-altitude areas such as mountains or high plains. Aspens typically grow in environments that are otherwise dominated by coniferous tree species, and which are often lacking other large deciduous tree species.</p>

</div>

<div id="douglas-fir" class="section level4">

<h4>Douglas-fir</h4>

<p>It is a coniferous type whose leaves are evergreen (i.e. always green in colors). It prefers neutral or acidic soils. This type is used widely for timber due to its strength, hardness, and durability.</p>

</div>

<div id="krummholz" class="section level4">

<h4>Krummholz</h4>

<p>It is a type of stunted, deformed vegetation encountered in the subarctic and subalpine tree line landscapes, shaped by continual exposure to fierce, freezing winds. Under these conditions, trees can only survive where they are sheltered by rock formations or snow cover. As the lower portion of these trees continues to grow, the coverage becomes extremely dense near the ground.</p>

</div>

</div>

<div id="wilderness-type" class="section level3">

<h3>Wilderness Type</h3>

<p>Next, we will discuss the four wilderness type.</p>

<div id="rawah-wilderness" class="section level4">

<h4>Rawah Wilderness</h4>

<ul>

<li>protects a scenic high country of U-shaped glacier-carved valleys and peaks reaching 12, 951 feet.</li>

<li>To the south and west of the area lies an almost roadless Colorado State Forest, an unofficial extension of the Wilderness.</li>

<li>Melting snow fills 26 lakes within the area.</li>

<li>On the upper forested slopes of the mountains, especially in the southern section, clusters of old-growth spruce and fir abound.</li>

<li>has a total of 74,408 acres.</li>

<li>located on the Canyon Lakes Ranger District of the Roosevelt National Forest in Colorado near the Wyoming border and also in the Routt National Forest to its south.</li>

<li>There are 85 miles (137 km) of trails in the area and elevation ranges from 8,400 feet (2,600 m) to 13,0000 feet (4,000 m).</li>

<li>The temperature in the Rawah Wilderness ranges from a low of 5 degrees Fahrenheit during the winder and a high of 77 degrees Fahrenheit during the summer.</li>

</ul>

</div>

<div id="neota-wilderness" class="section level4">

<h4>Neota Wilderness</h4>

<ul>

<li>located on the Canyon Lakes Ranger District of the Roosevelt National Forest in Colorado.</li>

<li>This wilderness area encompasses 9,924 acres (40 km2) and is bordered on the south by Rocky Mountain National Park.

<ul>

<li>Elevation ranges frm 10,000 ft (3,000 m) to 11,9896 ft (3,626 m) in the Rocky Mountains. There are only .5 miles (2.4 km) of trail in this area.</li>

<li>bordered by the Rawah Wilderness to the northwest, the COmanche Peak Wilderness to the east, and the Rocky Mountain National Park Wilderness to the south.</li>

</ul></li>

</ul>

</div>

<div id="comanche-peak-wilderness" class="section level4">

<h4>Comanche Peak Wilderness</h4>

<ul>

<li>located in the Roosevelt National Forest on the Canyon Lakes Ranger District in Colorado along the northern boundary of Rocky Mountain National Park.</li>

<li>comprises of 66,791-acre (27, 029 ha).</li>

<li>There are 121 miles (195 km) of hiking trails inside the wilderness.</li>

<li>There are also 7 named peaks, 6 named lakes (inclduing COmanche Reservoir) and 16 named rivers and creeks within the wilderness boundaries.</li>

<li>From the trails, the Wilderness appears to be nothing but trees until the forst gives way to abundant alpine tundra along the souther boundary near Comanche Peak.</li>

<li>Beyond Comanche to the South lies Rocky Mountain National Park.</li>

<li>The Cache la Poudre River tumbles over cascades or runs quietly in pools for over 10 miles to the north and west.</li>

</ul>

</div>

<div id="cache-la-poudre-wilderness" class="section level4">

<h4>Cache La Poudre Wilderness</h4>

<ul>

<li>located on the Canyon Lakes Ranger District on the Roosevelt National Forest in Colorado.</li>

<li>This wilderness covers 9,258 acres (37.47 km2) and is characterized by steep, rugged terrain along the Cache la Poudre River.</li>

<li>Elevations in this area varies from 6,200 feet(1,900 m) to 8,600 feet (2,600 m).</li>

</ul>

</div>

</div>

</div>

<div id="data-loading-and-initial-analysis" class="section level2">

<h2>Data Loading and Initial Analysis</h2>

<p>First, the researcher will load the libraries to be used in the analysis, the data itself, and do the preliminary analysis. This analysis will involve looking at the data type and number of null values.</p>

<p>The data is composed of train set and test set. The train set comprises of 15,120 observations while the test set comprises of 365,892 observations.</p>

<pre class="r"><code>## Loading the libraries

library(ggplot2)

library(tidyverse)

library(dplyr)

library(caret)

library(randomForest)

library(reshape2)





## Loading the datasets

    

    train &lt;- read.csv(&quot;train.csv&quot;, header = TRUE)

    

    ## Due to the limited computing power of my computer, I have divided the test data set into six. These data sets are already processed.

    test1 &lt;- read.csv(&quot;test1.csv&quot;, header = TRUE)

    test2 &lt;- read.csv(&quot;test2.csv&quot;, header = TRUE)

    test3 &lt;- read.csv(&quot;test3.csv&quot;, header = TRUE)

    test4 &lt;- read.csv(&quot;test4.csv&quot;, header = TRUE)

    test5 &lt;- read.csv(&quot;test5.csv&quot;, header = TRUE)

    test6 &lt;- read.csv(&quot;test6.csv&quot;, header = TRUE)

    

    ## Combining all of the datasets into one

    test &lt;- rbind(test1, test2, test3, test4, test5, test6)

    

    ## Renaming the Area Type and Soil Type columns of the test dataset

    names(test)[12:13] &lt;- c(&quot;Area_Type&quot;, &quot;Soil_Type&quot;)



## Number of observations

data.frame(data = c(&quot;train&quot;, &quot;test&quot;), columns = c(ncol(train), ncol(test)), rows = c(nrow(train), nrow(test)))</code></pre>

<pre><code>##    data columns   rows

## 1 train      56  15120

## 2  test      13 565892</code></pre>

<pre class="r"><code>## Number of null values in train set

colSums(is.na(train))</code></pre>

<pre><code>##                                 Id                          Elevation 

##                                  0                                  0 

##                             Aspect                              Slope 

##                                  0                                  0 

##   Horizontal_Distance_To_Hydrology     Vertical_Distance_To_Hydrology 

##                                  0                                  0 

##    Horizontal_Distance_To_Roadways                      Hillshade_9am 

##                                  0                                  0 

##                     Hillshade_Noon                      Hillshade_3pm 

##                                  0                                  0 

## Horizontal_Distance_To_Fire_Points                   Wilderness_Area1 

##                                  0                                  0 

##                   Wilderness_Area2                   Wilderness_Area3 

##                                  0                                  0 

##                   Wilderness_Area4                         Soil_Type1 

##                                  0                                  0 

##                         Soil_Type2                         Soil_Type3 

##                                  0                                  0 

##                         Soil_Type4                         Soil_Type5 

##                                  0                                  0 

##                         Soil_Type6                         Soil_Type7 

##                                  0                                  0 

##                         Soil_Type8                         Soil_Type9 

##                                  0                                  0 

##                        Soil_Type10                        Soil_Type11 

##                                  0                                  0 

##                        Soil_Type12                        Soil_Type13 

##                                  0                                  0 

##                        Soil_Type14                        Soil_Type15 

##                                  0                                  0 

##                        Soil_Type16                        Soil_Type17 

##                                  0                                  0 

##                        Soil_Type18                        Soil_Type19 

##                                  0                                  0 

##                        Soil_Type20                        Soil_Type21 

##                                  0                                  0 

##                        Soil_Type22                        Soil_Type23 

##                                  0                                  0 

##                        Soil_Type24                        Soil_Type25 

##                                  0                                  0 

##                        Soil_Type26                        Soil_Type27 

##                                  0                                  0 

##                        Soil_Type28                        Soil_Type29 

##                                  0                                  0 

##                        Soil_Type30                        Soil_Type31 

##                                  0                                  0 

##                        Soil_Type32                        Soil_Type33 

##                                  0                                  0 

##                        Soil_Type34                        Soil_Type35 

##                                  0                                  0 

##                        Soil_Type36                        Soil_Type37 

##                                  0                                  0 

##                        Soil_Type38                        Soil_Type39 

##                                  0                                  0 

##                        Soil_Type40                         Cover_Type 

##                                  0                                  0</code></pre>

<pre class="r"><code>## Number of null values in test set 

colSums(is.na(test))</code></pre>

<pre><code>##                              ï..Id                          Elevation 

##                                  0                                  0 

##                             Aspect                              Slope 

##                                  0                                  0 

##   Horizontal_Distance_To_Hydrology     Vertical_Distance_To_Hydrology 

##                                  0                                  0 

##    Horizontal_Distance_To_Roadways                      Hillshade_9am 

##                                  0                                  0 

##                     Hillshade_Noon                      Hillshade_3pm 

##                                  0                                  0 

## Horizontal_Distance_To_Fire_Points                          Area_Type 

##                                  0                                  0 

##                          Soil_Type 

##                                  0</code></pre>

<p>It can be seen that there are no null values in our dataset.</p>

</div>

<div id="data-cleaning" class="section level2">

<h2>Data Cleaning</h2>

<p>Second, the researcher will clean the data. The following will be performed during the cleaning:</p>

<ol style="list-style-type: decimal">

<li>Conversion of categorical variables into factor class.</li>

<li>Melting the Wilderness Area variables.</li>

<li>Melting the Soil Type variables.</li>

<li>Removing observations where the wilderness area and soil type are zero.</li>

<li>Converting the Elevation, Aspect, Slope, Horizontal Distance to Fire Points and Hydrology, Vertical Distance to Roadways and Hydrology, and Hillshade into numeric class.</li>

</ol>

<p>In addition to that, I have replaced grouped the soil type with similar features.</p>

<pre class="r"><code>## Replacing the numeric values into text values

    

    train$Wilderness_Area1 &lt;- ifelse(train$Wilderness_Area1 == 1, &quot;Rawah&quot;, train$Wilderness_Area1)

    train$Wilderness_Area2 &lt;- ifelse(train$Wilderness_Area2 == 1, &quot;Neota&quot;, train$Wilderness_Area2)

    train$Wilderness_Area3 &lt;- ifelse(train$Wilderness_Area3 == 1, &quot;Comanche Peak&quot;, train$Wilderness_Area3)

    train$Wilderness_Area4 &lt;- ifelse(train$Wilderness_Area4 == 1, &quot;Cache la Poudre&quot;, train$Wilderness_Area4)

  

    train$Soil_Type1 &lt;- ifelse(train$Soil_Type1 == 1, &quot;Cathedral Family&quot;, train$Soil_Type1)

    train$Soil_Type2 &lt;- ifelse(train$Soil_Type2 == 1, &quot;Vanet&quot;, train$Soil_Type2)

    train$Soil_Type3 &lt;- ifelse(train$Soil_Type3 == 1, &quot;Haploborolis&quot;, train$Soil_Type3)

    train$Soil_Type4 &lt;- ifelse(train$Soil_Type4 == 1, &quot;Ratake family&quot;, train$Soil_Type4)

    train$Soil_Type5 &lt;- ifelse(train$Soil_Type5 == 1, &quot;Vanet&quot;, train$Soil_Type5)

    train$Soil_Type6 &lt;- ifelse(train$Soil_Type6 == 1, &quot;Vanet&quot;, train$Soil_Type6)

    train$Soil_Type7 &lt;- ifelse(train$Soil_Type7 == 1, &quot;Gothic&quot;, train$Soil_Type7)

    train$Soil_Type8 &lt;- ifelse(train$Soil_Type8 == 1, &quot;Supervisor&quot;, train$Soil_Type8)

    train$Soil_Type9 &lt;- ifelse(train$Soil_Type9 == 1, &quot;Troutville&quot;, train$Soil_Type9)

    train$Soil_Type10 &lt;- ifelse(train$Soil_Type10 == 1, &quot;Bullwark&quot;, train$Soil_Type10)

    train$Soil_Type11 &lt;- ifelse(train$Soil_Type11 == 1, &quot;Bullwark&quot;, train$Soil_Type11)

    train$Soil_Type12 &lt;- ifelse(train$Soil_Type12 == 1, &quot;Legault&quot;, train$Soil_Type12)

    train$Soil_Type13 &lt;- ifelse(train$Soil_Type13 == 1, &quot;Catamount&quot;, train$Soil_Type13)

    train$Soil_Type14 &lt;- ifelse(train$Soil_Type14 == 1, &quot;Pachic&quot;, train$Soil_Type14)

    train$Soil_Type15 &lt;- ifelse(train$Soil_Type15 == 1, &quot;unspecified&quot;, train$Soil_Type15)

    train$Soil_Type16 &lt;- ifelse(train$Soil_Type16 == 1, &quot;Cryaquolis&quot;, train$Soil_Type16)

    train$Soil_Type17 &lt;- ifelse(train$Soil_Type17 == 1, &quot;Gateview&quot;, train$Soil_Type17)

    train$Soil_Type18 &lt;- ifelse(train$Soil_Type18 == 1, &quot;Rogert&quot;, train$Soil_Type18)

    train$Soil_Type19 &lt;- ifelse(train$Soil_Type19 == 1, &quot;Typic&quot;, train$Soil_Type19)

    train$Soil_Type20 &lt;- ifelse(train$Soil_Type20 == 1, &quot;Typic&quot;, train$Soil_Type20)

    train$Soil_Type21 &lt;- ifelse(train$Soil_Type21 == 1, &quot;Typic&quot;, train$Soil_Type21)

    train$Soil_Type22 &lt;- ifelse(train$Soil_Type22 == 1, &quot;Leighcan&quot;, train$Soil_Type22)

    train$Soil_Type23 &lt;- ifelse(train$Soil_Type23 == 1, &quot;Leighcan&quot;, train$Soil_Type23)

    train$Soil_Type24 &lt;- ifelse(train$Soil_Type24 == 1, &quot;Leighcan&quot;, train$Soil_Type24)

    train$Soil_Type25 &lt;- ifelse(train$Soil_Type25 == 1, &quot;Leighcan&quot;, train$Soil_Type25)

    train$Soil_Type26 &lt;- ifelse(train$Soil_Type26 == 1, &quot;Granile&quot;, train$Soil_Type26)

    train$Soil_Type27 &lt;- ifelse(train$Soil_Type27 == 1, &quot;Leighcan&quot;, train$Soil_Type27)

    train$Soil_Type28 &lt;- ifelse(train$Soil_Type28 == 1, &quot;Leighcan&quot;, train$Soil_Type28)

    train$Soil_Type29 &lt;- ifelse(train$Soil_Type29 == 1, &quot;Como&quot;, train$Soil_Type29)

    train$Soil_Type30 &lt;- ifelse(train$Soil_Type30 == 1, &quot;Como&quot;, train$Soil_Type30)

    train$Soil_Type31 &lt;- ifelse(train$Soil_Type31 == 1, &quot;Leighcan&quot;, train$Soil_Type31)

    train$Soil_Type32 &lt;- ifelse(train$Soil_Type32 == 1, &quot;Catamount&quot;, train$Soil_Type32)

    train$Soil_Type33 &lt;- ifelse(train$Soil_Type33 == 1, &quot;Leighcan&quot;, train$Soil_Type33)

    train$Soil_Type34 &lt;- ifelse(train$Soil_Type34 == 1, &quot;Cryorthents&quot;, train$Soil_Type34)

    train$Soil_Type35 &lt;- ifelse(train$Soil_Type35 == 1, &quot;Cryumbrepts&quot;, train$Soil_Type35)

    train$Soil_Type36 &lt;- ifelse(train$Soil_Type36 == 1, &quot;Bross&quot;, train$Soil_Type36)

    train$Soil_Type37 &lt;- ifelse(train$Soil_Type37 == 1, &quot;Cryumbrepts&quot;, train$Soil_Type37)

    train$Soil_Type38 &lt;- ifelse(train$Soil_Type38 == 1, &quot;Leighcan&quot;, train$Soil_Type38)

    train$Soil_Type39 &lt;- ifelse(train$Soil_Type39 == 1, &quot;Moran&quot;, train$Soil_Type39)

    train$Soil_Type40 &lt;- ifelse(train$Soil_Type40 == 1, &quot;Moran&quot;, train$Soil_Type40)

    

## Melting the dataset

    

    train &lt;- train %&gt;% 

                gather(key = &quot;Wilderness Area&quot;, value = &quot;Area_Type&quot;, 12:15) %&gt;%

                gather(key = &quot;Soil&quot;, value = &quot;Soil_Type&quot;, Soil_Type1:Soil_Type40)



## Converting the Cover Type, Area Type, and Soil Type into factors

    train$Cover_Type &lt;- as.factor(train$Cover_Type)

    train$Area_Type &lt;- as.factor(train$Area_Type)

    train$Soil_Type &lt;- as.factor(train$Soil_Type)

        

## Removing observations where the wilderness area and soil type are zero. 

    

    train &lt;- train[!(train$Area_Type %in% c(0)), ]

    train &lt;- train[!(train$Soil_Type %in% c(0)), ]

    train &lt;- select(train, -`Wilderness Area`, -Soil)



    

## Converting other variables into numeric type

    train$Elevation &lt;- as.numeric(train$Elevation)

    train$Aspect &lt;- as.numeric(train$Aspect)

    train$Slope &lt;- as.numeric(train$Slope)

    train$Horizontal_Distance_To_Fire_Points &lt;- as.numeric(train$Horizontal_Distance_To_Fire_Points)

    train$Vertical_Distance_To_Hydrology &lt;- as.numeric(train$Vertical_Distance_To_Hydrology)

    train$Horizontal_Distance_To_Hydrology &lt;- as.numeric(train$Horizontal_Distance_To_Hydrology)

    train$Horizontal_Distance_To_Roadways &lt;- as.numeric(train$Horizontal_Distance_To_Roadways)

    train$Hillshade_3pm &lt;- as.numeric(train$Hillshade_3pm)

    train$Hillshade_9am &lt;- as.numeric(train$Hillshade_9am)

    train$Hillshade_Noon &lt;- as.numeric(train$Hillshade_Noon)</code></pre>

</div>

<div id="data-exploration" class="section level2">

<h2>Data Exploration</h2>

<p>Next, we are going to use ggplot in generating graphs.</p>

<p>Presented below is the descriptive statistics.</p>

<pre class="r"><code>## Descriptive Statistics

summary(train)</code></pre>

<pre><code>##        Id          Elevation        Aspect          Slope     

##  Min.   :    1   Min.   :1863   Min.   :  0.0   Min.   : 0.0  

##  1st Qu.: 3781   1st Qu.:2376   1st Qu.: 65.0   1st Qu.:10.0  

##  Median : 7560   Median :2752   Median :126.0   Median :15.0  

##  Mean   : 7560   Mean   :2749   Mean   :156.7   Mean   :16.5  

##  3rd Qu.:11340   3rd Qu.:3104   3rd Qu.:261.0   3rd Qu.:22.0  

##  Max.   :15120   Max.   :3849   Max.   :360.0   Max.   :52.0  

##                                                               

##  Horizontal_Distance_To_Hydrology Vertical_Distance_To_Hydrology

##  Min.   :   0.0                   Min.   :-146.00               

##  1st Qu.:  67.0                   1st Qu.:   5.00               

##  Median : 180.0                   Median :  32.00               

##  Mean   : 227.2                   Mean   :  51.08               

##  3rd Qu.: 330.0                   3rd Qu.:  79.00               

##  Max.   :1343.0                   Max.   : 554.00               

##                                                                 

##  Horizontal_Distance_To_Roadways Hillshade_9am   Hillshade_Noon

##  Min.   :   0                    Min.   :  0.0   Min.   : 99   

##  1st Qu.: 764                    1st Qu.:196.0   1st Qu.:207   

##  Median :1316                    Median :220.0   Median :223   

##  Mean   :1714                    Mean   :212.7   Mean   :219   

##  3rd Qu.:2270                    3rd Qu.:235.0   3rd Qu.:235   

##  Max.   :6890                    Max.   :254.0   Max.   :254   

##                                                                

##  Hillshade_3pm   Horizontal_Distance_To_Fire_Points Cover_Type

##  Min.   :  0.0   Min.   :   0                       1:2160    

##  1st Qu.:106.0   1st Qu.: 730                       2:2160    

##  Median :138.0   Median :1256                       3:2160    

##  Mean   :135.1   Mean   :1511                       4:2160    

##  3rd Qu.:167.0   3rd Qu.:1988                       5:2160    

##  Max.   :248.0   Max.   :6993                       6:2160    

##                                                     7:2160    

##            Area_Type        Soil_Type   

##  0              :   0   Leighcan :3060  

##  Cache la Poudre:4675   Bullwark :2548  

##  Comanche Peak  :6349   Como     :2016  

##  Neota          : 499   Vanet    :1438  

##  Rawah          :3597   Catamount:1166  

##                         Moran    :1116  

##                         (Other)  :3776</code></pre>

<p>The variables that describe the most the forest type are Area Type, Elevation, and Soil Type. With that, we are going to check area type.</p>

<p>Remember the following integer represents each cover type: 1 - Spruce/Fir 2 - Lodgepole Pine 3 - Ponderosa Pine 4 - Cottonwood/Willow 5 - Aspen 6 - Douglas-fir 7 - Krummholz</p>

<pre class="r"><code>## Aspect  

g &lt;- ggplot(data = train, aes(x = Aspect, fill = Cover_Type))

g + geom_bar() + facet_grid(.~Cover_Type)</code></pre>

![png](https://github.com/jmsebastiancarino/Forest-Type-Prediction/blob/master/Barplot%20of%20Aspect.png)
<pre class="r"><code>## Elevation

h &lt;- ggplot(data = train, aes(x = Elevation, fill = Cover_Type))

h + geom_bar() + facet_grid(.~Cover_Type)</code></pre>

![png](https://github.com/jmsebastiancarino/Forest-Type-Prediction/blob/master/Barplot%20of%20Elevation.png)

<div id="machine-learning-modelling" class="section level2">

<h2>Machine Learning Modelling</h2>

<p>In our dataset, we are going to use multinomial logistic regression. Multinomial logistic regression allows for more than two categories of the dependent variable. In this dataset, we have used the nnet library where neural networks are going to be used to work hand in hand in performing multinomial logistic regression.</p>

<p>Diagnostic tests to be performed will only be Variable Importance</p>

<pre class="r"><code>## Dividing the train set into two: training and validation datasets

    inTrain &lt;- createDataPartition(y = train$Cover_Type, p = 0.70, list = FALSE)

    training &lt;- train[inTrain, ]

    validation &lt;- train[-inTrain, ]



## Creating a predictive model

    library(nnet)</code></pre>

<pre><code>## Warning: package 'nnet' was built under R version 3.5.3</code></pre>

<pre class="r"><code>    model &lt;- multinom(Cover_Type ~. -Id, data = training, maxit=500, trace = TRUE)</code></pre>

<pre><code>## # weights:  259 (216 variable)

## initial  value 20595.513018 

## iter  10 value 19339.588711

## iter  20 value 17064.656909

## iter  30 value 15083.318527

## iter  40 value 14349.729780

## iter  50 value 14140.806569

## iter  60 value 12119.462278

## iter  70 value 10321.255169

## iter  80 value 9379.559553

## iter  90 value 8639.145556

## iter 100 value 8108.965588

## iter 110 value 7741.836069

## iter 120 value 7564.119089

## iter 130 value 7508.141577

## iter 140 value 7479.753080

## iter 150 value 7472.607116

## iter 160 value 7471.939780

## iter 170 value 7471.450270

## iter 180 value 7471.166562

## iter 190 value 7471.057724

## iter 200 value 7471.029011

## iter 210 value 7471.014770

## final  value 7471.013917 

## converged</code></pre>

<pre class="r"><code>## Measuring the variance importance

    ImpVar &lt;- varImp(model)

    ImpVar$Variables &lt;- row.names(ImpVar)

    ImpVar &lt;- ImpVar[order(-ImpVar$Overall), ]

    print(head(ImpVar))</code></pre>

<pre><code>##                               Overall                   Variables

## `Area Type`Cache la Poudre  123.98329  `Area Type`Cache la Poudre

## `Soil Type`Cryorthents      106.31220      `Soil Type`Cryorthents

## `Area Type`Comanche Peak     90.85926    `Area Type`Comanche Peak

## `Soil Type`Rogert            85.62203           `Soil Type`Rogert

## `Soil Type`Legault           80.30939          `Soil Type`Legault

## `Soil Type`Cathedral Family  68.68303 `Soil Type`Cathedral Family</code></pre>

<p>It looks like most important variables are area type and soil type. Next, we are going to predict the cover type.</p>

<pre class="r"><code>## Summary

summary(model)</code></pre>

<pre><code>## Warning in sqrt(diag(vc)): NaNs produced</code></pre>

<pre><code>## Call:

## multinom(formula = Cover_Type ~ . - Id, data = training, maxit = 500, 

##     trace = TRUE)

## 

## Coefficients:

##   (Intercept)    Elevation        Aspect       Slope

## 2    17.96795 -0.009267891 -0.0009280679  0.04934965

## 3    54.94883 -0.034534363  0.0047149424  0.10371677

## 4    58.11421 -0.040739166  0.0033257121  0.05002970

## 5    21.99264 -0.015917967  0.0033975444  0.03729762

## 6    41.69034 -0.034394115  0.0043080497  0.18710687

## 7   -50.58038  0.021716394  0.0012262861 -0.04535657

##   Horizontal_Distance_To_Hydrology Vertical_Distance_To_Hydrology

## 2                     0.0018488629                    0.002519815

## 3                     0.0023276145                    0.017542791

## 4                    -0.0021060601                    0.020917243

## 5                     0.0001315519                    0.007611954

## 6                     0.0010182316                    0.014593018

## 7                    -0.0015337807                   -0.003121530

##   Horizontal_Distance_To_Roadways Hillshade_9am Hillshade_Noon

## 2                    0.0001656282    0.02210984     0.01647428

## 3                    0.0002150746    0.05637909    -0.01374225

## 4                    0.0013118928    0.06909485     0.02379915

## 5                   -0.0001240735    0.01108580     0.03968154

## 6                    0.0004175309    0.22466770    -0.18425130

## 7                   -0.0001584391   -0.02719897     0.03264086

##   Hillshade_3pm Horizontal_Distance_To_Fire_Points

## 2    0.01327414                      -6.047169e-05

## 3    0.02162102                      -2.333113e-04

## 4    0.00425120                       7.769012e-04

## 5   -0.01544076                      -3.994147e-04

## 6    0.17240973                       2.618697e-04

## 7   -0.04204805                       3.293629e-04

##   `Area Type`Cache la Poudre `Area Type`Comanche Peak `Area Type`Neota

## 2                  12.752240                 1.487317         2.548465

## 3                  31.996956                24.748305         6.873822

## 4                  42.183622                13.984644         5.555098

## 5                   7.876062                13.942617       -13.303493

## 6                  27.958802                20.796787         8.408596

## 7                   1.215610               -15.899589       -18.468079

##   `Area Type`Rawah `Soil Type`Bross `Soil Type`Bullwark

## 2         1.179931        16.570486           -2.956253

## 3        -8.670252         9.806677            1.006264

## 4        -3.609159        12.011968           -6.828321

## 5        13.477459         1.247440            1.203912

## 6       -15.473845         6.684421           -2.608135

## 7       -17.428318        18.917700          -33.910497

##   `Soil Type`Catamount `Soil Type`Cathedral Family `Soil Type`Como

## 2           -2.7299143                  -18.615288       -2.693514

## 3            0.2218033                   11.517308       16.209908

## 4            1.9920455                    3.186405       12.421945

## 5            2.0697812                   14.509522        1.766912

## 6           -1.4832142                    8.193467       17.751219

## 7           -0.8048370                  -12.661044        2.494253

##   `Soil Type`Cryaquolis `Soil Type`Cryorthents `Soil Type`Cryumbrepts

## 2            -3.7716241              19.661836             -0.4966736

## 3             7.6225424               4.305804              1.9709545

## 4             2.7175488              11.945808            -17.6194263

## 5             0.0274751              23.553148            -14.2651212

## 6             6.2515686              22.727608             -0.1954174

## 7           -14.9031342              24.117998              4.4228910

##   `Soil Type`Gateview `Soil Type`Granile `Soil Type`Haploborolis

## 2           -4.274768          -3.700136               12.673314

## 3            1.500140          -8.243374               14.725174

## 4           -2.984555           4.143781                8.546888

## 5            2.792331           1.986118                5.389164

## 6           -1.550134         -13.530241                9.883506

## 7          -25.636473         -26.541196                9.480713

##   `Soil Type`Legault `Soil Type`Leighcan `Soil Type`Moran

## 2          -2.222748           -3.407163        -4.269705

## 3          21.007482          -13.167898        -6.099956

## 4           9.325207            2.662065        17.122208

## 5         -21.876326            0.985027       -14.583308

## 6           5.120588           -3.053848       -19.817503

## 7         -20.757038            1.465127         2.456972

##   `Soil Type`Pachic `Soil Type`Ratake family `Soil Type`Rogert

## 2        -16.111736                -4.377128        26.6894745

## 3          6.772313                 2.488352         4.2931573

## 4          2.830475                -5.152385        10.7992047

## 5        -11.749851                 1.303215        31.0853345

## 6          5.376535                -2.437024        11.9075239

## 7         -1.641537                 9.842900         0.8473373

##   `Soil Type`Supervisor `Soil Type`Troutville `Soil Type`Typic

## 2            12.1087376             -4.583514       -3.4357830

## 3            -0.3023879            -20.561057      -14.0484059

## 4            -0.6229832             -4.347162       -9.9821681

## 5             0.1340054            -17.032237       -0.4512488

## 6             0.1626080            -13.911397       -3.8241176

## 7            -2.7594309              4.594197       -0.1615392

##   `Soil Type`Vanet

## 2         7.910053

## 3        13.924030

## 4         5.945658

## 5        13.897351

## 6        10.042324

## 7        10.556263

## 

## Std. Errors:

##    (Intercept)    Elevation       Aspect       Slope

## 2 0.0001190579 0.0003269942 0.0005519483 0.007828952

## 3 0.0003022136 0.0008096755 0.0010718325 0.015623225

## 4 0.0002420696 0.0009309112 0.0012397794 0.017074452

## 5 0.0001205829 0.0004684541 0.0007505506 0.010059731

## 6 0.0002584910 0.0007864661 0.0010165992 0.015501040

## 7 0.0001903737 0.0006238051 0.0007707437 0.013055308

##   Horizontal_Distance_To_Hydrology Vertical_Distance_To_Hydrology

## 2                     0.0002756044                    0.001073937

## 3                     0.0006462248                    0.002059097

## 4                     0.0008596876                    0.002515787

## 5                     0.0003917140                    0.001323730

## 6                     0.0006162093                    0.002002143

## 7                     0.0003397920                    0.001423654

##   Horizontal_Distance_To_Roadways Hillshade_9am Hillshade_Noon

## 2                    3.171641e-05   0.006766888    0.007874563

## 3                    1.189779e-04   0.013970194    0.014252606

## 4                    1.792539e-04   0.015758111    0.016321221

## 5                    5.013604e-05   0.008896388    0.009574562

## 6                    1.135652e-04   0.014301650    0.014839318

## 7                    4.810024e-05   0.013664584    0.013293983

##   Hillshade_3pm Horizontal_Distance_To_Fire_Points

## 2   0.006050353                       3.775757e-05

## 3   0.011784749                       1.288662e-04

## 4   0.013256520                       1.787635e-04

## 5   0.007630078                       5.351685e-05

## 6   0.012112392                       1.201295e-04

## 7   0.011750132                       5.905930e-05

##   `Area Type`Cache la Poudre `Area Type`Comanche Peak `Area Type`Neota

## 2               2.873672e-03             1.985672e-02     1.314408e-03

## 3               6.265365e-03             6.050134e-03     3.078896e-12

## 4               2.421011e-04             3.654093e-08     4.956850e-15

## 5               3.578878e-09             1.479028e-02     2.930129e-16

## 6               5.920947e-03             5.764256e-03     2.118203e-07

## 7               1.293072e-09             1.125025e-03     6.563903e-04

##   `Area Type`Rawah `Soil Type`Bross `Soil Type`Bullwark

## 2     1.947580e-02     7.634499e-05        4.277417e-03

## 3     5.015483e-09     1.598298e-14        2.749655e-02

## 4     1.526346e-09     6.118826e-21        4.239788e-04

## 5     1.479211e-02     1.348993e-14        3.503332e-03

## 6     7.458388e-08     1.506172e-14        2.972760e-02

## 7     6.764175e-04     7.634499e-05        2.144024e-22

##   `Soil Type`Catamount `Soil Type`Cathedral Family `Soil Type`Como

## 2         6.593346e-03                         NaN    1.846568e-02

## 3         5.206236e-04                2.964050e-04    3.428254e-10

## 4         2.231577e-09                3.250657e-04    1.523415e-09

## 5         5.576085e-03                1.647071e-09    1.390724e-02

## 6         1.323036e-03                4.283143e-04    1.423682e-07

## 7         4.939296e-04                1.141771e-21    6.638304e-04

##   `Soil Type`Cryaquolis `Soil Type`Cryorthents `Soil Type`Cryumbrepts

## 2          4.917464e-05           7.949295e-05           4.431061e-05

## 3          4.814784e-05           6.154133e-14           1.498682e-10

## 4          1.486594e-04           2.373352e-17           1.479757e-26

## 5          6.898463e-05           3.507245e-05           1.364190e-13

## 6          1.112291e-04           1.162656e-04           7.748925e-10

## 7          6.709076e-15           1.499981e-05           4.102099e-05

##   `Soil Type`Gateview `Soil Type`Granile `Soil Type`Haploborolis

## 2        1.897295e-04       4.210923e-04            2.447789e-04

## 3        3.615541e-04       3.169478e-09            1.158679e-03

## 4        4.306696e-04       2.192744e-09            9.942037e-04

## 5        7.248439e-04       4.270830e-04            1.605216e-08

## 6        1.123106e-03       1.177989e-09            1.158937e-04

## 7        8.093828e-20       8.962635e-18            5.520711e-14

##   `Soil Type`Legault `Soil Type`Leighcan `Soil Type`Moran

## 2       3.908679e-04        7.548778e-03     1.318330e-04

## 3       4.830530e-09        4.469029e-09     1.094028e-10

## 4       3.445342e-12        8.048489e-09     2.437186e-08

## 5       4.536657e-13        4.482732e-03     1.639304e-11

## 6       2.722635e-15        1.117114e-03     8.023489e-15

## 7       1.809843e-16        4.455130e-04     2.707655e-04

##   `Soil Type`Pachic `Soil Type`Ratake family `Soil Type`Rogert

## 2               NaN             7.842471e-04      9.977140e-05

## 3      2.107758e-05             2.660022e-03               NaN

## 4      2.973403e-04             8.929986e-05      4.161567e-22

## 5      4.168521e-14             2.051097e-03      9.977139e-05

## 6      3.009689e-04             5.702158e-04      2.673328e-23

## 7      7.851481e-19             3.795381e-04      1.916111e-24

##   `Soil Type`Supervisor `Soil Type`Troutville `Soil Type`Typic

## 2          3.208851e-12          8.885740e-05     3.627827e-04

## 3          1.239405e-24                   NaN     4.700721e-10

## 4          1.097109e-25          1.375937e-18     1.044367e-13

## 5          8.835130e-14          1.901147e-10     6.461166e-05

## 6          2.011165e-27          9.490302e-23     1.735698e-04

## 7          3.461224e-17          1.740135e-09     1.177484e-05

##   `Soil Type`Vanet

## 2     5.177170e-04

## 3     2.793746e-02

## 4     1.636545e-03

## 5     3.587709e-04

## 6     2.645484e-02

## 7     4.672145e-10

## 

## Residual Deviance: 14942.03 

## AIC: 15350.03</code></pre>

<pre class="r"><code>prediction &lt;- predict(model, newdata=validation)

postResample(prediction, validation$Cover_Type)</code></pre>

<pre><code>##  Accuracy     Kappa 

## 0.6940035 0.6430041</code></pre>

<p>We got an accuracy rate of 70.17% in our validation test dataset. Due to the low score, we will have to try another method which is random forest.</p>

<p>Randomforest is a type of algorithm that ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.</p>

<p>It can handle categorical and numerical variables.</p>

<pre class="r"><code>## Creating a predictive model using randomforest

names(training)[13:14] &lt;- c(&quot;Area_Type&quot;, &quot;Soil_Type&quot;)

model2 &lt;- randomForest(Cover_Type ~. -Id, data = training)



## Overview of the model

model2</code></pre>

<pre><code>## 

## Call:

##  randomForest(formula = Cover_Type ~ . - Id, data = training) 

##                Type of random forest: classification

##                      Number of trees: 500

## No. of variables tried at each split: 3

## 

##         OOB estimate of  error rate: 14.3%

## Confusion matrix:

##      1    2    3    4    5    6    7 class.error

## 1 1160  216    2    0   35    4   95  0.23280423

## 2  269 1014   38    0  126   47   18  0.32936508

## 3    0    8 1235   72   14  183    0  0.18320106

## 4    0    0   29 1465    0   18    0  0.03108466

## 5    3   35   20    0 1436   18    0  0.05026455

## 6    1    9  148   37   13 1304    0  0.13756614

## 7   52    3    0    0    1    0 1456  0.03703704</code></pre>

<pre class="r"><code>## Measuring the variance importance

    ImpVar &lt;- varImp(model2)

    ImpVar$Variables &lt;- row.names(ImpVar)

    ImpVar &lt;- ImpVar[order(-ImpVar$Overall), ]

    print(head(ImpVar))</code></pre>

<pre><code>##                                      Overall

## Elevation                          2493.6424

## Soil_Type                          1269.2659

## Horizontal_Distance_To_Roadways     853.0528

## Horizontal_Distance_To_Fire_Points  697.9085

## Area_Type                           584.9208

## Horizontal_Distance_To_Hydrology    581.9755

##                                                             Variables

## Elevation                                                   Elevation

## Soil_Type                                                   Soil_Type

## Horizontal_Distance_To_Roadways       Horizontal_Distance_To_Roadways

## Horizontal_Distance_To_Fire_Points Horizontal_Distance_To_Fire_Points

## Area_Type                                                   Area_Type

## Horizontal_Distance_To_Hydrology     Horizontal_Distance_To_Hydrology</code></pre>

<p>Next, we will test the accuracy of this model by applying it to the validation test.</p>

<pre class="r"><code>## Applying the model to the validation set

names(validation)[13:14] &lt;- c(&quot;Area_Type&quot;, &quot;Soil_Type&quot;)

prediction2 &lt;- predict(model2, newdata = validation)



## Getting the accuracy

postResample(prediction2, validation$Cover_Type)</code></pre>

<pre><code>##  Accuracy     Kappa 

## 0.8648589 0.8423354</code></pre>

<p>Using the randomforest, we have generated 85.19% accuracy which is a big improvement from our previous model. We will use this model to try to predict the test data we have.</p>

</div>

<div id="model-application-on-the-test-dataset" class="section level2">

<h2>Model Application on the Test Dataset</h2>

<p>Now, we are going to use the model itself to predict the test set and submit it to Kaggle.</p>

<pre class="r"><code>## Applying the random forest model to test set 

## names(test)[c(1, 12:13)] &lt;- c(&quot;Id&quot;,&quot;Area_Type&quot;, &quot;Soil_Type&quot;)

## test_prediction &lt;- predict(model2, newdata = test)



## Storing in a dataframe the result

## submission &lt;- data.frame(Id = test$Id, Cover_Type = test_prediction)



## library(xlsx)

## write.csv(submission, file = &quot;final_submission.xlsx&quot;)</code></pre>

</div>

<div id="conclusion" class="section level2">

<h2>Conclusion</h2>

<p>Initially, when I uploaded my submission using the multinomial logistic regression model, the score I got was just 34.202%. This means that the model fails to predict the test set and calls for improvement or revision of the model. Now, when I uploaded my submission using the random forest model, the score improved drastically to 67.715%.</p>

<p>In this challenge, I have learned the application and evaluation of different machine learning methods. According to the variable importance function, elevation, soil type, and horizontal distance to roadways are the top 3 most important variables in the model. The model might be overfitting the data because the score generated was 67.715% while the accuracy of the model based on the validation set was 85.19%. However, the low score may also tell that there are other variables that can predict the forest type more accurately.</p>

</div>









</div>

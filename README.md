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

<p><img src="data:image/png;base64,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jrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoUWjrQCoADq2FRgSwBNfXr6lmKFFo60AqAA6thUYEsATX16+pZihRaOtAKgAOrYVGBLAE19evqWYoXCjDdfuDKbzb744tuKz/XnxqydRy0XB1AAdWwqsCWApj49fUuxYrQYb36hIe7SN0Z/8kAAdEJFwJ4AtL+pwJYAmvr09C3FirFg3NoVcgA6oSJgTwDa31RgSwBNfXr6lmLFSC9KPz/3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" width="672" /></p>

<pre class="r"><code>## Elevation

h &lt;- ggplot(data = train, aes(x = Elevation, fill = Cover_Type))

h + geom_bar() + facet_grid(.~Cover_Type)</code></pre>

<p><img src="data:image/png;base64,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hQ0RgApaBqCuKgAN5gpAawKgXa6tZQAKQD3TA6B9AqANhQ0RgApaBqCuKgAN5gpAawKgXa6tZQAKQD3TA6B9AqANhQ0RgApaBqCuKgAN5gpAawKgXa6tZQAKQD3TA6B9AqANhQ0RgApaBqCuKgAN5gpAawKgXa6tZQAKQD3TA6B9AqANhQ0RgApaBqCuKgAN5poeQHePTha64wUv1wACoF2urWUACkA90wOgfQKgDYUNEYAKWgagrioADeaaHkAXurKz5ykv0xACoF2urWUACkA90wOgfQoA0OxA9H4vzyACoF2urWUACkA90wOgfQoA0DOTu17z8gwiSwCdc3OrAM2huXmAHsr+1MuHVqpC1+EAvfvuuyt8GgXovn2CleMA9DgAdWs4QG8cnjziZRlGALTLtVEGoADUN73zALRfwwF6ZrLXyzGQAGiXa6MMQAGob3rnAWi/BgP0xuEtXEGaAdBu10YZgAJQ3/TOA9B+DQbopa08AwpAu10bZQAKQH3TOw9A+zUUoLtHt/IMKADtdm2UASgA9U3vPADt11CAXtnZ/D30hQBol2ujDEABqG965wFov4YC9NJ2LiEB0G7XRhmAAlDf9M4D0H4NBejJbdxEnwuAdrk2ygAUgPqmdx6A9msgQHePbucaPADtdm2UASgA9U3vPADt10CA3ji8padAAWina6MMQAGob3rnAWi/BgL0ys52bmICoN2ujTIABaC+6Z0HoP0aCNBt3QUKQLtdG2UACkB90zsPQPvFO9I3FDZEAForA1AACkBTEADtcm2UASgA9U3vPADtFwBtKGyIALRWBqAAFICmIADa5dooA1AA6pveeQDaLwDaUNgQAWitDEABKABNQQC0y7VRBqAA1De98wC0XwC0obAhAtBaGYACUACaggBol2ujDEABqG965wFovwBoQ2FDBKC1MgAFoAA0BQHQLtdGGYACUN/0zgPQfgHQhsKGCEBrZQAKQMcFUA95bTi2AGiXa6MMQAGob3rnAeiIBUC7XBtlAApAfdM7D0BHLADa5dooA1AA6pveeQA6YgHQLtdGGYACUN/0zgPQEWswQG8emxZ6x0trhbAhAtBaGYACUACaggYD9NoRANpcGYDKXAGo0BWAJqrBAL08fai9EDZEAForA1AACkBT0GCAnpo+3l4IGyIArZUBKAAFoCloKEBvnTjwbHslbIgAtFYGoAAUgKagoQC9eezgD35gOv36FxuVsCEC0FoZgAJQAJqChgJ0cQ1p+vTikbeVGrjhNWXUnF1YfHYh7LY9dDr/m7GzAOjGXA9lf9Yf2bjunmUALT6UfyypBGj+cRv2x4/P/45BF7P/Cm27kTQ0FKCXp9PHzs3+7zPT5Zl8FIAW1JxDdGsAPb34O9cKQCOi9ND8nzkwDx06tPrwZrUA6FyWALqvBtBtEPR4qfzT2eIfo7o4W76tx7ZbSUFDAXq2vAjfvJYU9DSiOG+fn8Zv7RT+9OLvAqC1SqSTwEOrp/A5QMuHW1veyCn8XJZO4ffVTuH39a8c4RS+BGj2abHs7ZrSKfwSoJzCh3sl0uXpwXP1R4KGCECLBQCqcQWgHZsFoMMUCqDXjqzfSR80RABaLABQjSsA7dgsAB2mcADlCFS67ZnHjgKgw9MDoB2bBaDDNBCgt06Ul9+bL0gKGiIALRYAqMYVgHZsFoAO09Aj0FNzcC5BWiloiAC0WACgGlcA2rFZADpMAe4Dfezc7PoHGteQAKhz2z47CoAOTw+AdmwWgA7T4OdAz5ZvxtR4KVLQEAFosQBANa4AtGOzAHSYhl9Euv5t0+mBx9aPPwGoe9s+OwqADk8PgHZsFoAOk5F3pAegxQIA1bgC0I7NAtBhAqAdrnUBUADa6AmACnsaqwBoh2tdABSANnoCoMKexioA2uFaFwAFoI2eAKiwp7EKgHa41gVAAWijJwAq7GmsAqAdrnUBUADa6AmACnsaqwBoh2tdABSANnoCoMKexioA2uFaFwAFoI2eAKiwp7EKgHa41gVAAWijJwAq7GmsAqAdrnUBUADa6AmACnsaqwBoh2tdABSANnoCoMKexioA2uFaFwAFoI2eAKiwp7EKgHa41gVAAWijJwAq7GmsAqAdrnUBUADa6AmACnsaqwBoh2tdABSANnoCoMKexioA2uFaFwAFoI2eAKiwp7EKgHa41gVAAWijJwAq7GmsAqAdrnUBUADa6AmACnsaqwBoh2tdABSANnoCoMKexioA2uFaFwAFoI2eAKiwp7EKgHa41gVAAWijJwAq7GmsAqAdrnUBUADa6AmACnsaqwBoh2tdABSANnoCoMKexioA2uFaFwAFoI2eAKiwp7EKgHa41gVAAWijJwAq7GmsSh6gGSwvXFgB6IU5QC+IVg4M0NMNgJ5ePhpuBA+tLh/KlzNwzhfmAD10aPMALZg5/1sCtMDpUNfNAHSp+ae9KwcB6PFqxeNLgB4/n/8xDNCLALQuAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexioA6moKgK6uCEDbewKgwp7GKgDqagqArq4IQNt7AqDCnsYqAOpqCoCurghA23sCoMKexqq0AXqhBOiFNYAWDBVtu6MnyY/O6ZUHSlSeXur8CkBPBxvBQxVAD80BemjrAL3bJkD3Ff81ALpvQwBdkrIG0OPzv/PNHvf8jmRNyb8ZT9eLAHRdANTVFACtagDU1RMAFfY0VgFQV1MAtKoBUFdPAFTY01gFQF1NAdCqBkBdPQFQYU9jFQB1NQVAqxoAdfUEQIU9jVUA1NUUAK1qANTVEwAV9jRWAVBXUwC0qgFQV08AVNjTWAVAXU0B0KoGQF09AVBhT2MVAHU1BUCrGgB19QRAhT2NVQDU1RQArWoA1NUTABX2NFYBUFdTALSqAVBXTwBU2NNYBUBdTQHQqgZAXT0BUGFPYxUAdTUF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" width="672" /></p>

</div>

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

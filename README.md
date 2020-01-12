# Forest-Type-Prediction
This is my first Kaggle competition. It is about predicting the different forest type given with various geographic features of Colorado. 

## Introduction

There are different forest types around the world and their differences usually involve with ecological factors, geographical locations, habitants, and weather. It is quite interesting how unique each forest type is. In this competition, we are given data relating to the forest type and the objective is to predict the forest type as accurately as possible. By diving with my knowledge in machine learning and performing two model testing, I have decided to use random forest since it generated a higher accuracy rate than the multinomial logistic regression. In my submission at Kaggle, I got a higher score using random forest than the multinomial logistic regression. 

## Objective

The objective of this challenge is to predict the forest type based on various categorial and quantitative variables. 

## Background

Since the objective is to create a model that will predict the cover type, it is best to describe the qualitative variable, forest cover and wilderness area. 

### Cover Type

#### Spruce/Fir

  Their spatial patterns are dependent on the temperature. Because of the thicker trunks, they are able to survive fires. 

#### Lodgepole Pine

  It is commonly located near the ocean shore and in dry montane forests to the subalpine. It is an evergreen pine.

#### Ponderosa Pine

  Pinus ponderosa is a large coniferous pine (evergreen) tree. The bark helps to distinguish it from other species. Like most western pines, the ponderosa generally is associated with mountainous topography. <i> From Wikipedia </i>

#### Cottonwood/Willow

  Poplars of the cottonwood section are often wetlands or riparian trees. The aspens are among the most important boreal broadleaf trees. 

#### Aspen

  Aspen trees are all native to cold regions with cool summers, in the north of the Northern Hemisphere, extending south at high-altitude areas such as mountains or high plains. Aspens typically grow in environments that are otherwise dominated by coniferous tree species, and which are often lacking other large deciduous tree species.

#### Douglas-fir

  It is a coniferous type whose leaves are evergreen (i.e. always green in colors). It prefers neutral or acidic soils. This type is used widely for timber due to its strength, hardness, and durability. 

#### Krummholz

  It is a type of stunted, deformed vegetation encountered in the subarctic and subalpine tree line landscapes, shaped by continual exposure to fierce, freezing winds. Under these conditions, trees can only survive where they are sheltered by rock formations or snow cover. As the lower portion of these trees continues to grow, the coverage becomes extremely dense near the ground.

### Wilderness Type
Next, we will discuss the four wilderness type. 

#### Rawah Wilderness
  - protects a scenic high country of U-shaped glacier-carved valleys and peaks reaching 12, 951 feet. 
  - To the south and west of the area lies an almost roadless Colorado State Forest, an unofficial extension of the Wilderness. 
  - Melting snow fills 26 lakes within the area. 
  - On the upper forested slopes of the mountains, especially in the southern section, clusters of old-growth spruce and fir abound. 
  - has a total of 74,408 acres. 
  - located on the Canyon Lakes Ranger District of the Roosevelt National Forest in Colorado near the Wyoming border and also in the Routt National Forest to its south. 
  - There are 85 miles (137 km) of trails in the area and elevation ranges from 8,400 feet (2,600 m) to 13,0000 feet (4,000 m). 
  - The temperature in the Rawah Wilderness ranges from a low of 5 degrees Fahrenheit during the winder and a high of 77 degrees Fahrenheit during the summer. 


#### Neota Wilderness
  - located on the Canyon Lakes Ranger District of the Roosevelt National Forest in Colorado. 
  - This wilderness area encompasses 9,924 acres (40 km2) and is bordered on the south by Rocky Mountain National Park. 
    - Elevation ranges frm 10,000 ft (3,000 m) to 11,9896 ft (3,626 m) in the Rocky Mountains. There are only .5 miles (2.4 km) of trail in this area. 
    - bordered by the Rawah Wilderness to the northwest, the COmanche Peak Wilderness to the east, and the Rocky Mountain National Park Wilderness to the south. 
    
    
#### Comanche Peak Wilderness
  - located in the Roosevelt National Forest on the Canyon Lakes Ranger District in Colorado along the northern boundary of Rocky Mountain National Park. 
  - comprises of 66,791-acre (27, 029 ha). 
  - There are 121 miles (195 km) of hiking trails inside the wilderness. 
  - There are also 7 named peaks, 6 named lakes (inclduing COmanche Reservoir) and 16 named rivers and creeks within the wilderness boundaries. 
  - From the trails, the Wilderness appears to be nothing but trees until the forst gives way to abundant alpine tundra along the souther boundary near Comanche Peak. 
  - Beyond Comanche to the South lies Rocky Mountain National Park. 
  - The Cache la Poudre River tumbles over cascades or runs quietly in pools for over 10 miles to the north and west.
  
  
#### Cache La Poudre Wilderness
  - located on the Canyon Lakes Ranger District on the Roosevelt National Forest in Colorado. 
  - This wilderness covers 9,258 acres (37.47 km2) and is characterized by steep, rugged terrain along the Cache la Poudre River. 
  - Elevations in this area varies from 6,200 feet(1,900 m) to 8,600 feet (2,600 m). 

## Data Loading and Initial Analysis

First, the researcher will load the libraries to be used in the analysis, the data itself, and do the preliminary analysis. This analysis will involve looking at the data type and number of null values. 

The data is composed of train set and test set. The train set comprises of 15,120 observations while the test set comprises of 365,892 observations. 

```{r, data loading}
## Loading the libraries
library(ggplot2)
library(tidyverse)
library(dplyr)
library(caret)
library(randomForest)
library(reshape2)


## Loading the datasets
    
    train <- read.csv("train.csv", header = TRUE)
    
    ## Due to the limited computing power of my computer, I have divided the test data set into six. These data sets are already processed.
    test1 <- read.csv("test1.csv", header = TRUE)
    test2 <- read.csv("test2.csv", header = TRUE)
    test3 <- read.csv("test3.csv", header = TRUE)
    test4 <- read.csv("test4.csv", header = TRUE)
    test5 <- read.csv("test5.csv", header = TRUE)
    test6 <- read.csv("test6.csv", header = TRUE)
    
    ## Combining all of the datasets into one
    test <- rbind(test1, test2, test3, test4, test5, test6)
    
    ## Renaming the Area Type and Soil Type columns of the test dataset
    names(test)[12:13] <- c("Area_Type", "Soil_Type")

## Number of observations
data.frame(data = c("train", "test"), columns = c(ncol(train), ncol(test)), rows = c(nrow(train), nrow(test)))

## Number of null values in train set
colSums(is.na(train))

## Number of null values in test set 
colSums(is.na(test))
```

It can be seen that there are no null values in our dataset.

## Data Cleaning

Second, the researcher will clean the data. The following will be performed during the cleaning: 

1. Conversion of categorical variables into factor class. 
2. Melting the Wilderness Area variables.
3. Melting the Soil Type variables. 
4. Removing observations where the wilderness area and soil type are zero.
5. Converting the Elevation, Aspect, Slope, Horizontal Distance to Fire Points and Hydrology, Vertical Distance to Roadways and Hydrology, and Hillshade into numeric class.

In addition to that, I have replaced grouped the soil type with similar features. 

```{r data test set cleaning}
## Replacing the numeric values into text values
    
    train$Wilderness_Area1 <- ifelse(train$Wilderness_Area1 == 1, "Rawah", train$Wilderness_Area1)
    train$Wilderness_Area2 <- ifelse(train$Wilderness_Area2 == 1, "Neota", train$Wilderness_Area2)
    train$Wilderness_Area3 <- ifelse(train$Wilderness_Area3 == 1, "Comanche Peak", train$Wilderness_Area3)
    train$Wilderness_Area4 <- ifelse(train$Wilderness_Area4 == 1, "Cache la Poudre", train$Wilderness_Area4)
  
    train$Soil_Type1 <- ifelse(train$Soil_Type1 == 1, "Cathedral Family", train$Soil_Type1)
    train$Soil_Type2 <- ifelse(train$Soil_Type2 == 1, "Vanet", train$Soil_Type2)
    train$Soil_Type3 <- ifelse(train$Soil_Type3 == 1, "Haploborolis", train$Soil_Type3)
    train$Soil_Type4 <- ifelse(train$Soil_Type4 == 1, "Ratake family", train$Soil_Type4)
    train$Soil_Type5 <- ifelse(train$Soil_Type5 == 1, "Vanet", train$Soil_Type5)
    train$Soil_Type6 <- ifelse(train$Soil_Type6 == 1, "Vanet", train$Soil_Type6)
    train$Soil_Type7 <- ifelse(train$Soil_Type7 == 1, "Gothic", train$Soil_Type7)
    train$Soil_Type8 <- ifelse(train$Soil_Type8 == 1, "Supervisor", train$Soil_Type8)
    train$Soil_Type9 <- ifelse(train$Soil_Type9 == 1, "Troutville", train$Soil_Type9)
    train$Soil_Type10 <- ifelse(train$Soil_Type10 == 1, "Bullwark", train$Soil_Type10)
    train$Soil_Type11 <- ifelse(train$Soil_Type11 == 1, "Bullwark", train$Soil_Type11)
    train$Soil_Type12 <- ifelse(train$Soil_Type12 == 1, "Legault", train$Soil_Type12)
    train$Soil_Type13 <- ifelse(train$Soil_Type13 == 1, "Catamount", train$Soil_Type13)
    train$Soil_Type14 <- ifelse(train$Soil_Type14 == 1, "Pachic", train$Soil_Type14)
    train$Soil_Type15 <- ifelse(train$Soil_Type15 == 1, "unspecified", train$Soil_Type15)
    train$Soil_Type16 <- ifelse(train$Soil_Type16 == 1, "Cryaquolis", train$Soil_Type16)
    train$Soil_Type17 <- ifelse(train$Soil_Type17 == 1, "Gateview", train$Soil_Type17)
    train$Soil_Type18 <- ifelse(train$Soil_Type18 == 1, "Rogert", train$Soil_Type18)
    train$Soil_Type19 <- ifelse(train$Soil_Type19 == 1, "Typic", train$Soil_Type19)
    train$Soil_Type20 <- ifelse(train$Soil_Type20 == 1, "Typic", train$Soil_Type20)
    train$Soil_Type21 <- ifelse(train$Soil_Type21 == 1, "Typic", train$Soil_Type21)
    train$Soil_Type22 <- ifelse(train$Soil_Type22 == 1, "Leighcan", train$Soil_Type22)
    train$Soil_Type23 <- ifelse(train$Soil_Type23 == 1, "Leighcan", train$Soil_Type23)
    train$Soil_Type24 <- ifelse(train$Soil_Type24 == 1, "Leighcan", train$Soil_Type24)
    train$Soil_Type25 <- ifelse(train$Soil_Type25 == 1, "Leighcan", train$Soil_Type25)
    train$Soil_Type26 <- ifelse(train$Soil_Type26 == 1, "Granile", train$Soil_Type26)
    train$Soil_Type27 <- ifelse(train$Soil_Type27 == 1, "Leighcan", train$Soil_Type27)
    train$Soil_Type28 <- ifelse(train$Soil_Type28 == 1, "Leighcan", train$Soil_Type28)
    train$Soil_Type29 <- ifelse(train$Soil_Type29 == 1, "Como", train$Soil_Type29)
    train$Soil_Type30 <- ifelse(train$Soil_Type30 == 1, "Como", train$Soil_Type30)
    train$Soil_Type31 <- ifelse(train$Soil_Type31 == 1, "Leighcan", train$Soil_Type31)
    train$Soil_Type32 <- ifelse(train$Soil_Type32 == 1, "Catamount", train$Soil_Type32)
    train$Soil_Type33 <- ifelse(train$Soil_Type33 == 1, "Leighcan", train$Soil_Type33)
    train$Soil_Type34 <- ifelse(train$Soil_Type34 == 1, "Cryorthents", train$Soil_Type34)
    train$Soil_Type35 <- ifelse(train$Soil_Type35 == 1, "Cryumbrepts", train$Soil_Type35)
    train$Soil_Type36 <- ifelse(train$Soil_Type36 == 1, "Bross", train$Soil_Type36)
    train$Soil_Type37 <- ifelse(train$Soil_Type37 == 1, "Cryumbrepts", train$Soil_Type37)
    train$Soil_Type38 <- ifelse(train$Soil_Type38 == 1, "Leighcan", train$Soil_Type38)
    train$Soil_Type39 <- ifelse(train$Soil_Type39 == 1, "Moran", train$Soil_Type39)
    train$Soil_Type40 <- ifelse(train$Soil_Type40 == 1, "Moran", train$Soil_Type40)
    
## Melting the dataset
    
    train <- train %>% 
                gather(key = "Wilderness Area", value = "Area_Type", 12:15) %>%
                gather(key = "Soil", value = "Soil_Type", Soil_Type1:Soil_Type40)

## Converting the Cover Type, Area Type, and Soil Type into factors
    train$Cover_Type <- as.factor(train$Cover_Type)
    train$Area_Type <- as.factor(train$Area_Type)
    train$Soil_Type <- as.factor(train$Soil_Type)
        
## Removing observations where the wilderness area and soil type are zero. 
    
    train <- train[!(train$Area_Type %in% c(0)), ]
    train <- train[!(train$Soil_Type %in% c(0)), ]
    train <- select(train, -`Wilderness Area`, -Soil)

    
## Converting other variables into numeric type
    train$Elevation <- as.numeric(train$Elevation)
    train$Aspect <- as.numeric(train$Aspect)
    train$Slope <- as.numeric(train$Slope)
    train$Horizontal_Distance_To_Fire_Points <- as.numeric(train$Horizontal_Distance_To_Fire_Points)
    train$Vertical_Distance_To_Hydrology <- as.numeric(train$Vertical_Distance_To_Hydrology)
    train$Horizontal_Distance_To_Hydrology <- as.numeric(train$Horizontal_Distance_To_Hydrology)
    train$Horizontal_Distance_To_Roadways <- as.numeric(train$Horizontal_Distance_To_Roadways)
    train$Hillshade_3pm <- as.numeric(train$Hillshade_3pm)
    train$Hillshade_9am <- as.numeric(train$Hillshade_9am)
    train$Hillshade_Noon <- as.numeric(train$Hillshade_Noon)
    

```

## Data Exploration

Next, we are going to use ggplot in generating graphs. 

Presented below is the descriptive statistics. 
```{r descriptive stats}
## Descriptive Statistics
summary(train)
```

The variables that describe the most the forest type are Area Type, Elevation, and Soil Type. With that, we are going to check area type. 

Remember the following integer represents each cover type: 
1 - Spruce/Fir
2 - Lodgepole Pine
3 - Ponderosa Pine
4 - Cottonwood/Willow
5 - Aspen
6 - Douglas-fir
7 - Krummholz

```{r }
## Aspect  
g <- ggplot(data = train, aes(x = Aspect, fill = Cover_Type))
g + geom_bar() + facet_grid(.~Cover_Type)

## Elevation
h <- ggplot(data = train, aes(x = Elevation, fill = Cover_Type))
h + geom_bar() + facet_grid(.~Cover_Type)
```


## Machine Learning Modelling

In our dataset, we are going to use multinomial logistic regression. Multinomial logistic regression allows for more than two categories of the dependent variable. In this dataset, we have used the nnet library where neural networks are going to be used to work hand in hand in performing multinomial logistic regression. 

Diagnostic tests to be performed will only be Variable Importance

```{r data processing}
## Dividing the train set into two: training and validation datasets
    inTrain <- createDataPartition(y = train$Cover_Type, p = 0.70, list = FALSE)
    training <- train[inTrain, ]
    validation <- train[-inTrain, ]

## Creating a predictive model
    library(nnet)
    model <- multinom(Cover_Type ~. -Id, data = training, maxit=500, trace = TRUE)
    
## Measuring the variance importance
    ImpVar <- varImp(model)
    ImpVar$Variables <- row.names(ImpVar)
    ImpVar <- ImpVar[order(-ImpVar$Overall), ]
    print(head(ImpVar))
```

It looks like most important variables are area type and soil type. Next, we are going to predict the cover type. 

```{r summary model}
## Summary
summary(model)
```

```{r validation testing}
prediction <- predict(model, newdata=validation)
postResample(prediction, validation$Cover_Type)
```

We got an accuracy rate of 70.17% in our validation test dataset. Due to the low score, we will have to try another method which is random forest. 

Randomforest is a type of algorithm that ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

It can handle categorical and numerical variables. 

```{r randomforest}
## Creating a predictive model using randomforest
names(training)[13:14] <- c("Area_Type", "Soil_Type")
model2 <- randomForest(Cover_Type ~. -Id, data = training)

## Overview of the model
model2

## Measuring the variance importance
    ImpVar <- varImp(model2)
    ImpVar$Variables <- row.names(ImpVar)
    ImpVar <- ImpVar[order(-ImpVar$Overall), ]
    print(head(ImpVar))
```

Next, we will test the accuracy of this model by applying it to the validation test. 
```{r }
## Applying the model to the validation set
names(validation)[13:14] <- c("Area_Type", "Soil_Type")
prediction2 <- predict(model2, newdata = validation)

## Getting the accuracy
postResample(prediction2, validation$Cover_Type)
```

Using the randomforest, we have generated 85.19% accuracy which is a big improvement from our previous model. We will use this model to try to predict the test data we have. 


## Model Application on the Test Dataset
  Now, we are going to use the model itself to predict the test set and submit it to Kaggle. 
    
```{r }
## Applying the random forest model to test set 
## names(test)[c(1, 12:13)] <- c("Id","Area_Type", "Soil_Type")
## test_prediction <- predict(model2, newdata = test)

## Storing in a dataframe the result
## submission <- data.frame(Id = test$Id, Cover_Type = test_prediction)

## library(xlsx)
## write.csv(submission, file = "final_submission.xlsx")

```

## Conclusion

Initially, when I uploaded my submission using the multinomial logistic regression model, the score I got was just 34.202%. This means that the model fails to predict the test set and calls for improvement or revision of the model. Now, when I uploaded my submission using the random forest model, the score improved drastically to 67.715%. 

In this challenge, I have learned the application and evaluation of different machine learning methods. According to the variable importance function, elevation, soil type, and horizontal distance to roadways are the top 3 most important variables in the model. The model might be overfitting the data because the score generated was 67.715% while the accuracy of the model based on the validation set was 85.19%. However, the low score may also tell that there are other variables that can predict the forest type more accurately. 

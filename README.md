---
title: "Prediction Assignment For Determining Fitness Exercise Correctness"
output: html_document
---

### Abstract

---

In this assignment, I build a predictive model to determine whether a particular from of exercise is peformed correctly, using accelerometer data. The data set used is originally from [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

### Set Libraries

---

Preparing the library for the analysis.


```r
library(caret)
library(tree)
library(rattle)
library(rpart.plot)
library(randomForest)
set.seed(12345)
```

### Loading Data

---

The dataset from can be downloaded as follows


```r
if(!file.exists("pml-training.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile="pml-training.csv")
}
if(!file.exists("pml-testing.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile="pml-testing.csv")
}

tr.org <- read.csv("pml-training.csv", na.strings=c("", "NA", "NULL"))
te.org <- read.csv("pml-testing.csv", na.strings=c("", "NA", "NULL"))
dim(tr.org)
```

```
## [1] 19622   160
```

```r
dim(te.org)
```

```
## [1]  20 160
```

### Preprocessing the data

---


There are several approaches for reducing the number of predictors.

* Remove variables that we believe have too many NA values.


```r
tr.dena <- tr.org[, colSums(is.na(tr.org)) == 0]
dim(tr.dena)
```

```
## [1] 19622    60
```

* Remove unrelevant variables. Threre are some unrelevant that can be removed as they are unlikely to be related to dependent variable.


```r
cols.remove <- c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window')
tr.dere <- tr.dena[, -which(names(tr.dena) %in% cols.remove)]
dim(tr.dere)
```

```
## [1] 19622    53
```

* Check the variables that have extremely low variance.


```r
zero.var <- nearZeroVar(tr.dere[sapply(tr.dere, is.numeric)], saveMetrics=TRUE)
zero.var
```

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt             1.101904     6.7781062   FALSE FALSE
## pitch_belt            1.036082     9.3772296   FALSE FALSE
## yaw_belt              1.058480     9.9734991   FALSE FALSE
## total_accel_belt      1.063160     0.1477933   FALSE FALSE
## gyros_belt_x          1.058651     0.7134849   FALSE FALSE
## gyros_belt_y          1.144000     0.3516461   FALSE FALSE
## gyros_belt_z          1.066214     0.8612782   FALSE FALSE
## accel_belt_x          1.055412     0.8357966   FALSE FALSE
## accel_belt_y          1.113725     0.7287738   FALSE FALSE
## accel_belt_z          1.078767     1.5237998   FALSE FALSE
## magnet_belt_x         1.090141     1.6664968   FALSE FALSE
## magnet_belt_y         1.099688     1.5187035   FALSE FALSE
## magnet_belt_z         1.006369     2.3290184   FALSE FALSE
## roll_arm             52.338462    13.5256345   FALSE FALSE
## pitch_arm            87.256410    15.7323412   FALSE FALSE
## yaw_arm              33.029126    14.6570176   FALSE FALSE
## total_accel_arm       1.024526     0.3363572   FALSE FALSE
## gyros_arm_x           1.015504     3.2769341   FALSE FALSE
## gyros_arm_y           1.454369     1.9162165   FALSE FALSE
## gyros_arm_z           1.110687     1.2638875   FALSE FALSE
## accel_arm_x           1.017341     3.9598410   FALSE FALSE
## accel_arm_y           1.140187     2.7367241   FALSE FALSE
## accel_arm_z           1.128000     4.0362858   FALSE FALSE
## magnet_arm_x          1.000000     6.8239731   FALSE FALSE
## magnet_arm_y          1.056818     4.4439914   FALSE FALSE
## magnet_arm_z          1.036364     6.4468454   FALSE FALSE
## roll_dumbbell         1.022388    84.2065029   FALSE FALSE
## pitch_dumbbell        2.277372    81.7449801   FALSE FALSE
## yaw_dumbbell          1.132231    83.4828254   FALSE FALSE
## total_accel_dumbbell  1.072634     0.2191418   FALSE FALSE
## gyros_dumbbell_x      1.003268     1.2282132   FALSE FALSE
## gyros_dumbbell_y      1.264957     1.4167771   FALSE FALSE
## gyros_dumbbell_z      1.060100     1.0498420   FALSE FALSE
## accel_dumbbell_x      1.018018     2.1659362   FALSE FALSE
## accel_dumbbell_y      1.053061     2.3748853   FALSE FALSE
## accel_dumbbell_z      1.133333     2.0894914   FALSE FALSE
## magnet_dumbbell_x     1.098266     5.7486495   FALSE FALSE
## magnet_dumbbell_y     1.197740     4.3012945   FALSE FALSE
## magnet_dumbbell_z     1.020833     3.4451126   FALSE FALSE
## roll_forearm         11.589286    11.0895933   FALSE FALSE
## pitch_forearm        65.983051    14.8557741   FALSE FALSE
## yaw_forearm          15.322835    10.1467740   FALSE FALSE
## total_accel_forearm   1.128928     0.3567424   FALSE FALSE
## gyros_forearm_x       1.059273     1.5187035   FALSE FALSE
## gyros_forearm_y       1.036554     3.7763735   FALSE FALSE
## gyros_forearm_z       1.122917     1.5645704   FALSE FALSE
## accel_forearm_x       1.126437     4.0464784   FALSE FALSE
## accel_forearm_y       1.059406     5.1116094   FALSE FALSE
## accel_forearm_z       1.006250     2.9558659   FALSE FALSE
## magnet_forearm_x      1.012346     7.7667924   FALSE FALSE
## magnet_forearm_y      1.246914     9.5403119   FALSE FALSE
## magnet_forearm_z      1.000000     8.5771073   FALSE FALSE
```

```r
tr.nonzerovar <- tr.dere[, zero.var[, 'nzv']==0]
dim(tr.nonzerovar)
```

```
## [1] 19622    53
```

* Remove highly correleated variables 90%


```r
corr.mat <- cor(na.omit(tr.nonzerovar[sapply(tr.nonzerovar, is.numeric)]))
dim(corr.mat)
```

```
## [1] 52 52
```

```r
corr.df <- expand.grid(row=1:dim(corr.mat)[1], col=1:dim(corr.mat)[2])
corr.df$correlation <- as.vector(corr.mat)
levelplot(correlation ~ row + col, corr.df)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png) 


```r
rm.corr <- findCorrelation(corr.mat, cutoff=.90, verbose=FALSE)
tr.decor <- tr.nonzerovar[, -rm.corr]
dim(tr.decor)
```

```
## [1] 19622    46
```

We get the data that has 19,622 samples 46 variables.

* Split data to trainig and testing for cross validation.


```r
in.train <- createDataPartition(y=tr.decor$class, p=0.7, list=FALSE)
tr.set <- tr.decor[in.train,]
te.set <- tr.decor[-in.train,]
c("tr.set:", dim(tr.set), "te.set:", dim(te.set))
```

```
## [1] "tr.set:" "13737"   "46"      "te.set:" "5885"    "46"
```

Finally, we got 13,737 samples and 46 variables for training set, also 5,885 samples and 46 variables for testing set.

### Analysis

---

* **Regression Tree**

Now we fit a tree to these data, and summarize and plot it. First, we use the *tree* packages.


```r
set.seed(12345)
tree.tr <- tree(classe ~ ., data=tr.set)
summary(tree.tr)
```

```
## 
## Classification tree:
## tree(formula = classe ~ ., data = tr.set)
## Variables actually used in tree construction:
##  [1] "pitch_forearm"     "magnet_belt_y"     "accel_forearm_z"  
##  [4] "magnet_dumbbell_y" "roll_forearm"      "magnet_dumbbell_z"
##  [7] "accel_dumbbell_y"  "pitch_belt"        "yaw_belt"         
## [10] "yaw_dumbbell"      "accel_forearm_x"   "magnet_forearm_y" 
## [13] "accel_dumbbell_z"  "gyros_belt_z"     
## Number of terminal nodes:  20 
## Residual mean deviance:  1.676 = 22990 / 13720 
## Misclassification error rate: 0.3245 = 4457 / 13737
```

```r
plot(tree.tr)
text(tree.tr, pretty=0, cex=.8)
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-1.png) 

* **Use Rpart from caret**


```r
model.fit <- train(classe ~ ., method="rpart", data=tr.set)
print(model.fit)
```

```
## CART 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.03753433  0.4985883  0.3551123  0.03412715   0.05416161
##   0.03855152  0.4927029  0.3475606  0.03053079   0.04972528
##   0.04297630  0.4458229  0.2686243  0.08673337   0.14353033
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03753433.
```

* **Prettier plots**


```r
fancyRpartPlot(model.fit$finalModel)
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-1.png) 

* **Cross Validation**

We are going to check the performance of the tree on the testing set.


```r
t.tree.pred <- predict(tree.tr, te.set, type="class")
t.confusion.mat <- with(te.set, table(t.tree.pred, classe))
sum(diag(t.confusion.mat)) / sum(as.vector(t.confusion.mat))
```

```
## [1] 0.6669499
```

We got the accuracy *0.693* from *tree* package model.


```r
c.tree.pred <- predict(model.fit, te.set)
c.confusion.mat <- with(te.set, table(c.tree.pred, classe))
sum(diag(c.confusion.mat)) / sum(as.vector(c.confusion.mat))
```

```
## [1] 0.5233645
```

On the other hand, we got the accuracy *0.50* from *caret* package model.

* **Trying Pruning tree**

This tree was grown to full depth, and might be too variable. We now use cross validation to prune it.


```r
cv.tr = cv.tree(tree.tr, FUN=prune.misclass)
cv.tr
```

```
## $size
##  [1] 20 19 17 16 15 14 11 10  7  5  1
## 
## $dev
##  [1] 4416 4500 4588 5171 5171 5188 6035 6103 6858 7257 9831
## 
## $k
##  [1]     -Inf  78.0000 118.5000 143.0000 144.0000 147.0000 194.6667
##  [8] 197.0000 235.3333 272.0000 648.5000
## 
## $method
## [1] "misclass"
## 
## attr(,"class")
## [1] "prune"         "tree.sequence"
```

```r
plot(cv.tr)
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png) 

It shows that when the size of the tree gose down, the deviance gose up. It means the 21 is a good size for this tree. We do not need to prune it.

### Random Forests

---

These methods use threes as building blocks to build more complex models.

* **Random Forests**

Random forests build lots of bushy trees, and then average them to reduce the variance.

Fit a random forest and see how well it peforms.


```r
set.seed(12345)
rf.tr <- randomForest(classe ~ ., data=tr.set, ntree=100, importance=TRUE)
rf.tr
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = tr.set, ntree = 100,      importance = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 6
## 
##         OOB estimate of  error rate: 0.73%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3903    2    0    0    1 0.0007680492
## B   25 2622    9    2    0 0.0135440181
## C    1   19 2372    4    0 0.0100166945
## D    0    0   25 2222    5 0.0133214920
## E    1    0    2    4 2518 0.0027722772
```

```r
varImpPlot(rf.tr)
```

![plot of chunk unnamed-chunk-15](figure/unnamed-chunk-15-1.png) 

### Out-of Sample Accuracy

---

Lets evaluate the fandom forest tree model on the test set.


```r
rf.tree.pred <- predict(rf.tr, te.set, type="class")
rf.confusion.mat <- with(te.set, table(rf.tree.pred, classe))
sum(diag(rf.confusion.mat)) / sum(as.vector(rf.confusion.mat))
```

```
## [1] 0.9909941
```

We got very hight accuracy *0.99* from *randomForest* package model.


### Conclusion

---

Now we can predict the testing data.


```r
ans <- predict(rf.tr, te.org)
ans
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```




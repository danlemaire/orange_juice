library(mlbench)
library(caret)
library(ROCR)
library(e1071)
library(tidyverse)

## Set seed for reproducibility
set.seed(99894)

## Load data and prepare
powerData1 <- read.csv(url("http://data.mishra.us/files/german.csv")) %>% 
  map_at(c(2,4,5,7,8,9,10,11,12,13,15,16,17,18,19,20), as.factor) %>% 
  as_tibble %>% 
  mutate(Creditability = ifelse(Creditability == 1, "Y", "N"))

## Create train and test datasets
inTraining <- createDataPartition(powerData1$Creditability, p = .8, list = FALSE)
training <- powerData1[ inTraining,]
testing  <- powerData1[-inTraining,]

## Create trainControl and grid
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 3,
  ## repeated ten times
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  returnResamp = "all")

grid <- expand.grid(sigma = .12,
                    C = seq(.75, 1.5, .05))

###### Build SVM model and look at results
svmFit1 <- train(Creditability ~ ., data = training, 
                 method = 'svmRadial',  
                 trControl = fitControl,
                 preProc = c("center","scale"),
                 metric = "ROC",
                 verbose = FALSE,
                 probability = TRUE,
                 tuneGrid = grid
                 
)

svmFit1
svmPred <- predict(svmFit1, newdata = testing, probability = TRUE)
confusionMatrix(data = svmPred, testing$Creditability)

##### Build Logistic model and look at results
logFit <- train(Creditability ~ .,  
                data = training, 
                method = "glm", 
                family = "binomial", 
                trControl = fitControl,
                metric = "ROC")
logPred <- predict(logFit, 
                   newdata = testing)
confusionMatrix(data = logPred, 
                testing$Creditability)

## Compare Train performance across logistic and SVM
results <- resamples(list(Logistic_Model = logFit,SVM_Model = svmFit1))
summary(results)
bwplot(results)

## Compare Test performance across logistic and SVM
 #Make test results dataframe
test_results <- data_frame(log_predictions = logPred, 
                           svm_predictions = svmPred,
                           labels = testing$Creditability) %>% 
  mutate(log_predictions = ifelse(log_predictions == "Y", 1, 0),
         svm_predictions = ifelse(svm_predictions == "Y", 1, 0),
         labels = ifelse(labels == "Y", 1, 0))

 #Review test results for logistic model
prediction(test_results$log_predictions, test_results$labels) %>% 
  performance("tpr", "fpr") %>% 
  plot(main = "Performance of Logistic Model against Test Set")
paste0("Performance of Logistic Model against Test Set: ",
       (prediction(test_results$log_predictions, test_results$labels) %>% performance("auc"))@y.values[[1]][1])

 #Review test results for svm model
prediction(test_results$svm_predictions, test_results$labels) %>% 
  performance("tpr", "fpr") %>% 
  plot(main = "Performance of SVM Model against Test Set")
paste0("Performance of SVM Model against Test Set: ",
       (prediction(test_results$svm_predictions, test_results$labels) %>% performance("auc"))@y.values[[1]][1])

## Review the variability in the predicted ROC's from the train dataset and compare to test ROC
svmFit1$resample$ROC %>% hist(main = "Histogram of ROC's of resamples for SVM")

svmFit1$resample$ROC %>% 
  as_tibble %>% 
  ggplot(aes(value)) +
    geom_histogram(bins = 20, alpha = .3) +
    geom_density() +
    geom_vline(xintercept = (prediction(test_results$svm_predictions, test_results$labels) %>% performance("auc"))@y.values[[1]][1], color = "red")


## Review the impact of sigma and C on expected ROC

 #Grid given in Shiny illustrates that sigma has more effect than C
svmFit1$results %>% 
  as_tibble %>% 
  mutate(best = ifelse(ROC == max(ROC), 1, 0)) %>% 
  ggplot(aes(sigma, C, color = ROC, size = factor(best), alpha = ROC)) + 
    geom_point() +
    labs(title = "Sigma parameter has more effect than C in this range")

 #Effect of Sigma (best is sigma = .012)
svmFit1$results %>% 
  as_tibble %>% 
  ggplot(aes(sigma, ROC)) + 
    geom_line() +
    labs(title = "Best sigma is 0.012")

 #Effect of C (best C = 1.25)
svmFit1$results %>% 
  as_tibble %>% 
  ggplot(aes(C, ROC)) + 
    geom_line() +
    labs(title = "Best C is 1.25")



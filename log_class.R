library(plyr)
library(caret)
library(ggplot2)
library(Rcpp)
library(ROCR)


set.seed(1243546)
powerData1<-read.csv(url("http://data.mishra.us/files/logit_data.csv"))


powerData1$segment <- factor(powerData1$segment)
split <- sample(seq_len(nrow(powerData1)), size = floor(0.75 * nrow(powerData1))) # 75% train and 25% test split
row_num<-nrow(powerData1)
trainData <- powerData1[split, ]
testData <- powerData1[-split, ]


predictionModel <- glm(buy ~ money_spent+num_visits+segment, data = trainData,
                       family=binomial(link='logit'))# Model fitting on the train data
summary(predictionModel)
exp(coef(predictionModel))# this produces exponentiated coefficients
prediction1 <- predict(predictionModel, newdata = testData, type='response') # Predictions on the test data
pred<-ifelse(prediction1 > 0.5,1,0) # probability cutoff is .5 here
confusionMatrix(pred,testData$buy)

## ROC Curve and AUC values

roc_data<-prediction(prediction1,testData$buy)
perf <- performance(roc_data, "tpr", "fpr") ## ROC plot
plot(perf,col=rainbow(10))

perf <- performance(roc_data, "acc") ##accuracy plot
plot(perf,col=rainbow(10))

auc.temp <- performance(roc_data,"auc")
auc <- as.numeric(auc.temp@y.values)
auc  # this prints the AUC value
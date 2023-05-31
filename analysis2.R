#############################################
#                                           #
#                PACKAGES                   #
#                                           #
#############################################

library("tidyverse")
library("randomForest")
library("ROCR")
library("MASS")
library("DescTools")
library("xtable")

#############################################
#                                           #
#                READ DATA                  #
#                                           #
#############################################

# setting the working directory 

setwd("D:\\Uni\\6. Semester\\Seminar\\R")

# read the data

heartfail.data <- read.csv("D:\\Uni\\6. Semester\\Seminar\\R\\heart_failure_clinical_records_dataset.csv")

# check for NA's and glimpse for an overview of the data set

colSums(sapply(heartfail.data, is.na))

glimpse(heartfail.data)

# correlations of response and independent variables, checking after
# ML-algos implementation if the var selection/var importance is accurate

as.matrix(cor(heartfail.data, method = "spearman")[, 13][order(abs(cor(heartfail.data, method = "spearman")[, 13]), decreasing = TRUE)])

# converting the categorical variables to factor

for (i in c(2, 4, 6, 10, 11, 13)) {
  
  heartfail.data[, i] <- as.factor(heartfail.data[, i])
}

glimpse(heartfail.data)

#############################################
#                                           #
#              TRAIN/TEST SET               #
#                                           #
#############################################

# seed for reproducability

set.seed(100)

# random sampling the train and test indices

ind.train <- sample(1:nrow(heartfail.data), nrow(heartfail.data) * 0.7, replace = FALSE)
ind.train <- sort(ind.train)
ind.test <- !(1:nrow(heartfail.data) %in% ind.train)

#############################################
#                                           #
#            LOGISTIC REGRESSION            #
#                                           #
#############################################

# specifying a full and null model 

fullmodel <- glm(DEATH_EVENT ~ ., data = heartfail.data, subset = ind.train, family = "binomial")
nullmodel <- glm(DEATH_EVENT ~ 1, data = heartfail.data, subset = ind.train, family = "binomial")

# applying the backward elimination algorithm

logreg.fit <- stepAIC(fullmodel,
                      direction = "backward",
                      scope = list(upper = fullmodel,
                                   lower = nullmodel),
                      trace = 0)


summary(logreg.fit)

# cuttoffs for the logistic regression

cutoff.logreg <- seq(0.1, 1, 0.1)

# creating a list for the best model and parameters
# predict the outcomes 

logreg.results.opt <- vector("list", 6)
names(logreg.results.opt) <- c("cutoff", "auc", "pred.dich", "pred.prob", "rocr", "brier")
logreg.results.opt$auc <- -Inf
logreg.predict.prob <- predict(logreg.fit, heartfail.data[ind.test, -13], type = "response")

# for loop's algorithm:
#   getting the dichotome predictions with the different cutoffs for every iteration
#   creating the ROCR predictions objects and compute the AUC
#   if the AUC is larger than the before best measured value
#   we replace the parameters with the cutoff based on the best AUC

for (i in cutoff.logreg) {
  
  logreg.predict.dich <- ifelse(logreg.predict.prob > i, 1, 0)
  
  logreg.rocr.pred <- prediction(logreg.predict.dich, heartfail.data[ind.test, 13])
  logreg.rocr.auc <- performance(logreg.rocr.pred, measure = "auc")
  print(logreg.rocr.auc@y.values)
  
  if (logreg.rocr.auc@y.values[[1]] > logreg.results.opt$auc) {
    
    logreg.results.opt$cutoff <- i
    logreg.results.opt$auc <- logreg.rocr.auc@y.values[[1]]
    logreg.results.opt$pred.dich <- logreg.predict.dich
    logreg.results.opt$pred.prob <- logreg.predict.prob
    logreg.results.opt$rocr <- prediction(logreg.predict.prob, heartfail.data[ind.test, 13])
    logreg.results.opt$brier <- BrierScore(as.numeric(as.character(heartfail.data[ind.test, 13])),
                                           logreg.predict.prob)
    
  }
}


#############################################
#                                           #
#               RANDOM FOREST               #
#                                           #
#############################################

cutoff.rf <- seq(0.1, 1, 0.1)

rf.results.opt <- vector("list", 6)

set.seed(123)

names(rf.results.opt) <- c("cutoff", "auc", "pred.dich", "pred.prob", "rocr", "brier")
rf.results.opt$auc <- -Inf
fit.rf <- tuneRF(heartfail.data[ind.train, -13], heartfail.data[ind.train, 13], doBest = TRUE)
rf.predict.prob <- predict(fit.rf, heartfail.data[ind.test, -13], type = "prob")[, 2]

# loop works the same as for the logistic regression

for (i in cutoff.rf) {
  
  rf.predict.dich <- ifelse(rf.predict.prob > i, 1, 0)

  rf.rocr.pred <- prediction(rf.predict.dich, heartfail.data[ind.test, 13])
  rf.rocr.auc <- performance(rf.rocr.pred, measure = "auc")
  print(rf.rocr.auc@y.values)
  
  if (rf.rocr.auc@y.values[[1]] > rf.results.opt$auc) {
    
    rf.results.opt$cutoff <- i
    rf.results.opt$auc <- rf.rocr.auc@y.values[[1]]
    rf.results.opt$pred.dich <- rf.predict.dich
    rf.results.opt$pred.prob <- rf.predict.prob
    rf.results.opt$rocr <- prediction(rf.predict.prob, heartfail.data[ind.test, 13])
    rf.results.opt$brier <- BrierScore(as.numeric(as.character(heartfail.data[ind.test, 13])),
                                       rf.predict.prob)
  }
}


#############################################
#                                           #
#                 RESULTS                   #
#                                           #
#############################################

# confusion matrix for the results

table(heartfail.data[ind.test, 13], logreg.results.opt$pred.dich)
table(heartfail.data[ind.test, 13], rf.results.opt$pred.dich)

# plotting the ROC curves

ggplot() + 
  geom_line(aes(x = performance(logreg.results.opt$rocr, measure = "rch")@x.values[[1]],
                y = performance(logreg.results.opt$rocr, measure = "rch")@y.values[[1]], 
                color = "Logistic Regression"), 
            linewidth = 1.15, alpha = 0.7) +
  geom_line(aes(x = performance(rf.results.opt$rocr, measure = "rch")@x.values[[1]],
                y = performance(rf.results.opt$rocr, measure = "rch")@y.values[[1]], 
                color = "Random Forest"), 
            linewidth = 1.15, alpha = 0.7) +
  xlab("False-Positive-Rate") +
  ylab("True-Positive-Rate") +
  scale_color_manual(name = "Method", 
                     breaks = c("Logistic Regression", "Random Forest"),
                     values = c("Logistic Regression" = "red", "Random Forest" = "blue")) +
  theme(legend.position="bottom")

# calculation of the performance measurements

logreg.sens <- performance(logreg.results.opt$rocr, measure = "sens")
rf.sens <- performance(rf.results.opt$rocr, measure = "sens")

logreg.sens <- logreg.sens@y.values[[1]][max(which(logreg.sens@x.values[[1]] > 
                                                     logreg.results.opt$cutoff))]
rf.sens <- rf.sens@y.values[[1]][max(which(rf.sens@x.values[[1]] > 
                                             rf.results.opt$cutoff))]



logreg.spec <- performance(logreg.results.opt$rocr, measure = "spec")
rf.spec <- performance(rf.results.opt$rocr, measure = "spec")


logreg.spec <-logreg.spec@y.values[[1]][max(which(logreg.spec@x.values[[1]] > 
                                                    logreg.results.opt$cutoff))]
rf.spec <-rf.spec@y.values[[1]][max(which(rf.spec@x.values[[1]] > 
                                            rf.results.opt$cutoff))]



logreg.acc <- performance(logreg.results.opt$rocr, measure = "acc")
rf.acc <- performance(rf.results.opt$rocr, measure = "acc")

logreg.acc <- logreg.acc@y.values[[1]][max(which(logreg.acc@x.values[[1]] > 
                                                   logreg.results.opt$cutoff))]
rf.acc <- rf.acc@y.values[[1]][max(which(rf.acc@x.values[[1]] > 
                                           rf.results.opt$cutoff))]


logreg.auc <- performance(logreg.results.opt$rocr, measure = "auc")
rf.auc <- performance(rf.results.opt$rocr, measure = "auc")

logreg.auc <- logreg.auc@y.values
rf.auc <- rf.auc@y.values


rf.importance <- as.matrix(importance(fit.rf)[order(importance(fit.rf), decreasing = TRUE), ])
quant.measures <- matrix(c(logreg.acc, logreg.sens, logreg.spec, logreg.auc, logreg.results.opt$brier, 
                         rf.acc, rf.sens, rf.spec, rf.auc, rf.results.opt$brier),
                         ncol = 2)

colnames(quant.measures) <- c("Logistic Regression", "Random Forest")
rownames(quant.measures) <- c("Accuracy", "Sensitivity", "Specificity", "AUC", "Brier Score")


# variable importances show, that the stepwise backward elimination algo was accurate for logreg
# as well as the the var importance measurements from rf


# tables for latex

print(xtable(logreg.fit, type = "latex"), file = "logreg.tex")
print(xtable(rf.importance, type = "latex"), file = "rf.tex")
print(xtable(quant.measures, type = "latex"), file = "quant.measures.tex")






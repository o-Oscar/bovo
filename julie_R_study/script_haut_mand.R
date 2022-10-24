setwd('/home/oscar/bovo/julie_R_study')

### Packages

library(dplyr)
library(tidyr)
library(readxl)
library(e1071)
library(caTools)
library(caret)
library(FactoMineR)
library(Hmisc)
library(randomForest)
library(glmnet)

### ANALYSES SUR HAUTEURS MANDIBULAIRES ###

#Données

data2 <- read.csv('data2.csv')
indicateurs<-c('surf_alv','surf_mand','surf_rac','surf_souche','haut_alv_mand','haut_alv','haut_mand')
var_expli<- colnames(data2)[!(colnames(data2) %in% indicateurs)]

data_haut_mand<-data2 %>% dplyr::select(haut_mand,all_of(var_expli))

data_haut_mand <- lapply(data_haut_mand,as.numeric)
data_haut_mand <- as.data.frame(data_haut_mand)


#Construction des classes
hist(data_haut_mand$haut_mand, breaks = seq(min(data_haut_mand$haut_mand), max(data_haut_mand$haut_mand), length.out = 30))

data_haut_mand$classes <- case_when(data_haut_mand$haut_mand<=0.14~'moyenne',
                                    TRUE~'forte')


data_haut_mand$classes <- factor(data_haut_mand$classes,levels=c('moyenne','forte'))

#variables pertinentes ?
control <- rfeControl(functions=rfFuncs, method="cv", number=10)


rfe <- rfe(haut_mand ~ ., data = data_haut_mand[colnames(data_haut_mand)%in%c(var_expli,'haut_mand')] ,
           sizes = (5:60),
           method = "svm",
           rfeControl = control)

nouvelles_expli_haut_mand <- predictors(rfe)

data_haut_mand <- data_haut_mand %>% dplyr::select(nouvelles_expli_haut_mand,haut_mand,classes)

#bases train/test

set.seed(123) 
split = sample.split(data_haut_mand$classes, SplitRatio = 0.75) 

training= subset(data_haut_mand, split == TRUE) 
test= subset(data_haut_mand, split == FALSE)

training <- training %>% dplyr::select(-haut_mand)
test <- test %>% dplyr::select(-haut_mand)


# SVM 

#ajustement des paramètres

tuned_parameters <- tune.svm(factor(classes)~., data = training, gamma = 10^(-5:-1), cost = (10^(-2:1)))
summary(tuned_parameters)

svm_haut_mand<-svm(classes~.,data=training,type="C-classification",kernel='radial',gamma=tuned_parameters$best.parameters$gamma,cost=tuned_parameters$best.parameters$cost,cross=10,probability=TRUE,na.action = na.omit)
summary(svm_haut_mand)

#prédiction ?

pred <- predict(svm_haut_mand,test,probability = TRUE)
matconf_haut_mand<-confusionMatrix(table(na.omit(test)$classes,pred))
matconf_haut_mand

###distinction des valeurs faibles 


data_haut_mand$classes2 <- case_when(data_haut_mand$haut_mand<=0.05~'faible',
                                     TRUE~'moyenne')

data_haut_mand$classes2 <- factor(data_haut_mand$classes2,levels=c('faible','moyenne'))


data_haut_mand <- data_haut_mand %>% select(-classes)

data_haut_mand <- data_haut_mand %>% dplyr::select(haut_mand,classes2,all_of(nouvelles_expli_haut_mand))

#bases train/test

set.seed(123) 
split = sample.split(data_haut_mand$classes2, SplitRatio = 0.75) 

training= subset(data_haut_mand, split == TRUE) 
test= subset(data_haut_mand, split == FALSE)

training <- training %>% dplyr::select(-haut_mand)
test <- test %>% dplyr::select(-haut_mand)


#SVM classes (faible)

tuned_parameters <- tune.svm(factor(classes2)~., data = training, gamma = 10^(-5:-1), cost = (10^(-2:1)))
summary(tuned_parameters)

svm_haut_mand2<-svm(classes2~.,data=training,type="C-classification",kernel='radial',gamma=tuned_parameters$best.parameters$gamma,cost=tuned_parameters$best.parameters$cost,cross=10,probability=TRUE,na.action = na.omit)
summary(svm_haut_mand2)

pred2 <- predict(svm_haut_mand2,test,probability = TRUE)
matconf_haut_mand2<-confusionMatrix(table(na.omit(test)$classes2,pred2))
matconf_haut_mand2



## Autres modèles ##

#On prend une base avec classes, et une avec classes 2
data_haut_mand2 <- data_haut_mand

data_haut_mand <- data_haut_mand %>% select(-classes2) 
data_haut_mand$classes <- case_when(data_haut_mand$haut_mand<=0.14~'moyenne',
                                    TRUE~'forte')

data_haut_mand$classes <- factor(data_haut_mand$classes,levels=c('moyenne','forte'))

#bases train/test
#base classe faible (classes2)
set.seed(123) 
split2 = sample.split(data_haut_mand2$classes2, SplitRatio = 0.75) 

training2= subset(data_haut_mand2, split2 == TRUE) 
test2= subset(data_haut_mand2, split == FALSE)

training2 <- training2 %>% dplyr::select(-haut_mand)
test2 <- test2 %>% dplyr::select(-haut_mand)

#base classe élevée (classes)
set.seed(123) 
split = sample.split(data_haut_mand$classes2, SplitRatio = 0.75) 

training= subset(data_haut_mand, split == TRUE) 
test= subset(data_haut_mand, split == FALSE)

training <- training %>% dplyr::select(-haut_mand)
test <- test %>% dplyr::select(-haut_mand)

#random forest : classes (forte)

#variation nombre d'arbres pour le choix du paramètre

set.seed(123)
rf_ntree <- randomForest(classes ~ ., data = training, ntree = 5000, 
                         mtry = 2, na.action = na.omit)
plot(rf_ntree$err.rate[, 1], type = "l", xlab = "nombre d'arbres", ylab = "erreur OOB")

set.seed(123)
rf <- randomForest(classes ~ ., data = training, ntree=500,na.action = na.omit)
confusionMatrix(table(test$classes,predict(rf,test)))


#random forest : classes2 (faible)

rf2 <- randomForest(classes2 ~ ., data = training2, na.action = na.omit)

#variation ntree

set.seed(123)
rf_ntree2 <- randomForest(classes2 ~ ., data = training2, ntree = 5000, 
                          mtry = 2, na.action = na.omit)
plot(rf_ntree2$err.rate[, 1], type = "l", xlab = "nombre d'arbres", ylab = "erreur OOB")

set.seed(123)
rf2 <- randomForest(classes2 ~ ., data = training2, ntree=500,na.action = na.omit)
confusionMatrix(table(test2$classes2,predict(rf2,test2)))


#régression logistique

    #classes (forte)
reglog <- glm(classes ~ ., data = training, family = "binomial")
summary(reglog)

#predictions
probabilities <- reglog %>% predict(test, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "forte", "moyenne")

#accuracy
mean(predicted.classes == test$classes)


    #classes2 (faible)
reglog2 <- glm(classes2 ~ ., data = training2, family = "binomial")
summary(reglog2)

#predictions
probabilities2 <- reglog2 %>% predict(test2, type = "response")
predicted.classes2 <- ifelse(probabilities2 > 0.5, "moyenne", "faible")

#accuracy
mean(predicted.classes2 == test2$classes2)



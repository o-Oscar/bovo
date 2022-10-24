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

### ANALYSES SUR SURFACES ALVEOLAIRES ###

#Données

data2 <- read.csv('data2.csv')
indicateurs<-c('surf_alv','surf_mand','surf_rac','surf_souche','haut_alv_mand','haut_alv','haut_mand')
var_expli<- colnames(data2)[!(colnames(data2) %in% indicateurs)]

data_surf_alv<-data2 %>% dplyr::select(surf_alv,all_of(var_expli))

data_surf_alv <- lapply(data_surf_alv,as.numeric)
data_surf_alv <- as.data.frame(data_surf_alv)

#Construction des classes
hist(data_surf_alv$surf_alv, breaks = seq(min(data_surf_alv$surf_alv), max(data_surf_alv$surf_alv), length.out = 30))


data_surf_alv$classes <- case_when(data_surf_alv$surf_alv<=0.21~'moyenne',
                                    TRUE~'forte')


data_surf_alv$classes <- factor(data_surf_alv$classes,levels=c('moyenne','forte'))


#variables pertinentes ?
control <- rfeControl(functions=rfFuncs, method="cv", number=10)


rfe <- rfe(surf_alv ~ ., data = data_surf_alv[colnames(data_surf_alv)%in%c(var_expli,'surf_alv')] ,
           sizes = (5:60),
           method = "svm",
           rfeControl = control)

nouvelles_expli_surf_alv <- predictors(rfe)
data_surf_alv <- data_surf_alv %>% dplyr::select(nouvelles_expli_surf_alv,surf_alv,classes)

#bases train/test

set.seed(123) 
split = sample.split(data_surf_alv$classes, SplitRatio = 0.75) 

training= subset(data_surf_alv, split == TRUE) 
test= subset(data_surf_alv, split == FALSE)

training <- training %>% dplyr::select(-surf_alv)
test <- test %>% dplyr::select(-surf_alv)


# SVM 

#ajustement des paramètres

tuned_parameters <- tune.svm(factor(classes)~., data = training, gamma = 10^(-5:-1), cost = (10^(-2:1)))
summary(tuned_parameters)

svm_surf_alv<-svm(classes~.,data=training,type="C-classification",kernel='radial',gamma=tuned_parameters$best.parameters$gamma,cost=tuned_parameters$best.parameters$cost,cross=10,probability=TRUE,na.action = na.omit)
summary(svm_surf_alv)

#prédiction ?

pred <- predict(svm_surf_alv,test,probability = TRUE)
matconf_surf_alv<-confusionMatrix(table(na.omit(test)$classes,pred))
matconf_surf_alv

###distinction des valeurs faibles 


data_surf_alv$classes2 <- case_when(data_surf_alv$surf_alv<=0.05~'faible',
                                     TRUE~'moyenne')

data_surf_alv$classes2 <- factor(data_surf_alv$classes2,levels=c('faible','moyenne'))


data_surf_alv <- data_surf_alv %>% select(-classes)

data_surf_alv <- data_surf_alv %>% dplyr::select(surf_alv,classes2,all_of(nouvelles_expli_surf_alv))

#bases train/test

set.seed(123) 
split = sample.split(data_surf_alv$classes2, SplitRatio = 0.75) 

training= subset(data_surf_alv, split == TRUE) 
test= subset(data_surf_alv, split == FALSE)

training <- training %>% dplyr::select(-surf_alv)
test <- test %>% dplyr::select(-surf_alv)


#SVM classes (faible)

tuned_parameters <- tune.svm(factor(classes2)~., data = training, gamma = 10^(-5:-1), cost = (10^(-2:1)))
summary(tuned_parameters)

svm_surf_alv2<-svm(classes2~.,data=training,type="C-classification",kernel='radial',gamma=tuned_parameters$best.parameters$gamma,cost=tuned_parameters$best.parameters$cost,cross=10,probability=TRUE,na.action = na.omit)
summary(svm_surf_alv2)

pred2 <- predict(svm_surf_alv2,test,probability = TRUE)
matconf_surf_alv2<-confusionMatrix(table(na.omit(test)$classes2,pred2))
matconf_surf_alv2

## Autres modèles ##

#On prend une base avec classes, et une avec classes 2
data_surf_alv2 <- data_surf_alv

data_surf_alv <- data_surf_alv %>% select(-classes2) 
data_surf_alv$classes <- case_when(data_surf_alv$surf_alv<=0.14~'moyenne',
                                    TRUE~'forte')

data_surf_alv$classes <- factor(data_surf_alv$classes,levels=c('moyenne','forte'))

#bases train/test
#base classe faible (classes2)
set.seed(123) 
split2 = sample.split(data_surf_alv2$classes2, SplitRatio = 0.75) 

training2= subset(data_surf_alv2, split2 == TRUE) 
test2= subset(data_surf_alv2, split == FALSE)

training2 <- training2 %>% dplyr::select(-surf_alv)
test2 <- test2 %>% dplyr::select(-surf_alv)

#base classe élevée (classes)
set.seed(123) 
split = sample.split(data_surf_alv$classes2, SplitRatio = 0.75) 

training= subset(data_surf_alv, split == TRUE) 
test= subset(data_surf_alv, split == FALSE)

training <- training %>% dplyr::select(-surf_alv)
test <- test %>% dplyr::select(-surf_alv)

#random forest : classes (forte)

#variation ntree

set.seed(123)
rf_ntree <- randomForest(classes ~ ., data = training, ntree = 5000, 
                         mtry = 2, na.action = na.omit)
plot(rf_ntree$err.rate[, 1], type = "l", xlab = "nombre d'arbres", ylab = "erreur OOB")
#ntree=3000

set.seed(123)
rf <- randomForest(classes ~ ., data = training, ntree=500,na.action = na.omit)
rf

confusionMatrix(table(test$classes,predict(rf,test)))

#random forest : classes2 (faible)

#variation ntree

set.seed(123)
rf_ntree2 <- randomForest(classes2 ~ ., data = training2, ntree = 5000, 
                          mtry = 2, na.action = na.omit)
plot(rf_ntree$err.rate[, 1], type = "l", xlab = "nombre d'arbres", ylab = "erreur OOB")

set.seed(123)
rf2 <- randomForest(classes2 ~ ., data = training2, ntree=500,na.action = na.omit)
rf2

confusionMatrix(table(test2$classes2,predict(rf2,test2)))


#régression logistique

#classes (forte)
reglog <- glm(classes ~ ., data = training, family = "binomial")
summary(reglog)

probabilities <- reglog %>% predict(test, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "forte", "moyenne")
#accuracy
mean(predicted.classes == test$classes)

#classes2 (faible)
reglog2 <- glm(classes2 ~ ., data = training2, family = "binomial")
summary(reglog2)

probabilities2 <- reglog2 %>% predict(test2, type = "response")
predicted.classes2 <- ifelse(probabilities2 > 0.5, "moyenne", "faible")
mean(predicted.classes2 == test2$classes2)


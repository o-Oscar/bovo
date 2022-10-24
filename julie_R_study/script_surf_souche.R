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

### ANALYSES SUR SURFACES SOUCHES ###

#Données

data2 <- read.csv('data2.csv')
indicateurs<-c('surf_alv','surf_mand','surf_rac','surf_souche','haut_alv_mand','haut_alv','haut_mand')
var_expli<- colnames(data2)[!(colnames(data2) %in% indicateurs)]

data_surf_souche<-data2 %>% dplyr::select(surf_souche,all_of(var_expli))

data_surf_souche <- lapply(data_surf_souche,as.numeric)
data_surf_souche <- as.data.frame(data_surf_souche)

#Construction des classes
data_surf_souche <- na.omit(data_surf_souche,cols=surf_souche)

hist(data_surf_souche$surf_souche, breaks = seq(min(data_surf_souche$surf_souche), max(data_surf_souche$surf_souche), length.out = 30))

data_surf_souche$classes <- case_when(data_surf_souche$surf_souche<=0.1~'moyenne',
                                    TRUE~'forte')


data_surf_souche$classes <- factor(data_surf_souche$classes,levels=c('moyenne','forte'))


#variables pertinentes ?
# control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# 
# 
# rfe <- rfe(surf_souche ~ ., data = data_surf_souche[colnames(data_surf_souche)%in%c(var_expli,'surf_souche')] ,
#            sizes = (5:60),
#            method = "svm",
#            rfeControl = control)
# 
# predictors(rfe)
#On retient beaucoup trop de variables avec cette méthode : par défaut, on prend les variables retenues 
#pour la prédiction de surf_mand

nouvelles_expli_surf_souche<-c('SPeToMa_dente', 'SPeToCo_dente', 'BC_dente', 'M_CEJ_dente', 'MC_dente', 'SSuApexC_dente', 'CApex_dente', 'SSuToMa_dente', 'SPeToTr_dente', 'Airesecteur1_dente', 'SLaTaOsAp_dente', 'SSuApexM_dente', 'FM2_edente', 'SSuToCo_dente', 'SSuMaPaAl_dente')

data_surf_souche <- data_surf_souche %>% dplyr::select(nouvelles_expli_surf_souche,surf_souche,classes)

#bases train/test

set.seed(123) 
split = sample.split(data_surf_souche$classes, SplitRatio = 0.75) 

training= subset(data_surf_souche, split == TRUE) 
test= subset(data_surf_souche, split == FALSE)

training <- training %>% dplyr::select(-surf_souche)
test <- test %>% dplyr::select(-surf_souche)


# SVM 

#ajustement des paramètres

tuned_parameters <- tune.svm(factor(classes)~., data = training, gamma = 10^(-5:-1), cost = (10^(-2:1)))
summary(tuned_parameters)

svm_surf_souche<-svm(classes~.,data=training,type="C-classification",kernel='radial',gamma=tuned_parameters$best.parameters$gamma,cost=tuned_parameters$best.parameters$cost,cross=10,probability=TRUE,na.action = na.omit)
summary(svm_surf_souche)

#prédiction ?

pred <- predict(svm_surf_souche,test,probability = TRUE)
matconf_surf_souche<-confusionMatrix(table(na.omit(test)$classes,pred))
matconf_surf_souche

###distinction des valeurs faibles 


data_surf_souche$classes2 <- case_when(data_surf_souche$surf_souche<=0.05~'faible',
                                     TRUE~'moyenne')

data_surf_souche$classes2 <- factor(data_surf_souche$classes2,levels=c('faible','moyenne'))


data_surf_souche <- data_surf_souche %>% select(-classes)

data_surf_souche <- data_surf_souche %>% dplyr::select(surf_souche,classes2,all_of(nouvelles_expli_surf_souche))

#bases train/test

set.seed(123) 
split = sample.split(data_surf_souche$classes2, SplitRatio = 0.75) 

training= subset(data_surf_souche, split == TRUE) 
test= subset(data_surf_souche, split == FALSE)

training <- training %>% dplyr::select(-surf_souche)
test <- test %>% dplyr::select(-surf_souche)


#SVM classes (faible)

tuned_parameters <- tune.svm(factor(classes2)~., data = training, gamma = 10^(-5:-1), cost = (10^(-2:1)))
summary(tuned_parameters)

svm_surf_souche2<-svm(classes2~.,data=training,type="C-classification",kernel='radial',gamma=tuned_parameters$best.parameters$gamma,cost=tuned_parameters$best.parameters$cost,cross=10,probability=TRUE,na.action = na.omit)
summary(svm_surf_souche2)

pred2 <- predict(svm_surf_souche2,test,probability = TRUE)
matconf_surf_souche2<-confusionMatrix(table(na.omit(test)$classes2,pred2))
matconf_surf_souche2

## Autres modèles ##

#On prend une base avec classes, et une avec classes 2
data_surf_souche2 <- data_surf_souche

data_surf_souche <- data_surf_souche %>% select(-classes2) 
data_surf_souche$classes <- case_when(data_surf_souche$surf_souche<=0.14~'moyenne',
                                    TRUE~'forte')

data_surf_souche$classes <- factor(data_surf_souche$classes,levels=c('moyenne','forte'))

#bases train/test
#base classe faible (classes2)
set.seed(123) 
split2 = sample.split(data_surf_souche2$classes2, SplitRatio = 0.75) 

training2= subset(data_surf_souche2, split2 == TRUE) 

test2= subset(data_surf_souche2, split == FALSE)

training2 <- training2 %>% dplyr::select(-surf_souche)
test2 <- test2 %>% dplyr::select(-surf_souche)

#base classe élevée (classes)
set.seed(123) 
split = sample.split(data_surf_souche$classes2, SplitRatio = 0.75) 

training= subset(data_surf_souche, split == TRUE) 
test= subset(data_surf_souche, split == FALSE)

training <- training %>% dplyr::select(-surf_souche)
test <- test %>% dplyr::select(-surf_souche)

#random forest : classes (forte)

#variation ntree

set.seed(123)
rf_ntree <- randomForest(classes ~ ., data = training, ntree = 5000, 
                         mtry = 2, na.action = na.omit)
plot(rf_ntree$err.rate[, 1], type = "l", xlab = "nombre d'arbres", ylab = "erreur OOB")

set.seed(123)
rf <- randomForest(classes ~ ., data = training, ntree=5,na.action = na.omit)
confusionMatrix(table(test$classes,predict(rf,test)))

#random forest : classes2 (faible)

#variation ntree

set.seed(123)
rf_ntree2 <- randomForest(classes2 ~ ., data = training2, ntree = 5000, 
                          mtry = 2, na.action = na.omit)
plot(rf_ntree$err.rate[, 1], type = "l", xlab = "nombre d'arbres", ylab = "erreur OOB")

set.seed(123)
rf2 <- randomForest(classes2 ~ ., data = training2, ntree=20,na.action = na.omit)
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

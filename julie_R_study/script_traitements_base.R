# Mission JE Bovo Predict
# Prévision de perte osseuse suite à une extraction dentaire

  ### Dossier de travail

setwd('/home/oscar/bovo/julie_R_study')

  ### Packages

library(dplyr)
library(tidyr)
library(readxl)
library(e1071)
library(caTools)
library(caret)
library(FactoMineR)

  ### Données

data<-read_excel('15 10 2020 VFxlsx  Version Finale Etude 116 cas.xlsx',sheet = "Quantification")

  ### Traitement préliminaires


data<-data %>% mutate(ID_patient=substr(Patient,1,4)) %>% 
  dplyr::select(-c(Patient,Cote))

#Patient 2644 en double (4 lignes au lieu de 2) : impossible de retrouver les "bonnes" paires
#Suppression de ce patient

data<-data%>%filter(!(ID_patient %in% c('2644','2609','2642','2632','2832')))

#Recodage de sexe

data$sexe <- case_when(data$sexe=='F'~1,
                       data$sexe=='M'~0)

data$sexe <- as.factor(data$sexe)

  #Création d'une base avec un patient par ligne, et des variables avec suffixe denté/édenté
  #un certain nombre de variables n'ont pas besoin d'être distinguées par dente/edenté ou ne seront pas
  #utilisées dans l'étude : à supprimer

dente<-data%>%filter(cote=='dente') %>% 
  dplyr::select(-c(cote,PositionMedian,cote_resorption,position_resorption,cote_dente,position_dente,NouvelleSelection_dente,position_NouvelleSelection,écart,yC,xC,CLLaTaOsAp,CLLaTrAp,CLSuApexC,CLSuApexM))
edente<-data %>% filter(cote=="edente") %>% 
  dplyr::select(-c(cote,PositionMedian,cote_resorption,position_resorption,cote_dente,position_dente,NouvelleSelection_dente,position_NouvelleSelection,écart,yC,xC,CLLaTaOsAp,CLLaTrAp,CLSuApexC,CLSuApexM))

data2<-left_join(dente,edente,by='ID_patient',suffix=c('_dente','_edente'))

  #Création des indicateurs de mesure de perte osseuse

#Vérification des NA dans les variables utilisées pour les indicateurs
data2 <- na_if(data2,'NA')
colSums(is.na(data2))

data2 <- data2 %>% mutate(surf_alv=(SSuMaPaAl_dente-SSuMaPaAl_edente)/SSuMaPaAl_dente,
                          surf_mand=((SSuToMa_dente)-(SSuToMa_edente))/(SSuToMa_dente),
                          surf_rac=(SSuApexC_dente-SSuApexC_edente)/SSuApexC_dente,
                          haut_alv_mand=((MC_dente/BC_dente)-(MC_edente/BC_edente))/(MC_dente/ BC_dente),
                          haut_alv=(MC_dente-MC_edente)/MC_dente,
                          haut_mand=(BC_dente-BC_edente)/BC_dente,
                          surf_souche=(SSuApexM_dente-SSuApexM_edente)/SSuApexM_dente)

#Finalisation de la base pour l'étude : on ne garde que les variables expliquées (qu'on vient de créer)
# et les variables explicatives (sur le côté dente)

data2 <- data2 %>% dplyr::select(ends_with('_dente'),starts_with(c('surf','haut')),FM1_edente,FM2_edente) %>% 
  dplyr::select(-c(Age_dente,FM1_dente,FM2_dente,ResorptionCortical_dente))

#On supprime les variables relatives à l'échancrure, car elles introduisent beaucoup de NA. 
#Il sera intéressant par la suite de faire une analyse particulière pour les individus qui ont une échancrure

data2 <- data2 %>% dplyr::select(-c(profondeurEchancrure_dente,E_B_dente,E_CEJ_dente,AngleEchancrure_dente))

write.csv(data2,'data2.csv')

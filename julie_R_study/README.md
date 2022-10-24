

# lancement des scripts de Julie

## modifications des scripts

changement des lignes 

```
setwd('C:/Users/Laurent Sixou/Documents/ENSAE/Mission JE Bovo Predict')
```

en 

```
setwd('/home/oscar/bovo/julie_R_study')
```

## output non nominal des scripts

```
Error in sample.split(data_surf_mand$classes2, SplitRatio = 0.75) : 
  Error in sample.split: 'SplitRatio' parameter has to be i [0, 1] range or [1, length(Y)] range
Execution halted
```

Interprétation : Cette erreur empêche d'exécuter le code et a fortiori d'avoir les résultats des svm sur les fortes pertes osseuses.

# comparaisons des résultats

## script_haut_mand.R

          faible moyenne
  faible       0       6
  moyenne      0      21
  pred2

## script_surf_alv.R

          faible moyenne
  faible       0       6
  moyenne      0      22

Donne les mêmes résultats que sur le rapport

## script2_surf_mand.R

          faible moyenne
  faible       2       4
  moyenne      2      20

## script_haut_alv_mand.R

          faible moyenne
  faible       0      11
  moyenne      0      16


## script_surf_rac.R

          faible moyenne
  faible       0       5
  moyenne      0      21


## script_surf_souche.R

          faible moyenne
  faible      18       0
  moyenne      8       0

Donne les mêmes résultats que sur le rapport

# Conclusion 

Seuls deux SVM donnent les mêmes résultats que dans le rapport. 
rm(list=ls())

source('./cali_functions.R')

#library(survcomp)
library(dplyr)
library(reshape2)
library(ggplot2)
library(e1071)
library(pROC)
#library(compareC)

RS_composite_all.pred_score.dsv.train.txt = read.delim('../../RS_composite_all.pred_score.dsv.train.txt', header = F)
RS_composite_all.pred_score.dsv.test.txt = read.delim('../../RS_composite_all.pred_score.dsv.test.txt', header = F)
RS_composite_all.pred_score.cox.train.txt = read.delim('../../RS_composite_all.pred_score.cox.train.txt', header = F)
RS_composite_all.pred_score.cox.test.txt = read.delim('../../RS_composite_all.pred_score.cox.test.txt', header = F)
RS_composite_all.pred_score.rsf.train.txt = read.delim('../../rsf/RS_composite_all.pred_score.rsf.train.txt', header = F)
RS_composite_all.pred_score.rsf.test.txt = read.delim('../../rsf/RS_composite_all.pred_score.rsf.test.txt', header = F)
RS_composite_all.E_train.txt = read.delim('../../RS_composite_all.E_train.txt', header = F)
RS_composite_all.T_train.txt = read.delim('../../RS_composite_all.T_train.txt', header = F)
RS_composite_all.E_test.txt = read.delim('../../RS_composite_all.E_test.txt', header = F)
RS_composite_all.T_test.txt = read.delim('../../RS_composite_all.T_test.txt', header = F)

RS_composite_sans_base.pred_score.dsv.train.txt = read.delim('../../RS_composite_sans_base.pred_score.dsv.train.txt', header = F)
RS_composite_sans_base.pred_score.dsv.test.txt = read.delim('../../RS_composite_sans_base.pred_score.dsv.test.txt', header = F)
RS_composite_sans_base.pred_score.cox.train.txt = read.delim('../../RS_composite_sans_base.pred_score.cox.train.txt', header = F)
RS_composite_sans_base.pred_score.cox.test.txt = read.delim('../../RS_composite_sans_base.pred_score.cox.test.txt', header = F)
RS_composite_sans_base.pred_score.rsf.train.txt = read.delim('../../rsf/RS_composite_sans_base.pred_score.rsf.train.txt', header = F)
RS_composite_sans_base.pred_score.rsf.test.txt = read.delim('../../rsf/RS_composite_sans_base.pred_score.rsf.test.txt', header = F)
RS_composite_sans_base.E_train.txt = read.delim('../../RS_composite_sans_base.E_train.txt', header = F)
RS_composite_sans_base.T_train.txt = read.delim('../../RS_composite_sans_base.T_train.txt', header = F)
RS_composite_sans_base.E_test.txt = read.delim('../../RS_composite_sans_base.E_test.txt', header = F)
RS_composite_sans_base.T_test.txt = read.delim('../../RS_composite_sans_base.T_test.txt', header = F)

RS_composite_lab.pred_score.dsv.train.txt = read.delim('../../RS_composite_lab.pred_score.dsv.train.txt', header = F)
RS_composite_lab.pred_score.dsv.test.txt = read.delim('../../RS_composite_lab.pred_score.dsv.test.txt', header = F)
RS_composite_lab.pred_score.cox.train.txt = read.delim('../../RS_composite_lab.pred_score.cox.train.txt', header = F)
RS_composite_lab.pred_score.cox.test.txt = read.delim('../../RS_composite_lab.pred_score.cox.test.txt', header = F)
RS_composite_lab.pred_score.rsf.train.txt = read.delim('../../rsf/RS_composite_lab.pred_score.rsf.train.txt', header = F)
RS_composite_lab.pred_score.rsf.test.txt = read.delim('../../rsf/RS_composite_lab.pred_score.rsf.test.txt', header = F)
RS_composite_lab.E_train.txt = read.delim('../../RS_composite_lab.E_train.txt', header = F)
RS_composite_lab.T_train.txt = read.delim('../../RS_composite_lab.T_train.txt', header = F)
RS_composite_lab.E_test.txt = read.delim('../../RS_composite_lab.E_test.txt', header = F)
RS_composite_lab.T_test.txt = read.delim('../../RS_composite_lab.T_test.txt', header = F)

RS_composite_lab_demo.pred_score.dsv.train.txt = read.delim('../../RS_composite_lab_demo.pred_score.dsv.train.txt', header = F)
RS_composite_lab_demo.pred_score.dsv.test.txt = read.delim('../../RS_composite_lab_demo.pred_score.dsv.test.txt', header = F)
RS_composite_lab_demo.pred_score.cox.train.txt = read.delim('../../RS_composite_lab_demo.pred_score.cox.train.txt', header = F)
RS_composite_lab_demo.pred_score.cox.test.txt = read.delim('../../RS_composite_lab_demo.pred_score.cox.test.txt', header = F)
RS_composite_lab_demo.pred_score.rsf.train.txt = read.delim('../../rsf/RS_composite_lab_demo.pred_score.rsf.train.txt', header = F)
RS_composite_lab_demo.pred_score.rsf.test.txt = read.delim('../../rsf/RS_composite_lab_demo.pred_score.rsf.test.txt', header = F)
RS_composite_lab_demo.E_train.txt = read.delim('../../RS_composite_lab_demo.E_train.txt', header = F)
RS_composite_lab_demo.T_train.txt = read.delim('../../RS_composite_lab_demo.T_train.txt', header = F)
RS_composite_lab_demo.E_test.txt = read.delim('../../RS_composite_lab_demo.E_test.txt', header = F)
RS_composite_lab_demo.T_test.txt = read.delim('../../RS_composite_lab_demo.T_test.txt', header = F)

# dsv.pred = RS_composite_lab.pred_score.dsv.test.txt
# rsf.pred = RS_composite_lab.pred_score.rsf.test.txt
# cox.pred = RS_composite_lab.pred_score.cox.test.txt
# labels = RS_composite_lab.E_test.txt

cali.pipe = function(dsv.pred, rsf.pred, cox.pred, labels, auc.title){
  print(auc.title)
  ## range standardize
  standardized.pred_score.dsv.test = range01((dsv.pred))
  standardized.pred_score.cox.test = range01((cox.pred))
  standardized.pred_score.rsf.test = range01((rsf.pred+0.000001))
  
  
  reliability_diagram(as.vector(RS_composite_lab.E_test.txt[,1]),
                      list(as.vector(standardized.pred_score.dsv.test[,1]),
                           as.vector(standardized.pred_score.rsf.test[,1]),
                           as.vector(standardized.pred_score.cox.test[,1])),
                      # 'H',
                      'C',
                      c('DSV', 'rsf', 'cox'),
                      c('blue', 'red', 'green'),
                      'LR SVM original H')
  
  print('ECE')
  print(ece_mce(labels[,1], standardized.pred_score.dsv.test[,1], 'C'))
  print(ece_mce(labels[,1], standardized.pred_score.rsf.test[,1], 'C'))
  print(ece_mce(labels[,1], standardized.pred_score.cox.test[,1], 'C'))
  
  
  print('ICI')
  print(ici(labels[,1], standardized.pred_score.dsv.test[,1]))
  print(ici(labels[,1], standardized.pred_score.rsf.test[,1]))
  print(ici(labels[,1], standardized.pred_score.cox.test[,1]))
  # browser()
  dsv.auc = auc(roc(as.vector(labels[,1]),as.vector(standardized.pred_score.dsv.test[,1]))) %>% round(., 3)
  rsf.auc = auc(roc(as.vector(labels[,1]),as.vector(standardized.pred_score.rsf.test[,1]))) %>% round(., 3)
  cox.auc = auc(roc(as.vector(labels[,1]),as.vector(standardized.pred_score.cox.test[,1]))) %>% round(., 3)
  
  plot.test = plot(roc(as.vector(labels[,1]),as.vector(standardized.pred_score.dsv.test[,1])),
                   print.auc = F, col = "red", main = auc.title)
  plot.test = plot(roc(as.vector(labels[,1]),as.vector(standardized.pred_score.rsf.test[,1])),
                   print.auc = F, col = "green", lty = 2, print.auc.y = .4,  add = TRUE)
  plot.test = plot(roc(as.vector(labels[,1]),as.vector(standardized.pred_score.cox.test[,1])),
                   print.auc = F, col = "blue",print.auc.y = .4,  add = TRUE)
  legend("bottomright", (c(paste0('DSV \n AUC:',dsv.auc, '\n'), paste0('RSF \n AUC:', rsf.auc), paste0('COX \n AUC:', cox.auc))), lty=c(1,2,1), lwd = 3,
         bty="n", col = c('red', 'green', 'blue'))
}



# cali.pipe(RS_composite_all.pred_score.dsv.test.txt,
#           RS_composite_all.pred_score.rsf.test.txt,
#           RS_composite_all.pred_score.cox.test.txt,
#           RS_composite_all.E_test.txt,
#           'ALL')

cali.pipe(RS_composite_sans_base.pred_score.dsv.test.txt,
          RS_composite_sans_base.pred_score.rsf.test.txt,
          RS_composite_sans_base.pred_score.cox.test.txt,
          RS_composite_sans_base.E_test.txt,
          'Sans Base')

cali.pipe(RS_composite_lab.pred_score.dsv.test.txt,
          RS_composite_lab.pred_score.rsf.test.txt,
          RS_composite_lab.pred_score.cox.test.txt,
          RS_composite_lab.E_test.txt, 
          'Lab')

cali.pipe(RS_composite_lab_demo.pred_score.dsv.test.txt,
          RS_composite_lab_demo.pred_score.rsf.test.txt,
          RS_composite_lab_demo.pred_score.cox.test.txt,
          RS_composite_lab_demo.E_test.txt,
          'Lab Demo')
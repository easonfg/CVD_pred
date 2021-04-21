#library(survcomp)
#library(dplyr)
library(compareC)

RS_cvd_all.pred_score.dsv.train.txt = read.delim('../RS_cvd_all.pred_score.dsv.train.txt', header = F)
RS_cvd_all.pred_score.dsv.test.txt = read.delim('../RS_cvd_all.pred_score.dsv.test.txt', header = F)
RS_cvd_all.pred_score.cox.train.txt = read.delim('../RS_cvd_all.pred_score.cox.train.txt', header = F)
RS_cvd_all.pred_score.cox.test.txt = read.delim('../RS_cvd_all.pred_score.cox.test.txt', header = F)
RS_cvd_all.E_train.txt = read.delim('../RS_cvd_all.E_train.txt', header = F)
RS_cvd_all.T_train.txt = read.delim('../RS_cvd_all.T_train.txt', header = F)
RS_cvd_all.E_test.txt = read.delim('../RS_cvd_all.E_test.txt', header = F)
RS_cvd_all.T_test.txt = read.delim('../RS_cvd_all.T_test.txt', header = F)

#compareC(RS_cvd_all.T_train.txt[,1], RS_cvd_all.E_train.txt[,1], RS_cvd_all.pred_score.dsv.train.txt[,1], RS_cvd_all.pred_score.cox.train.txt[,1])
#$est.c
#      Cxy       Cxz
#      0.1797215 0.2021295
#
#      $est.diff_c
#      [1] -0.02240805
#
#      $est.vardiff_c
#      [1] 2.240864e-07
#
#      $est.varCxy
#      [1] 1.200641e-06
#
#      $est.varCxz
#      [1] 1.413775e-06
#
#      $est.cov
#      [1] 1.195164e-06
#
#      $zscore
#      [1] -47.33652
#
#      $pval
#      [1] 0
#compareC(RS_cvd_all.T_test.txt[,1], RS_cvd_all.E_test.txt[,1], RS_cvd_all.pred_score.dsv.test.txt[,1], RS_cvd_all.pred_score.cox.test.txt[,1])

RS_cvd_sans_base.pred_score.dsv.train.txt = read.delim('../RS_cvd_sans_base.pred_score.dsv.train.txt', header = F)
RS_cvd_sans_base.pred_score.dsv.test.txt = read.delim('../RS_cvd_sans_base.pred_score.dsv.test.txt', header = F)
RS_cvd_sans_base.pred_score.cox.train.txt = read.delim('../RS_cvd_sans_base.pred_score.cox.train.txt', header = F)
RS_cvd_sans_base.pred_score.cox.test.txt = read.delim('../RS_cvd_sans_base.pred_score.cox.test.txt', header = F)
RS_cvd_sans_base.E_train.txt = read.delim('../RS_cvd_sans_base.E_train.txt', header = F)
RS_cvd_sans_base.T_train.txt = read.delim('../RS_cvd_sans_base.T_train.txt', header = F)
RS_cvd_sans_base.E_test.txt = read.delim('../RS_cvd_sans_base.E_test.txt', header = F)
RS_cvd_sans_base.T_test.txt = read.delim('../RS_cvd_sans_base.T_test.txt', header = F)
print('SANS BASE')
compareC(RS_cvd_sans_base.T_train.txt[,1], RS_cvd_sans_base.E_train.txt[,1], RS_cvd_sans_base.pred_score.dsv.train.txt[,1], RS_cvd_sans_base.pred_score.cox.train.txt[,1])
compareC(RS_cvd_sans_base.T_test.txt[,1], RS_cvd_sans_base.E_test.txt[,1], RS_cvd_sans_base.pred_score.dsv.test.txt[,1], RS_cvd_sans_base.pred_score.cox.test.txt[,1])

RS_cvd_lab.pred_score.dsv.train.txt = read.delim('../RS_cvd_lab.pred_score.dsv.train.txt', header = F)
RS_cvd_lab.pred_score.dsv.test.txt = read.delim('../RS_cvd_lab.pred_score.dsv.test.txt', header = F)
RS_cvd_lab.pred_score.cox.train.txt = read.delim('../RS_cvd_lab.pred_score.cox.train.txt', header = F)
RS_cvd_lab.pred_score.cox.test.txt = read.delim('../RS_cvd_lab.pred_score.cox.test.txt', header = F)
RS_cvd_lab.E_train.txt = read.delim('../RS_cvd_lab.E_train.txt', header = F)
RS_cvd_lab.T_train.txt = read.delim('../RS_cvd_lab.T_train.txt', header = F)
RS_cvd_lab.E_test.txt = read.delim('../RS_cvd_lab.E_test.txt', header = F)
RS_cvd_lab.T_test.txt = read.delim('../RS_cvd_lab.T_test.txt', header = F)
print('lab')
compareC(RS_cvd_lab.T_train.txt[,1], RS_cvd_lab.E_train.txt[,1], RS_cvd_lab.pred_score.dsv.train.txt[,1], RS_cvd_lab.pred_score.cox.train.txt[,1])
compareC(RS_cvd_lab.T_test.txt[,1], RS_cvd_lab.E_test.txt[,1], RS_cvd_lab.pred_score.dsv.test.txt[,1], RS_cvd_lab.pred_score.cox.test.txt[,1])


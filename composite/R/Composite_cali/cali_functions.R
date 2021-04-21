range01 <- function(x){(x-min(x))/(max(x)-min(x))}
range.n1to1 <- function(x){(2*(x-min(x))/(max(x)-min(x)))-1}

ici = function(Y, P){
  # Y = c(rep(0, 9), rep(1,1), rep(0,8), rep(1, 2), rep(0,7),rep(1,3), rep(0,6),rep(1,4), rep(0,5), rep(1,5), rep(0,4),rep(1,6),
  #       rep(0,3),rep(1,7), rep(0,2), rep(1,8), rep(0,1), rep(1,9))
  # # P = c(rep(0.1, 10), rep(0.2, 10),rep(0.3, 10),rep(0.4, 10),rep(0.5, 10),rep(0.6, 10),rep(0.7, 10),rep(0.8, 10),rep(0.9, 10))
  # P = c(rep(0.3, 10), rep(0.3, 10),rep(0.3, 10),rep(0.4, 10),rep(0.5, 10),rep(0.6, 10),rep(0.7, 10),rep(0.8, 10),rep(0.9, 10))
  # browser()
  loess.calibrate <- loess(Y ~ P)
  # 
  # Estimate loess‐based smoothed calibration curve
  P.calibrate <- predict (loess.calibrate, newdata = P)
  
  # This is the point on the loess calibration curve corresponding to a given predicted probability.
  ICI <- mean (abs(P.calibrate - P))
  # browser()
  # quartz('test')
  # plot(P, P.calibrate)
  # # plot(main="Loess Smoothing and Prediction", xlab="Date", ylab="Unemployment (Median)")
  # lines(P.calibrate, x=P)
  return(ICI)
}


ece_mce = function(y, prob, stat_type){
  m = sum(y)
  n = length(y)
  #browser()
  g = max(10, min(m/2, (n-m)/2, 2+8*((n/1000)**2)))
  cat('g', g, '\n')
  
  mtx = cbind(y, y_not = 1- y, prob, prob_not = 1-prob)
  mtx = as.data.frame(mtx)
  mtx = mtx[order(mtx$prob),]
  n <- length(prob)/g
  nr <- nrow(mtx)
  
  if (stat_type == 'C'){
    split_mtx = split(mtx, rep(1:ceiling(nr/n), each=n, length.out=nr))
  }else{ ### H statistics, equal intervals
    split_mtx = split(mtx, cut(mtx$prob, seq(0,1,1/g), include.lowest=TRUE))
    split_mtx = split_mtx[sapply(split_mtx, nrow)>0]
    ###
  }
  #split_mtx = split(mtx, cut(mtx$prob, seq(0,1,1/g), include.lowest=TRUE))
  #split_mtx = split_mtx[sapply(split_mtx, nrow)>0]
  ###
  
  H_stat = c()
  for (i in 1:length(split_mtx)){
    #obs = sum(split_mtx[[i]][split_mtx[[i]]$y == 1,]$y)
    obs = mean(split_mtx[[i]]$y == 1)
    #exp = sum(split_mtx[[i]][split_mtx[[i]]$y == 1,]$prob)
    exp = mean(split_mtx[[i]]$prob)
    # #obs_not = length(split_mtx[[i]][split_mtx[[i]]$y == 0,]$y)
    # obs_not = sum(split_mtx[[i]]$y == 0)
    # #exp_not = sum(split_mtx[[i]][split_mtx[[i]]$y == 0,]$prob_not)
    # exp_not = sum(split_mtx[[i]]$prob_not)
    
    
    
    
    H_stat = c(H_stat, abs(obs - exp))
    
  }
  return(list(ece = sum(H_stat)/length(split_mtx), mce = max(H_stat)))
}

reliability_datapts <- function(obs, pred, bins=10, stat_type ='H') {
  min.pred <- min(pred)
  max.pred <- max(pred)
  min.max.diff <- max.pred - min.pred
  
  if (stat_type == 'H'){
    mtx = cbind(obs, pred)
    mtx = as.data.frame(mtx)
    mtx = mtx[order(mtx$pred),]
    res = data.frame(V1= numeric(0), V2 = numeric(0))
    split_mtx = split(mtx, cut(mtx$pred, seq(0,1,1/10), include.lowest=TRUE))
    # split_mtx = split(mtx, cut(mtx$pred, seq(0,1,1/bins), include.lowest=TRUE))
    
    for (i in 1:length(split_mtx)){
      col_mean = colMeans(split_mtx[[i]])
      if (sum(is.na(col_mean)) > 0) {
        next
      }
      res[i,] = col_mean
    }
    
  }else{
    ## C statistics, same number of instances in each bin
    mtx = cbind(obs, pred)
    mtx = as.data.frame(mtx)
    mtx = mtx[order(mtx$pred),]
    n <- length(pred)/10
    # n <- length(pred)/bins
    nr <- nrow(mtx)
    split_mtx = split(mtx, rep(1:ceiling(nr/n), each=n, length.out=nr))
    res = data.frame(V1= numeric(0), V2 = numeric(0))
    for (i in 1:length(split_mtx)){
      res[i,] = colMeans(split_mtx[[i]])
    }
  }
  
  return(res)
}

reliability_diagram = function(obs_rep, data_ls, stat_type,
                               title_ls, 
                               color_ls, title) {
  
  data_ls_len = length(data_ls)
  
  m = sum(obs_rep)
  n = length(obs_rep)
  # browser()
  g = max(10, min(m/2, (n-m)/2, 2+8*((n/1000)**2))) %>% round()
  # cat('g', g, '\n')
  
  ### create bin averages
  for (data_i in 1:data_ls_len){
    temp_res = reliability_datapts(obs_rep, data_ls[[data_i]], bins = g, stat_type = stat_type)
    assign(paste("recal_bins", data_i, sep=""),temp_res)
  }
  
  for (data_i in 1:data_ls_len){
    temp_res = melt(get(paste('recal_bins', data_i, sep='')), id="V2")
    temp_res[, "variable"] <- paste("Vol.x", data_i, sep='')
    assign(paste('melt', data_i, sep=''), temp_res)
  }
  
  data = melt1
  if (data_ls_len > 1){
    for (data_i in 2:data_ls_len){
      data = rbind(data, get(paste('melt', data_i, sep='')))
    }
  }
  
  line_plot = ggplot(data, aes(x=V2,  y=value, color=variable))   +  geom_point()+ geom_line() +
    scale_color_manual(labels = title_ls,
                       values = color_ls) +
    guides(color=guide_legend(" ")) + 
    ggtitle(paste(stat_type, 'Statistics')) +
    xlab("Mean Predicted") + ylab("Mean Observed") + 
    xlim(0, 1) + ylim(-0.05, 1.05)
  
  line_plot = line_plot + geom_abline(intercept = 0, slope = 1, color="black",
                                      linetype="dashed", size=1) 
  #remove background
  line_plot = line_plot + theme_bw()
  line_plot = line_plot + theme(legend.position="bottom")
  
  ## add data points
  for (data_i in 1:data_ls_len){
    temp_obs = obs_rep
    temp_obs[temp_obs==0] = -0.005 * data_i
    temp_obs[temp_obs==1] = 1 + 0.005 * data_i
    assign(paste('obs_rep_offset', data_i, sep=''), data.frame(cbind(data_ls[[data_i]], temp_obs)))
  }
  
  for (data_i in 1:data_ls_len){
    temp_res = melt(get(paste('obs_rep_offset', data_i, sep='')), id="temp_obs")
    temp_res[, "variable"] <- paste("Vol.x", data_i, sep='')
    assign(paste('obs_rep_offset', data_i, sep=''), temp_res)
  }
  
  data_points = obs_rep_offset1
  if (data_ls_len > 1){
    for (data_i in 2:data_ls_len){
      data_points = rbind(data_points, get(paste('obs_rep_offset', data_i, sep='')))
    }
  }
  
  line_plot = line_plot + geom_point(data = data_points, aes(x=data_points$value,  y=data_points$temp_obs, color=variable)) + geom_point(alpha=0.2)
  
  
  print(line_plot)
  #ggsave(title, device = 'png', width = 6, height = 6)
}
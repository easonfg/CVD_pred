def CI(x, sigma2, C=95):
  dx = stats.norm.ppf((1 + C/100) / 2, loc = 0, scale = 1) * np.sqrt(sigma2)
  return x - dx, x + dx

def c_index_bootstrap(T, E, scores, alpha=0.05, n_bootstrap=10, random_state=None):
  np.random.seed(random_state)

  c_index = concordance_index(T, -scores, E)
  
  stacked_arr = np.stack((T, -scores, E), axis=-1)
  c_idx_boot = np.array([concordance_index(*resample(stacked_arr).T) for _ in range(n_bootstrap)])

  c_index_boot = c_idx_boot.mean()
  sigma2 = c_idx_boot.var()
  CI_lower = np.quantile(c_idx_boot, alpha/2)
  CI_upper = np.quantile(c_idx_boot, 1 - alpha/2)

  return c_index, c_index_boot, sigma2, CI_lower, CI_upper

def c_index_jackknife(T, E, scores, alpha=0.05):

  c_index = concordance_index(T, -scores, E)
  
  stacked_arr = np.stack((T, -scores, E), axis=-1)
  n = stacked_arr.shape[0]
  c_idx_jack = np.array([concordance_index(*np.delete(stacked_arr, i, axis=0).T) for i in range(n)])
  
  c_index_jack = c_idx_jack.mean()
  sigma2 = (n-1) * c_idx_jack.var()
  CI_lower, CI_upper = CI(c_index_jack, sigma2, C=95)

  return c_index, c_index_jack, sigma2, CI_lower, CI_upper
  
def c_diff_bootstrap(T, E, scores1, scores2, alpha=0.05, n_bootstrap=10, random_state=None):

  np.random.seed(random_state)
  c1 = concordance_index(T, -scores1, E)
  c2 = concordance_index(T, -scores2, E)
  dc = c1 - c2

  stacked_arr = np.stack((T, -scores1, -scores2, E), axis=-1)
    
  def c_diff(data):
    sample = resample(data)
    return concordance_index(*sample[:, [0,1,3]].T) - concordance_index(*sample[:, [0,2,3]].T)

  dc_sample = np.array([c_diff(stacked_arr) for _ in range(n_bootstrap)])

  dc_boot = dc_sample.mean()
  sigma2 = dc_sample.var()
  CI_lower = np.quantile(dc_sample, alpha/2)
  CI_upper = np.quantile(dc_sample, 1 - alpha/2)

  return dc, dc_boot, sigma2, CI_lower, CI_upper
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import ticker, colors, cm

def stat_analyze(data_x, y):
    data_x = data_x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    
    x = np.ones((data_x.shape[0], 2))
    x[:,0] = data_x
    
    glm = sm.GLM(y, x, family=sm.families.Binomial(link=sm.families.links.Logit()))
    fit = glm.fit()
    print(fit.cov_params())
    print(fit.summary())
    
    return fit

def plot_points(data_x, y):
    data_x = data_x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    plt.scatter(data_x, y)
    plt.ylabel('Detected?', fontsize=16)
    plt.xlabel('Size', fontsize=16)
    plt.grid(True)
    plt.savefig('tmp_imgs/simple_data_pod/detection.png')
    
def draw_segm_ratio(ax, x, tg, pred, alpha=1.):
    bins = np.linspace(x.min(), x.max(), 20)
    bin_width = bins[1]-bins[0]
    #print(bins)
    
    unique_pred = np.unique(pred)
    print(unique_pred)
    tg_ind = np.argwhere(unique_pred == tg[0])
    unique_pred[tg_ind] = unique_pred[0]
    unique_pred[0] = tg[0]
    print(unique_pred)
    fract = np.zeros((unique_pred.shape[0], bins.shape[0]))
    
    for i in range(bins.shape[0]-1):
        mask = np.logical_and(x >= bins[i], x < bins[i+1])
        total = np.count_nonzero(x[mask])
        unique_counts = np.array([np.count_nonzero(pred[mask] == val) for val in unique_pred]).astype(np.float32)
        if total != 0:
            fract[:,i] =  unique_counts / total
        else:
            fract[:,i] = 1.
        
    fract[:,-1] = fract[:,-2]
    
    y_offset = np.zeros_like(bins)
    for i in range(unique_pred.shape[0]):
        ax.bar(bins, fract[i,:], bin_width, bottom=y_offset, align='edge', alpha=alpha, label = 'Pr. Class {:d}'.format(int(unique_pred[i])))
        y_offset += fract[i,:]
        
    ax.set_xlabel('Defect size', fontsize=16)
    ax.set_ylabel("Detection ratio", fontsize=16)
    ax.legend(loc = 4, fontsize=16)
    #ax.set_ylim(0., 1.)
    #ax.set_xlim(0., 2.5)
    #ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.grid(True)
    
def compute_s90(fit, x_range=np.linspace(0., 3., 1000)):
    '''Computes s_90 - defect size for 90\% probability of good segmentation and s_90/95 - lower bound of the same value for 95% confidence interval.
    It is possiblee that one value or both do not exist. The function will return in this case.
    
    :param fit: Statsmodels object with fit results
    :type fit: :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    :param x_range: Range of defect size to compute probabilities
    :type x_range: :class:`np.ndarray`

    :return: List containing s_90 and s_90/95. If any does not exist, it is replaced with -1.
    :rtype: :class:`list`
    '''
    fit_x = np.ones((x_range.shape[0], 2))
    fit_x[:,0] = x_range
    prediction = fit.get_prediction(fit_x)
    fit_y = prediction.summary_frame(alpha=0.05)
    
    p = fit_y['mean']
    p_low = fit_y['mean_ci_lower']
    p_high = fit_y['mean_ci_upper']
    
    res = [-1., -1.]
    s90_95_exists = np.any(p_low > 0.9)
    s90_exists = np.any(p > 0.9)
    if s90_exists:
        s90 = x_range[np.where(p > 0.9)].min()
        res[0] = s90
    if s90_95_exists:
        s90_95 = x_range[np.where(p_low > 0.9)].min()
        res[1] = s90_95
    return res
    
def draw_pod(ax, fit, res=None, NN_confidence=False, default_x_range=np.linspace(0., 3., 1000), 
             draw_confidence_interval=True, draw_s90=False, label='POD', colors = ['r', 'b'], linestyle='-', linewidth=1.5):
    '''Draws POD curve on ax based on fit parameters
    s_90 and s_90/95 are drawn if they exist in this size range
    
    :param ax: Subplot axes to draw on
    :type ax: :class:`matplotlib.axes._subplots.AxesSubplot`
    :param fit: Statsmodels object with fit results
    :type fit: :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    :param res: Array with model results. Only used to get range of FO size. If None, default_x_range will be used instead
    :type res: :class:`np.ndarray`
    :param NN_confidence: Draw confidence intervals based on the variance of different instances of the same model
    :type NN_confidence: :class:`bool`
    :param default_x_range: Default range of defect size to use if res array is not provided
    :type default_x_range: :class:`np.ndarray`
    :param draw_confidence_interval: Set to True to draw confidence interval
    :type draw_confidence_interval: :class:`bool`
    :param draw_s90: Set to True to draw s_90 and s_90/95
    :type draw_s90: :class:`bool`
    :param label: Label to use in legend
    :type label: :class:`str`
    :param colors: Colors to use for the curve and confidence interval
    :type colors: :class:`list`
    :param linestyle: Linestyle for POD curve
    :type linestyle: :class:`str`
    
    '''    
    x_range = default_x_range
    if res is not None:
        x = res
        x_range = np.linspace(x.min(), x.max(), 1000)
    
    if not NN_confidence:
        fit_x = np.ones((x_range.shape[0], 2))
        fit_x[:,0] = x_range
        prediction = fit.get_prediction(fit_x)
        fit_y = prediction.summary_frame(alpha=0.05)
        
        p = fit_y['mean']
        p_low = fit_y['mean_ci_lower']
        p_high = fit_y['mean_ci_upper']
    else:
        p, p_low, p_high = compute_confidence_NN(res, x_range, fit)
        
    s90, s90_95 = compute_s90(fit, x_range=x_range)
    s90_exists = True if s90 > -1 else False
    s90_95_exists = True if s90_95 > -1 else False
    
    if draw_confidence_interval:
        #ax.fill_between(x_range, p_low, p_high, color=colors[1], alpha = 0.2, label = '{} 95% confidence'.format(label))
        ax.fill_between(x_range, p_low, p_high, color=colors[1], alpha = 0.2)
    ax.plot(x_range, p, c=colors[0], label = label, linestyle=linestyle, linewidth=linewidth)
    if draw_s90 and s90_95_exists:
        ax.vlines([s90, s90_95], 0., 0.9, linestyles='--', color='k')
        ax.scatter([s90, s90_95], [0.9, 0.9], color='k', s=20)
        plt.scatter(s90_95, 0.9, color='g', s=30, label = label + ' ' + r"$s_{90/95\%}$")
    if draw_s90 and s90_exists:
        ax.vlines([s90], 0., 0.9, linestyles='--', color='k')
        ax.scatter([s90], [0.9], color='k', s=20)
        plt.scatter(s90, 0.9, color='k', s=30, label = label + ' ' + r"$s_{90}$")
        
    ax.set_ylim(0., 1.)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax.grid(True)
    ax.legend(loc = 4, fontsize=16)
    ax.set_xlabel('FO size', fontsize=20)
    ax.set_ylabel("Probability of Detection", fontsize=20)

def plot_pod(fit, x, tg, pred):
    fig, ax = plt.subplots(1, 2, figsize=(16,9))
    
    x = x.detach().cpu().numpy()
    tg = tg.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    
    draw_segm_ratio(ax[0], x, tg, pred, alpha=1.)
    draw_pod(ax[1], fit, x)
    
    plt.savefig('tmp_imgs/simple_data_pod/pod.png')

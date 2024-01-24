import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import ticker, colors, cm

def data_transform(x, mean, std):
    
    transformed = x
    return transformed

def stat_analyze(data_x, y):
    x = np.ones((data_x.shape[0], 2))
    x[:,0] = data_transform(data_x, data_x.mean(), data_x.std())
    
    glm = sm.GLM(y, x, family=sm.families.Binomial(link=sm.families.links.Logit()))
    fit = glm.fit(start_params = [0, 0])
    #print(fit.params)
    #print(fit.cov_params())
    #print(fit.aic)
    #print(fit.summary())
    
    #x_range = np.linspace(data_x.min(), data_x.max(), 1000)
    x_range = np.linspace(0., 0.5, 1000)
    mean = x_range * fit.params[0] + fit.params[1]
    sigma = np.sqrt(np.power(x_range, 2) * fit.cov_params()[0,0] + 2 * x_range * fit.cov_params()[0,1] + fit.cov_params()[1,1])
    
    if False:
        plt.clf()
        
        fig, ax = plt.subplots(1, 2, figsize = (16, 9))
        ax[0].plot(x_range, mean)
        ax[0].set_xlabel('Signal')
        ax[0].set_ylabel('log p / (1-p)')
        ax[0].grid(True)
        ax[1].plot(x_range, sigma / mean)
        ax[1].set_xlabel('Signal')
        ax[1].set_ylabel('Logit relative error')
        ax[1].grid(True)
        plt.show()
    
    return fit

def plot_points(data_x, y):
    fig, ax = plt.subplots(1,1, figsize=(12,9))
    
    ax.scatter(data_x, y, alpha=0.5)
    ax.set_ylabel('Is the prediction correct?', fontsize=20)
    ax.set_xlabel('Size', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    
    plt.savefig('tmp_imgs/chicken_data_pod/detection.png')
    
def correlate_size_snr(fo_size, snr):
    fig, ax = plt.subplots(1,1, figsize=(12,9))
    
    ax.scatter(1/fo_size, snr)
    ax.set_xlabel('1 / Attenuation at 40kV', fontsize=16)
    ax.set_ylabel('Quotient dR', fontsize=16)
    ax.grid(True)
    plt.savefig('tmp_imgs/chicken_data_pod/size_contrast.png')
    
def draw_histo(ax, x):
    bins = np.linspace(x.min(), x.max(), 10)
    bin_width = bins[1]-bins[0]
    print(bins)
        
    total = np.zeros((bins.shape[0]))
    for i in range(bins.shape[0]-1):
        mask = np.logical_and(x >= bins[i], x < bins[i+1])
        total[i] = np.count_nonzero(x[mask])
        
    total[-1] = total[-2]
    
    ax.bar(bins, total, bin_width, align='edge')
        
    ax.set_xlabel('FO Size', fontsize=16)
    ax.set_ylabel("Number of samples", fontsize=16)
    #ax.legend(loc = 4, fontsize=16)
    #ax.set_ylim(0., 1.)
    #ax.set_xlim(0., 3.)
    #ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.grid(True)

def plot_histo(x):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    
    draw_histo(ax, x)
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/chicken_data_pod/histo.png')
    
def draw_segm_ratio(ax, x, tg, pred, alpha=1.):
    bins = np.linspace(x.min(), x.max(), 10)
    bin_width = bins[1]-bins[0]
    print(bins)
    
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
        print(i)
        print(total)
        print(unique_counts)
        if total != 0:
            fract[:,i] =  unique_counts / total
        else:
            fract[:,i] = 0.
        
    fract[:,-1] = fract[:,-2]
    
    y_offset = np.zeros_like(bins)
    for i in range(unique_pred.shape[0]):
        ax.bar(bins, fract[i,:], bin_width, bottom=y_offset, align='edge', alpha=alpha, label = 'Pr. Class {:d}'.format(int(unique_pred[i])))
        y_offset += fract[i,:]
        
    ax.set_xlabel('Quotient FO Signal', fontsize=16)
    ax.set_ylabel("Detection ratio", fontsize=16)
    ax.legend(loc = 4, fontsize=16)
    #ax.set_ylim(0., 1.)
    #ax.set_xlim(0., 2.5)
    #ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.grid(True)
    
def compute_s90(fit, x_range=np.linspace(0., 0.5, 1000)):
    '''Computes s_90 - defect size for 90\% probability of good segmentation 
    It is possiblee that one value or both do not exist. The function will return -1 in this case.
    
    :param fit: Statsmodels object with fit results
    :type fit: :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    :param x_range: Range of defect size to compute probabilities
    :type x_range: :class:`np.ndarray`

    :return: List containing s_90 and 
    :rtype: :class:`list`
    '''
    fit_x = np.ones((x_range.shape[0], 2))
    fit_x[:,0] = x_range
    prediction = fit.get_prediction(fit_x)
    fit_y = prediction.summary_frame(alpha=0.05)
    #print(fit_y)
    
    p = fit_y['mean']
    p_low = fit_y['mean_ci_lower']
    p_high = fit_y['mean_ci_upper']
    #print(p_high)
    
    res = [-1., -1., -1]
    lower_exists = np.any(p_low > 0.9)
    upper_exists = np.any(p_high > 0.9)
    s90_exists = np.any(p > 0.9)
    if s90_exists:
        s90 = x_range[np.where(p > 0.9)].min()
        res[0] = s90
    if lower_exists:
        lower = x_range[np.where(p_low > 0.9)].min()
        res[1] = lower
    if upper_exists:
        upper = x_range[np.where(p_high > 0.9)].min()
        res[2] = upper
    return res

def compute_curve(fit, x):
    #print(fit.params)
    #print(fit.cov_params())
    
    x_range = np.linspace(x.min(), x.max(), 1000)
    l = (fit.params[0] - 60) * x_range + fit.params[1]
    
    p = np.exp(l) / (1 + np.exp(l))
    
    return p

def compute_residuals(fit, par, y):
    x = np.copy(par)
    #x = np.sort(x)
    #x_range = np.linspace(x.min(), x.max(), 1000)
    #x_range = np.linspace(0., 0.5, 1000)
    
    fit_x = np.ones((x.shape[0], 2))
    fit_x[:,0] = data_transform(x, x.mean(), x.std())
    prediction = fit.get_prediction(fit_x)
    fit_y = prediction.summary_frame(alpha=0.05)
    p = fit_y['mean']
    
    r = y - p
    
    return r
    
def draw_pod(ax, fit, par, xlabel,
             draw_confidence_interval=True, draw_s90=False, label=None, colors = ['r', 'b'], linestyle='-', linewidth=1.5, pod_alpha = 1., xlabel_size = 20, ylabel_size = 20):
    '''Draws POD curve on ax based on fit parameters
    s_90 and s_90/95 are drawn if they exist in this size range
    
    :param ax: Subplot axes to draw on
    :type ax: :class:`matplotlib.axes._subplots.AxesSubplot`
    :param fit: Statsmodels object with fit results
    :type fit: :class:`statsmodels.genmod.generalized_linear_model.GLMResultsWrapper`
    
    '''    
    x = par
    #x_range = np.linspace(x.min(), x.max(), 1000)
    x_range = np.linspace(0., 0.5, 1000)
    
    fit_x = np.ones((x_range.shape[0], 2))
    #fit_x[:,0] = x_range
    fit_x[:,0] = data_transform(x_range, x.mean(), x.std())
    prediction = fit.get_prediction(fit_x)
    fit_y = prediction.summary_frame(alpha=0.05)
    #print(fit_y)
    
    p2 = compute_curve(fit, x)
        
    p = fit_y['mean']
    p_low = fit_y['mean_ci_lower']
    p_high = fit_y['mean_ci_upper']

    s90, s90_95, _ = compute_s90(fit, x_range=x_range)
    s90_exists = True if s90 > -1 else False
    s90_95_exists = True if s90_95 > -1 else False
    #print(p)
    if draw_confidence_interval:
        #ax.fill_between(x_range, p_low, p_high, color=colors[1], alpha = 0.2, label = '{} 95% confidence'.format(label))
        ax.fill_between(x_range, p_low, p_high, color=colors[1], alpha = 0.2)
    ax.plot(x_range, p, c=colors[0], label = label, linestyle=linestyle, linewidth=linewidth, alpha=pod_alpha)
    #ax.plot(x_range, p2, c='g', label = 'Manual compute', linestyle=linestyle, linewidth=linewidth)
    if draw_s90 and s90_95_exists:
        ax.vlines([s90, s90_95], 0., 0.9, linestyles='--', color='k')
        ax.scatter([s90, s90_95], [0.9, 0.9], color='k', s=20)
        ax.scatter(s90_95, 0.9, color='g', s=30, label = label + ' ' + r"$s_{90/95\%}$")
    if draw_s90 and s90_exists:
        ax.vlines([s90], 0., 0.9, linestyles='--', color='k')
        ax.scatter([s90], [0.9], color='k', s=20)
        ax.scatter(s90, 0.9, color='k', s=30, label = label + ' ' + r"$s_{90}$")
        print('s90', s90)
        
    #ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.yaxis.set_major_locator(plt.LinearLocator(11))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax.grid(True)
    #ax.legend(loc = 4, fontsize=16)
    #ax.set_xlabel('FO size, mm', fontsize=16)
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    ax.set_ylabel("Probability of Detection", fontsize=ylabel_size)

def plot_fract_pod(fit, x, tg, pred):
    fig, ax = plt.subplots(1, 2, figsize=(16,9))
    
    draw_segm_ratio(ax[0], x, tg, pred, alpha=1.)
    draw_pod(ax[1], fit, x, 'Quotient FO Signal')
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/chicken_data_pod/fract_pod.png')
    
def plot_pod(fname, xlabel, fit, x, tg, pred):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    
    #draw_pod(ax, fit, x, 'Quotient dR')
    draw_pod(ax, fit, x, xlabel)
    if xlabel == 'Quotient FO Signal':
        ax.set_xlim(0., 0.2)
    elif xlabel == 'Attenuation at 40kV':
        ax.set_xlim(0., 0.9)
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/chicken_data_pod/{}.png'.format(fname))
    
def comp_pod_arg(fit_snr, snr, fit_size, fo_size, tg, pred):
    fig, ax = plt.subplots(1, 2, figsize=(16,7))
    
    draw_pod(ax[0], fit_snr, snr, 'Foreign object SNR')
    draw_pod(ax[1], fit_size, fo_size, 'Foreign object size')
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/chicken_data_pod/comp_size_snr.pdf', format='pdf')

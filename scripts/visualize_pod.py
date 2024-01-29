import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import ticker, colors, cm
import statsmodels
import scipy
import pod2settings as pts
    
def comparison_part(ax, fname, legend_loc = 2, **pod_args):
    res_arr = np.genfromtxt(fname, delimiter=',', names=True)
    contrast = res_arr['Contrast']
    correct_det = np.where(res_arr['Prediction'] == res_arr['Target'], 1, 0)
    
    try:
        fit_contrast = pts.pod.stat_analyze(contrast, correct_det)
        pts.pod.draw_pod(ax, fit_contrast, contrast, 'Image Contrast', **pod_args)
        ax.set_xlim(0., 0.25)
        #ax.legend(loc=legend_loc, fontsize=14)
        return fit_contrast
    except statsmodels.tools.sm_exceptions.PerfectSeparationError:
        print('Perfect Separation')
        return -1
                
def plot_points(ax, fname):
    res_arr = np.genfromtxt(fname, delimiter=',', names=True)
    contrast = res_arr['Contrast']
    correct_det = np.where(res_arr['Prediction'] == res_arr['Target'], 1, 0)
    
    ax.scatter(contrast, correct_det, alpha=0.5)
    ax.set_ylabel('Is the prediction correct?', fontsize=20)
    ax.set_xlabel('Image Contrast', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(0., 0.2)
    ax.grid(True)
        
def plot_pod(res_folder, name):
    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    
    default_num = 5
    plot_points(ax[0], res_folder / '{}_train{}_{:02d}-test{}.csv'.format(name['arch'], name['train'], default_num, name['test']))
    comparison_part(ax[1], res_folder / '{}_train{}_{:02d}-test{}.csv'.format(name['arch'], name['train'], default_num, name['test']))
        
    plt.tight_layout()
    plt.show()
    plt.savefig('tmp_imgs/chicken_data_pod/pod_contrast.pdf', format='pdf')
    
def compute_network_variance(res_folder, name):
    k_list = []
    b_list = []
    
    for i in range(0, 100):
        fname = res_folder / '{}_train{}_{:02d}-test{}.csv'.format(name['arch'], name['train'], i, name['test'])
        res_arr = np.genfromtxt(fname, delimiter=',', names=True)
        contrast = res_arr['Contrast']
        correct_det = np.where(res_arr['Prediction'] == res_arr['Target'], 1, 0)
        try:
            fit = pts.pod.stat_analyze(contrast, correct_det)
            k, b = fit.params
            k_list.append(k)
            b_list.append(b)
        except statsmodels.tools.sm_exceptions.PerfectSeparationError:
            print('Perfect Separation')
            
    k_list = np.array(k_list)
    b_list = np.array(b_list)
    
    k = k_list.mean()
    b = b_list.mean()
    par = np.array([k, b])
    cov = np.zeros((2, 2))
    cov[0,0] = ( (k_list - k_list.mean())*(k_list - k_list.mean()) ).mean()
    cov[1,0] = ( (b_list - b_list.mean())*(k_list - k_list.mean()) ).mean()
    cov[0,1] = ( (k_list - k_list.mean())*(b_list - b_list.mean()) ).mean()
    cov[1,1] = ( (b_list - b_list.mean())*(b_list - b_list.mean()) ).mean()

    print(par)
    print(cov)
    
    return par, cov, fit

def plot_average_network(ax, res_folder, name, label, colors=['r', 'b'], plot_uncertainty=True):
    par, cov, fit = compute_network_variance(res_folder, name)
    
    alpha = 0.05
    x_range = np.linspace(0., 0.25, 100)
    fit_x = np.ones((x_range.shape[0], 2))
    fit_x[:,0] = x_range
        
    fit_y = fit_x @ par
    var = fit_x @ cov @ fit_x.T
    print(var.shape)
    
    var_y = var.diagonal()
    se = np.sqrt(var_y)
    q = scipy.stats.norm.ppf(1 - alpha / 2.)
    lower = fit_y - q * se
    upper = fit_y + q * se
        
    p = fit.family.link.inverse(fit_y)
    p_low = fit.family.link.inverse(lower)
    p_high = fit.family.link.inverse(upper)
    
    if plot_uncertainty:
        ax.fill_between(x_range, p_low, p_high, color = colors[1], alpha = 0.2)
    ax.plot(x_range, p, color = colors[0], label=label)

def compare_average_networks(res_folder, name, add_real):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    
    plot_average_network(ax, res_folder, name, label = 'Generated data', colors=['r', 'b'])
    plot_average_network(ax, res_folder, add_real, label = 'Real data', colors=['g', 'b'])
    
    plt.grid(True)
    plt.legend(fontsize = 16)
    plt.tight_layout()
    plt.savefig('tmp_imgs/chicken_data_pod/average_comparison.pdf', format='pdf')
    
    
def network_pods(ax, res_folder, add_gen=None, add_real=None, **pod_args):
    res = [0., 0.]
    
    if add_gen:
        s_list = []
        s2_list = []
        for i in range(0, 100):
            flag = False
            c_list = ['r', 'b']
            
            if i == 0:
                fit = comparison_part(ax, res_folder / '{}_train{}_{:02d}-test{}.csv'.format(add_gen['arch'], add_gen['train'], i, add_gen['test']), draw_confidence_interval=False, colors = c_list, pod_alpha=0.2, label='Test on generated data', **pod_args)
            else:
                fit = comparison_part(ax, res_folder / '{}_train{}_{:02d}-test{}.csv'.format(add_gen['arch'], add_gen['train'], i, add_gen['test']), draw_confidence_interval=False, colors = c_list, pod_alpha=0.2, **pod_args)
            if fit != -1:
                s90, _, _ = pts.pod.compute_s90(fit)
                s_list.append(s90)
                #print(i, s90)
                
        s_list = np.array(s_list)
        s_mean = s_list.mean()
        s_se = s_list.std()
        alpha = 0.05
        q = scipy.stats.norm.ppf(1 - alpha / 2.)
        print('On generated test data:')
        print('s_90 = {:.2f} +- {:.2f}'.format(s_mean, s_se))
        print(s_mean - q*s_se, s_mean, s_mean + q*s_se)
        ax.vlines([s_mean, s_mean - q*s_se, s_mean + q*s_se], 0., 0.9, linestyles='--', color='r')
        ax.scatter([s_mean, s_mean - q*s_se, s_mean + q*s_se], [0.9, 0.9, 0.9], color='r', s=16)
        res = [s_mean, s_se]
            
    if add_real:
        s_list = []
        s2_list = []
        for i in range(0, 100):
            c_list = ['g', 'b']
            if i==0:
                fit = comparison_part(ax, res_folder / '{}_train{}_{:02d}-test{}.csv'.format(add_real['arch'], add_real['train'], i, add_real['test']), draw_confidence_interval=False, colors = c_list, pod_alpha=0.2, label='Test on real data', **pod_args)
            else:
                fit = comparison_part(ax, res_folder / '{}_train{}_{:02d}-test{}.csv'.format(add_real['arch'], add_real['train'], i, add_real['test']), draw_confidence_interval=False, colors = c_list, pod_alpha=0.2, **pod_args)
            if fit != -1:
                s90, _, _ = pts.pod.compute_s90(fit)
                s2_list.append(s90)
        s2_list = np.array(s2_list)
        s2_mean = s2_list.mean()
        print(s2_list)
        print('On real test data:')
        print('s_90 = {:.2f} +- {:.2f}'.format(s2_list.mean(), s2_list.std()))
        ax.vlines([s2_mean], 0., 0.9, linestyles='--', color='g')
        ax.scatter([s2_mean], [0.9], color='g', s=16)
        res = [s_mean, s_se]
    
    #ax.legend([ellipse2, ellipse, artists[0]], ['Network variance', 'Gaussian fit', '95% confidence'])
    ax.legend(fontsize=16)
    
    return res
    
def single_network_pods(res_folder, add_gen=None, add_real=None):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    
    network_pods(ax, res_folder, add_gen=add_gen, add_real=add_real)
    
    plt.tight_layout()
    #plt.show()
    #plt.savefig('tmp_imgs/chicken_data_pod/pod_contrast.pdf', format='pdf')
    plt.savefig('tmp_imgs/chicken_data_pod/network_pods.pdf', format='pdf')
    
def comp_network_pods(res_folder, names):
    fig, ax = plt.subplots(2, 2, figsize=(16,12))
    
    network_pods(ax[0,0], res_folder, add_real=names[0], xlabel_size=14, ylabel_size=14)
    ax[0,0].set_title('(a) t = 1s', y = -0.2, fontsize=18, weight='bold')
    network_pods(ax[0,1], res_folder, add_gen=names[2], add_real=names[1], xlabel_size=14, ylabel_size=14)
    ax[0,1].set_title('(b) t = 100ms', y = -0.2, fontsize=18, weight='bold')
    network_pods(ax[1,0], res_folder, add_gen=names[4], add_real=names[3], xlabel_size=14, ylabel_size=14)
    ax[1,0].set_title('(c) t = 50ms', y = -0.2, fontsize=18, weight='bold')
    network_pods(ax[1,1], res_folder, add_gen=names[6], add_real=names[5], xlabel_size=14, ylabel_size=14)
    ax[1,1].set_title('(d) t = 20ms', y = -0.2, fontsize=18, weight='bold')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('tmp_imgs/chicken_data_pod/network_pods.pdf', format='pdf')
                
def settings2contrast(ax, res_folder, names):
    nums = [0, 2, 4, 6]
    t_vals = [1000, 100, 50, 20]
    c_vals = []
    c_std = []
    
    for i in nums:
        mean, std = network_pods(ax, res_folder, add_gen=names[i])
        c_vals.append(mean)
        c_std.append(std)
        
    plt.cla()
    
    ax.plot(t_vals, c_vals, linestyle='--', marker='o')
    ax.set_xlabel('Exposure time, ms', fontsize=16)
    ax.set_ylabel("Image Contrast for P=90\%", fontsize=16)
    ax.grid(True)
    ax.set_title('(b)', y = -0.13, fontsize=18, weight='bold')
    
def pod_same(res_folder, names):
    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    
    selector = np.zeros_like(names, dtype=bool)
    selector[0] = True
    selector[2] = True
    selector[4] = True
    selector[6] = True
    names = np.array(names)
    labels = ['1s/obj', '100ms/obj', '50ms/obj', '20ms/obj']
    c = ['r', 'g', 'b', 'k']
    for i, name in enumerate(names[selector]):
        plot_average_network(ax[0], res_folder, name, labels[i], colors=[c[i], 'b'], plot_uncertainty=False)
    
    ax[0].legend(fontsize=16)
    ax[0].grid(True)
    ax[0].set_xlim(0., 0.25)
    ax[0].set_ylim(0., 1.)
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[0].yaxis.set_major_locator(plt.LinearLocator(11))
    ax[0].yaxis.set_major_formatter(ticker.PercentFormatter(1.))
    ax[0].set_xlabel('Image Contrast', fontsize=16)
    ax[0].set_ylabel("Probability of Detection", fontsize=16)
    ax[0].set_title('(a)', y = -0.13, fontsize=18, weight='bold')
    
    settings2contrast(ax[1], res_folder, names)

    plt.tight_layout()
    plt.savefig('tmp_imgs/chicken_data_pod/pod_same.pdf', format='pdf')
    plt.show()
        
if __name__ == "__main__":
    res_folder = Path('./tmp_res')
    names = [
        {'train' : '40kV_40W_100ms_10avg',
         'test' : '40kV_40W_100ms_10avg',
         'arch' : 'eff'},
        
        {'train' : 'gen_40kV_40W_100ms_1avg',
         'test' : '40kV_40W_100ms_1avg',
         'arch' : 'eff'},
        {'train' : 'gen_40kV_40W_100ms_1avg',
         'test' : 'gen_40kV_40W_100ms_1avg',
         'arch' : 'eff'},
        
        {'train' : 'gen_40kV_40W_50ms_1avg',
         'test' : '40kV_40W_50ms_1avg',
         'arch' : 'eff'},
        {'train' : 'gen_40kV_40W_50ms_1avg',
         'test' : 'gen_40kV_40W_50ms_1avg',
         'arch' : 'eff'},
        
        {'train' : 'gen_40kV_40W_20ms_1avg',
         'test' : '40kV_40W_20ms_1avg',
         'arch' : 'eff'},
        {'train' : 'gen_40kV_40W_20ms_1avg',
         'test' : 'gen_40kV_40W_20ms_1avg',
         'arch' : 'eff'},
        
        {'train' : 'gen_40kV_40W_200ms_1avg',
         'test' : 'gen_40kV_40W_200ms_1avg',
         'arch' : 'eff'},
        {'train' : 'gen_40kV_40W_75ms_1avg',
         'test' : 'gen_40kV_40W_75ms_1avg',
         'arch' : 'eff'}
    ]
        
    # This function produces Fig. 6
    comp_network_pods(res_folder, names)
        
    # This function produces Fig. 7
    pod_same(res_folder, names)

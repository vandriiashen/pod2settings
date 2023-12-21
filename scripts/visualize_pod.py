import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
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
    
def plot_residuals(res_folder, name):
    res_arr = np.genfromtxt(res_folder / '{}.csv'.format(name), delimiter=',', names=True)
    contrast = res_arr['Contrast']
    correct_det = np.where(res_arr['Prediction'] == res_arr['Target'], 1, 0)
    fit_contrast = pts.pod.stat_analyze(contrast, correct_det)
    
    r = pts.pod.compute_residuals(fit_contrast, contrast, correct_det)
    print(type(r))
    
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    ax.scatter(contrast, r)
    ax.set_ylabel('Residual', fontsize=20)
    ax.set_xlabel('Image Contrast', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    #ax.set_xlim(0., 0.2)
    ax.grid(True)
    
    #plt.savefig('tmp_imgs/chicken_data_pod/likelihood.png')

def compute_confidence(z):
    import scipy
    norm = z.sum()
    
    def conf_area(x, z):
        area = z[z > 10**x].sum()
        print(x, area, area/norm)
        return area/norm
    f = lambda x: np.abs(conf_area(x, z) - 0.95)
    
    out = scipy.optimize.minimize(f, method='Nelder-Mead', x0=[-23])
    res = 10**out['x']
    print(res)
    return res

def confidence_ellipse(ax, means, cov, n_std=2., c='y'):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, ec=c, fc=None, linewidth=3)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    mean_x, mean_y = means

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ellipse.set_fill(False)
    ax.add_patch(ellipse)
    
    return ellipse


def plot_probability_map(a, b, z, conf_level = 0.05, glm_fit=None, f1=None, network_fit=None):
    fig, ax = plt.subplots()

    #z = z[:-1, :-1]
    z_min, z_max = z.min(), z.max()
    #z_min, z_max = 10**-20, z.max()
    #z_min, z_max = f1.min(), f1.max()

    c = ax.pcolor(a, b, z[:-1, :-1], cmap='RdBu', vmin=z_min, vmax=z_max)
    #ax.imshow(f1)
    
    cont = ax.contour(a, b, z, levels=[conf_level], colors='g', linewidths=[3])
    artists, labels = cont.legend_elements()

    if glm_fit:
        cov = glm_fit.cov_params()
        means = glm_fit.params
        print(means)
        print(cov)
        ellipse = confidence_ellipse(ax, means, cov, n_std=2.0)
        #ax.legend([ellipse, artists[0]], ['Gaussian fit', '95% confidence'])
        #ax.legend([ellipse], ['Gaussian fit'])
        #ax.legend(artists, labels)
        
    if network_fit:
        means, cov, k, b = network_fit
        print(means)
        print(cov)
        ellipse2 = confidence_ellipse(ax, means, cov, n_std=2.0, c='b')
        ax.scatter(k, b, c='cyan', s=20)
        ax.legend([ellipse2, ellipse, artists[0]], ['Network variance', 'Gaussian fit', '95% confidence'])
    
    ax.set_title('Likelihood')
    ax.set_xlabel('k')
    ax.set_ylabel('b')
    #fig.colorbar(c, ax=ax)
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/chicken_data_pod/likelihood.png')
    #plt.show()
    
def network_variance(data):
    k = np.zeros((10,))
    b = np.zeros((10,))
    for i in range(10):
        name = 'tmp_res/{}_train{}_{:02d}-test{}.csv'.format(data['arch'], data['train'], i, data['test'])
        res_arr = np.genfromtxt(name, delimiter=',', names=True)
        contrast = res_arr['Contrast']
        correct_det = np.where(res_arr['Prediction'] == res_arr['Target'], 1, 0)
        fit_contrast = pts.pod.stat_analyze(contrast, correct_det)
        
        k[i], b[i] = fit_contrast.params
        
    cov = np.zeros((2,2))
    cov[0,0] = ((k - k.mean()) * (k - k.mean())).mean()
    cov[1,0] = ((b - b.mean()) * (k - k.mean())).mean()
    cov[0,1] = ((k - k.mean()) * (b - b.mean())).mean()
    cov[1,1] = ((b - b.mean()) * (b - b.mean())).mean()
    
    means = np.zeros((2))
    means[0] = k.mean()
    means[1] = b.mean()
    
    return means, cov, k, b
        
def probability_distribution(res_folder, name):
    default_num = 0
    res_arr = np.genfromtxt(res_folder / '{}_train{}_{:02d}-test{}.csv'.format(name['arch'], name['train'], default_num, name['test']), delimiter=',', names=True)
    contrast = res_arr['Contrast']
    correct_det = np.where(res_arr['Prediction'] == res_arr['Target'], 1, 0)
    
    fit = pts.pod.stat_analyze(contrast, correct_det)
    print(fit.params)
    print(fit.cov_params())

    
    #contrast = np.concatenate((contrast, np.repeat(0., 100)))
    #correct_det = np.concatenate((correct_det, np.repeat(0., 100)))
    #contrast = np.concatenate((contrast, np.repeat(1., 100)))
    #correct_det = np.concatenate((correct_det, np.repeat(1., 100)))
    
    print(contrast)
    print(correct_det)

    
    a, b = np.meshgrid(np.linspace(-150, 150, 100), np.linspace(-15, 15, 100))

    def l(k, b, verbose = False):
        f = np.nan_to_num(k * contrast + b)
        p = np.exp(f) / (1 + np.exp(f))
        l = np.nan_to_num(correct_det * np.log(p) + (1-correct_det) * np.log(1-p))
        if verbose:
            print(f)
            print(p)
            print(l)
        res = -l.sum()
        return res
        
    def f1_score(k, b):
        f = np.nan_to_num(k * contrast + b)
        tp = np.count_nonzero(np.logical_and(f >= 0, correct_det == 1))
        fn = np.count_nonzero(np.logical_and(f < 0, correct_det == 1))
        fp = np.count_nonzero(np.logical_and(f >= 0, correct_det == 0))
        
        return tp
    
    z = np.zeros_like(a)
    f1 = np.zeros_like(a)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            z[y,x] = np.exp(-l(a[y,x], b[y,x]))
            f1[x,y] = f1_score(a[y,x], b[y,x])
            
    idx = np.unravel_index(np.argmin(-z, axis=None), z.shape)

    print('Min log L')
    print(a[idx], b[idx], z[idx])
    
    conf_lvl = compute_confidence(z)
    print('Conf lvl', conf_lvl)
    
    network_fit = network_variance(name)
    print(network_fit)
    
    plot_probability_map(a, b, z, conf_level=conf_lvl, glm_fit=fit, network_fit=network_fit, f1=f1)
        
def model_test():
    px = np.array([1., 3., 2., 4.,])
    py = np.array([0, 0, 1, 1])
    
    a, b = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    def l(k, b, verbose = False):
        f = np.nan_to_num(k * px + b)
        p = np.exp(f) / (1 + np.exp(f))
        l = np.nan_to_num(py * np.log(p) + (1-py) * np.log(1-p))
        if verbose:
            print('Mu: ', f)
            print('Probability: ', p)
            print('y: ', py)
            print('log: ', l)
        res = -l.sum()
        return res
    
    def f1_score(k, b):
        f = np.nan_to_num(k * px + b)
        tp = np.logical_and(f >= 0, py == 1)
        fn = np.logical_and(f < 0, py == 1)
        fp = np.logical_and(f >= 0, py == 0)
        return tp
    
    z = np.zeros_like(a)
    f1 = np.zeros_like(a)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            z[y,x] = np.exp(-l(a[y,x], b[y,x]))
            f1[x,y] = f1_score(a[y,x], b[y,x])
            
    idx = np.unravel_index(np.argmin(-z, axis=None), z.shape)
    print('Min log L')
    print(a[idx], b[idx], z[idx])
    
    conf_lvl = compute_confidence(z)
    plot_probability_map(a, b, z, conf_level=conf_lvl)
    
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
    
def plot_recall(res_folder, name):
    fig, ax = plt.subplots(1, 1, figsize=(18,9))
    
    res_arr = np.genfromtxt(res_folder / '{}.csv'.format(name), delimiter=',', names=True)
    contrast = res_arr['Contrast']
    recall = res_arr['Recall']
    non_zero = recall != 0
    contrast = contrast[non_zero]
    recall = recall[non_zero]
    
    ax.scatter(contrast, recall, alpha=0.5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Recall', fontsize=20)
    ax.set_xlabel('Image Contrast', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(0., 0.2)
    ax.grid(True)
    
    plt.show()
    
def plot_pod(res_folder, name):
    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    
    default_num = 5
    plot_points(ax[0], res_folder / '{}_train{}_{:02d}-test{}.csv'.format(name['arch'], name['train'], default_num, name['test']))
    comparison_part(ax[1], res_folder / '{}_train{}_{:02d}-test{}.csv'.format(name['arch'], name['train'], default_num, name['test']))
        
    plt.tight_layout()
    #plt.show()
    #plt.savefig('tmp_imgs/chicken_data_pod/pod_contrast.pdf', format='pdf')
    plt.savefig('tmp_imgs/chicken_data_pod/pod_contrast.png')
    
def network_pods(res_folder, name, add_real=None):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    s_list = []
    s2_list = []
    k_list = []
    b_list = []
    
    for i in range(0, 25):
        flag = False
        c_list = ['r', 'b']
        
        fit = comparison_part(ax, res_folder / '{}_train{}_{:02d}-test{}.csv'.format(name['arch'], name['train'], i, name['test']), draw_confidence_interval=False, colors = c_list, pod_alpha=0.2)
        if fit != -1:
            s90, _, _ = pts.pod.compute_s90(fit)
            s_list.append(s90)
            k, b = fit.params
            k_list.append(k)
            b_list.append(b)
            #print(i, s90)
            
    s_list = np.array(s_list)
    s_mean = s_list.mean()
    s_se = s_list.std()
    alpha = 0.05
    q = scipy.stats.norm.ppf(1 - alpha / 2.)
    print('On generated test data:')
    print(s_mean - q*s_se, s_mean, s_mean + q*s_se)
    ax.vlines([s_mean, s_mean - q*s_se, s_mean + q*s_se], 0., 0.9, linestyles='--', color='r')
    ax.scatter([s_mean, s_mean - q*s_se, s_mean + q*s_se], [0.9, 0.9, 0.9], color='r', s=16)
            
    if add_real:
        for i in range(10, 31):
            c_list = ['g', 'b']
            fit = comparison_part(ax, res_folder / '{}_train{}_{:02d}-test{}.csv'.format(add_real['arch'], add_real['train'], i, add_real['test']), draw_confidence_interval=False, colors = c_list, pod_alpha=0.2)
            if fit != -1:
                s90, _, _ = pts.pod.compute_s90(fit)
                s2_list.append(s90)
        s2_list = np.array(s2_list)
        s2_mean = s2_list.mean()
        print(s2_list)
        print('On real test data:')
        print(s2_mean)
        ax.vlines([s2_mean], 0., 0.9, linestyles='--', color='g')
        ax.scatter([s2_mean], [0.9], color='g', s=16)
    
    
    k_list = np.array(k_list)
    b_list = np.array(b_list)
    
    
    
    '''
    print(s_list.mean())
    print(s_list.std())
    k = k_list.mean()
    b = b_list.std()
    cov = np.zeros((2, 2))
    cov[0,0] = ((k_list - k)*(k_list - k)).mean()
    cov[1,0] = ((b_list - b)*(k_list - k)).mean()
    cov[0,1] = ((k_list - k)*(b_list - b)).mean()
    cov[1,1] = ((b_list - b)*(b_list - b)).mean()
    par = np.array([k, b])
    print(par)
    print(cov)
    alpha = 0.05
    
    x_range = np.linspace(0., 0.2, 100)
    fit_x = np.ones((x_range.shape[0], 2))
    fit_x[:,0] = x_range
        
    fit_y = fit_x @ par
    #print(fit_y)
    var = fit_x @ cov @ fit_x.T
    var_y = var.sum(1)
    var_y[var_y < 0.] = 0.
    #print(var_y)
    se = np.sqrt(var_y)
    q = scipy.stats.norm.ppf(1 - alpha / 2.)
    lower = fit_y - q * se
    upper = fit_y + q * se
        
    p = fit.family.link.inverse(fit_y)
    p_low = fit.family.link.inverse(lower)
    p_high = fit.family.link.inverse(upper)
    
    ax.fill_between(x_range, p_low, p_high, color='b', alpha = 0.2)
    '''
        
    plt.tight_layout()
    #plt.show()
    #plt.savefig('tmp_imgs/chicken_data_pod/pod_contrast.pdf', format='pdf')
    plt.savefig('tmp_imgs/chicken_data_pod/network_pods.png')
    
def network_test(i):
    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    
    name = 'mob3s_traingen_40kV_40W_100ms_1avg_{:02d}-test40kV_40W_100ms_1avg'.format(i)
    
    plot_points(ax[0], res_folder / '{}.csv'.format(name))
    #comparison_part(ax[1], res_folder / '{}.csv'.format(name))
        
    plt.tight_layout()
    #plt.show()
    #plt.savefig('tmp_imgs/chicken_data_pod/pod_contrast.pdf', format='pdf')
    plt.savefig('tmp_imgs/chicken_data_pod/mob3s_pod_contrast_{:02d}n.png'.format(i))
        
def pod_comparison(res_folder, names):
    fig, ax = plt.subplots(2, 2, figsize=(16,12))
    
    comparison_part(ax[0,0], res_folder / '{}.csv'.format(names[0]), legend_loc = 4, label='Real Test set')
    comparison_part(ax[0,1], res_folder / '{}.csv'.format(names[1]), label='Real Test set')
    comparison_part(ax[0,1], res_folder / '{}.csv'.format(names[2]), label='Generated Test set', colors = ['g', 'b'], linestyle='-', draw_confidence_interval=False)
    comparison_part(ax[1,0], res_folder / '{}.csv'.format(names[3]), label='Real Test set')
    comparison_part(ax[1,0], res_folder / '{}.csv'.format(names[4]), label='Generated Test set', colors = ['g', 'b'], linestyle='-', draw_confidence_interval=False)
    comparison_part(ax[1,1], res_folder / '{}.csv'.format(names[5]), label='Real Test set')
    comparison_part(ax[1,1], res_folder / '{}.csv'.format(names[6]), label='Generated Test set', colors = ['g', 'b'], linestyle='-', draw_confidence_interval=False)
    
    plt.tight_layout()
    plt.savefig('tmp_imgs/chicken_data_pod/pod_comparison.pdf', format='pdf')
    
def pod_same(res_folder, names):
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    
    comparison_part(ax, res_folder / '{}.csv'.format(names[0]), draw_confidence_interval=False, label='1s/obj')
    comparison_part(ax, res_folder / '{}.csv'.format(names[2]), draw_confidence_interval=False, colors = ['g', 'b'], label='100ms/obj')
    comparison_part(ax, res_folder / '{}.csv'.format(names[4]), draw_confidence_interval=False, colors = ['b', 'b'], label='50ms/obj')
    comparison_part(ax, res_folder / '{}.csv'.format(names[6]), draw_confidence_interval=False, colors = ['k', 'b'], legend_loc=4, label='20ms/obj')
    
    plt.tight_layout()
    #plt.savefig('tmp_imgs/chicken_data_pod/pod_same.pdf', format='pdf')
    plt.savefig('tmp_imgs/chicken_data_pod/pod_same.png')
    
def settings2contrast(res_folder, names):
    '''
    pt2name_gen = {
        6 : 20,
        4 : 50,
        2 : 100,
        0 : 1000
    }
    pt2name_real = {
        1 : 100,
        3 : 50,
        5 : 20
    }
    
    thr_contrast = []
    thr_lower = []
    thr_upper = []
    pt_list = []
    thr_contrast_real = []
    pt_list_real = []
    
    for idx, pt in pt2name_gen.items():
        res_arr = np.genfromtxt(res_folder / '{}.csv'.format(names[idx]), delimiter=',', names=True)
        contrast = res_arr['Contrast']
        correct_det = np.where(res_arr['Prediction'] == res_arr['Target'], 1, 0)
        fit_contrast = pts.pod.stat_analyze(contrast, correct_det)
        s90, lower, upper = pts.pod.compute_s90(fit_contrast)
        print(s90, lower, upper)
        if lower == -1:
            lower = 0.2 # max value
        if s90 != -1:
            pt_list.append(1/pt)
            thr_contrast.append(s90)
            thr_lower.append(lower)
            thr_upper.append(upper)
        
    for idx, pt in pt2name_real.items():
        res_arr = np.genfromtxt(res_folder / '{}.csv'.format(names[idx]), delimiter=',', names=True)
        contrast = res_arr['Contrast']
        correct_det = np.where(res_arr['Prediction'] == res_arr['Target'], 1, 0)
        fit_contrast = pts.pod.stat_analyze(contrast, correct_det)
        s90, _, _ = pts.pod.compute_s90(fit_contrast)
        if s90 != -1:
            pt_list_real.append(1/pt)
            thr_contrast_real.append(s90)
    '''
    
    pt_list = [20, 50, 100, 1000]
    thr_lower = [0.143, 0.087, 0.082, 0.01]
    thr_contrast = [0.179, 0.132, 0.101, 0.033]
    thr_upper = [0.213, 0.177, 0.120, 0.087]
    pt_list_real = [20, 50, 100]
    thr_contrast_real = [0.205, 0.114, 0.082] 
    
    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    ax.fill_between(pt_list, thr_lower, thr_upper, color='b', alpha = 0.2)
    ax.plot(pt_list, thr_contrast, label = 'Prediction based on generated data', c='r')
    ax.scatter(pt_list_real, thr_contrast_real, label = 'Performance on real data', marker = 'o', s = 12, c = 'g')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Exposure, ms', fontsize=14)
    ax.set_ylabel('Detectable FO quotient contrast', fontsize=14)
    ax.grid(True)
    ax.legend(fontsize = 12)
    plt.tight_layout()
    #plt.savefig('tmp_imgs/chicken_data_pod/settings2contrast.pdf', format='pdf')
    plt.savefig('tmp_imgs/chicken_data_pod/settings2contrast.png')
    
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
         'test' : 'gen_40kV_40W_200ms_1avg'},
        
        {'train' : 'gen_40kV_40W_500ms_1avg',
         'test' : 'gen_40kV_40W_500ms_1avg'}
    ]
    #'traingen_40kV_40W_100ms_1avg-testgen_40kV_40W_100ms_1avg',
    
    data = names[0]
    
    plot_pod(res_folder, data)
    network_pods(res_folder, names[0], add_real=None)
    #plot_recall(res_folder, names[1])
    #pod_comparison(res_folder, names)
    #pod_same(res_folder, names)
    settings2contrast(res_folder, names)
    #plot_residuals(res_folder, names[1])
    #probability_distribution(res_folder, data)
    #print(data)
    
    #for i in range(10):
    #    network_test(i)
    
    #network_variance()
    #model_test()

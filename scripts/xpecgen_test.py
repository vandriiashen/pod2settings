from xpecgen import xpecgen as xg
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.optimize

def gen_spectrum(voltage):
    E0 = voltage
    theta = 90
    start = 10
    num_points = E0+1-start

    s=xg.calculate_spectrum(E0, theta, start, num_points)
    s.attenuate(0.05, xg.get_mu(29))
    return s
    
def plot_spectrum(ax, s, label):
    points = s.get_points(num_discrete=0)
    print(points[0])
    x = points[0]
    y = points[1] / np.array(points[1]).sum()
    ax.plot(x, y, label=label)

def test_spectra():
    fig, ax = plt.subplots(figsize=(12,9))
    
    s = gen_spectrum(40)
    plot_spectrum(ax, s, '40kV')
    s = gen_spectrum(60)
    plot_spectrum(ax, s, '60kV')
    s = gen_spectrum(90)
    plot_spectrum(ax, s, '90kV')
    
    ax.grid(True)
    ax.legend(fontsize=20)
    ax.set_ylabel('Intensity', fontsize=20)
    ax.set_xlabel('Energy, keV', fontsize=20)
    
    plt.savefig('tmp_imgs/spectrum/spectra.png')
        
class MaterialCalibration():
    def __init__(self):
        self.k1 = 1.
        self.k2 = 0.05
        #self.k1 = 0.78
        #self.k2 = 0.14
        self.E_range = np.arange(10, 91, 1)
        # Al
        self.rho = [1.3, 4.06]
        self.z_eff = [13, 29]
        # Cu
        #self.rho = 4.06
        # density = 8.96 g/cm^3 * 29(Z) / 64(A)
        #self.z_eff = 29
        
        self.n = 3.8
        
        self.precomputed_spectra = {}
        self.voltages = [40, 50, 60, 70, 80, 90]
        for voltage in self.voltages:
            self.precomputed_spectra[voltage] = gen_spectrum(voltage)
            
        self.X = np.zeros((1,3))
        self.X[0,:] = [40, 0., 0.]
        self.y = np.zeros((1))
        
    def f_kn(self, E):
        e = E / 511. # in keV
        p1 = (1 + e) / e**2 
        p2 = 2 * (1+e) / (1+2*e) - np.log(1 + 2*e) / e
        p3 = np.log(1 + 2*e) / (2*e)
        p4 = (1 + 3*e) / (1 + 2*e)**2
        return p1*p2 + p3 - p4
    
    def p_E(self):
        return self.k1 * np.power(self.E_range.astype(np.float32), -3)
    
    def c_E(self):
        return self.k2 * self.f_kn(self.E_range)
    
    def mu(self, mat):
        vals = self.rho[mat] * (np.power(self.z_eff[mat], self.n - 1) * self.p_E() + self.c_E())
        return vals.astype(np.float32)
    
    def compute_att(self, voltage, l, mat):
        s = self.precomputed_spectra[voltage]
        points = s.get_points(num_discrete=0)
        fract_vals = np.array(points[1])
        fract_vals = np.pad(fract_vals, (0, self.E_range.shape[0] - fract_vals.shape[0]))
        fract = fract_vals / fract_vals.sum()
        
        #print(fract)
        #print(self.mu())
        att_E = fract * np.exp(-self.mu(mat) * l)
        #print(att_E)
        att = -np.log(att_E.sum())
        return att
    
    def plot_curves(self):
        fig, ax = plt.subplots(figsize=(12,9))
    
        l_range = np.linspace(0., 1.5, 100)
        for v in self.voltages:
            att_vals = [self.compute_att(v, l, 0) for l in l_range]
            ax.plot(l_range, att_vals, label='{} kV'.format(v))
        
        ax.grid(True)
        ax.legend(fontsize=20)
        ax.set_ylabel('Attenuation', fontsize=20)
        ax.set_xlabel('Voltage, kV', fontsize=20)
        plt.savefig('tmp_imgs/spectrum/gen_att.png')
    
    def compare_with_data(self, mat):
        fig, ax = plt.subplots(figsize=(12,9))
    
        l_range = np.linspace(0., self.X[ self.X[:,2] == mat ,1].max(), 100)
        for v in self.voltages:
            att_vals = [self.compute_att(v, l, mat) for l in l_range]
            ax.plot(l_range, att_vals, label='{} kV'.format(v))
            
            select = np.logical_and(self.X[:,0] == v, self.X[:,2] == mat)
            ax.scatter(self.X[select,1], self.y[select], label='Exp {} kV'.format(v))
        
        ax.grid(True)
        ax.legend(fontsize=20)
        ax.set_ylabel('Attenuation', fontsize=20)
        ax.set_xlabel('Thickness, mm', fontsize=20)
        plt.savefig('tmp_imgs/spectrum/gen_att_{}.png'.format(mat))
        
    def load_exp_data(self, fname, mat_num):
        data = np.loadtxt(fname, delimiter = ',')
        voltages = data[1:,0]
        l_vals = data[0,1:]
        
        pairs = [x for x in itertools.product(voltages, l_vals)]
        X = np.zeros((len(pairs), 3))
        X[:,:2] = np.array(pairs)
        X[:,2] = mat_num
        y = data[1:,1:].ravel()
        
        print(self.X.shape)
        self.X = np.concatenate((self.X, X))
        self.y = np.concatenate((self.y, y))
        print(self.X.shape)
        
    def cost_func(self, exp_X, exp_y, k1, k2, rho, z_eff, n):
        cost = 0
        for i in range(exp_X.shape[0]):
            v = exp_X[i,0]
            l = exp_X[i,1]
            m = int(exp_X[i,2])
            tg = exp_y[i]
            
            s = self.precomputed_spectra[v]
            points = s.get_points(num_discrete=0)
            fract_vals = np.array(points[1])
            fract_vals = np.pad(fract_vals, (0, self.E_range.shape[0] - fract_vals.shape[0]))
            fract = fract_vals / fract_vals.sum()
            
            p_E = k1 * np.power(self.E_range.astype(np.float32), -3)
            c_E = k2 * self.f_kn(self.E_range)
            mu = rho[m]* (np.power(z_eff[m], n - 1) * p_E + c_E)
            
            att_E = fract * np.exp(-mu * l)
            att = -np.log(att_E.sum())
            
            if tg > 0:
                cost += (att - tg)**2 / tg
        
        return cost
    
    def cost_k(self, x, args):
        k1, k2, n = x
        exp_X, exp_y, rho, z_eff = args
        return self.cost_func(exp_X, exp_y, k1, k2, rho, z_eff, n)
            
    def opt_k(self):
        x0 = [self.k1, self.k2, self.n]
        args = [self.X, self.y, self.rho, self.z_eff]
        bnds = ((0, None), (0, None), (0, None))
        res = scipy.optimize.minimize(self.cost_k, x0, args, method='Powell')
        self.k1, self.k2, self.n = res.x
        
        print(res)
            
    def plot_c_E(self):
        fig, ax = plt.subplots(figsize=(12,9))
    
        p = self.p_E()
        ax.plot(self.E_range, p, label = 'Absorption')
        c = self.c_E()
        ax.plot(self.E_range, c, label = 'Scattering')
        
        ax.grid(True)
        ax.legend(fontsize=20)
        ax.set_ylabel('Values', fontsize=20)
        ax.set_xlabel('Energy, keV', fontsize=20)
        plt.savefig('tmp_imgs/spectrum/basis.png')
    
if __name__ == "__main__":
    #test_spectra()
    calibration = MaterialCalibration()
    #calibration.plot_c_E()
    #print(calibration.compute_att(60, 1.))
    #calibration.plot_curves()
    calibration.load_exp_data('tmp_res/al_calibr.csv', 0)
    calibration.load_exp_data('tmp_res/cu_calibr.csv', 1)
    calibration.opt_k()
    calibration.compare_with_data(0)
    calibration.compare_with_data(1)

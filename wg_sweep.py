import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pylab
import os.path
import mems_fitting
import tf_fitting
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class sweep:
    #mems file location and naming structure
    mems_freq_path = \
    '/home/vlu/Dropbox/Data/CD160112/MEMS'
    mems_ex_path = \
    '/Users/cbarquist/Documents/Physics/Research/4He/Data/cd141121/Excitation/mems'
    mems_sweep_name = 'MEMS_{}.dat'
  
  
    #TF file location and naming structure
    tf_freq_path = \
    '/home/vlu/Dropbox/Data/CD160112/TF'
    tf_ex_path = \
    "/home/vlu/Dropbox/Data/CD160112/TF/Excitation"
    tf_sweep_name = 'TF_{}.dat'
    

    def __init__(self, idn, **kwargs):
        #Attributes
        self.idn = idn #string: corresponding to specific sweep in a folder
        self.f0 = None #float: resonance frequency fitting-parameter
        self.df = None #float: resonance width fitting-parameter
        self.A = None #float: peak amplitude fitting-parameter
        self.ph = None #float: phase fitting parameter
        self.bg_0 = None #float: 0th order polynomial background fitting-paramete 
        self.bg_1 = None #float: 1st order polynomial background fitting-paramate

        if 'excitation' in kwargs:
            self.excitation = kwargs['excitation']
        else:
            self.excitation = None
        # In order to make it easy to manipulate data the storage location of
        # sweep data is kept in a golbal variable and the specific device to be
        # is specified by using the kwarg device (e.g. sweep('###' , device =
        # 'mems'. The MEMS is the default device and will be loaded if noting
        # is specified. The same is true for the sweep method: the frequency
        # sweep is assumed unless specified through the kwarg.
        if 'device' in kwargs:    
            self.device = kwargs['device']
            self.load_data(device = kwargs['device'].lower(), method = \
                    kwargs['method'].lower())
        else:
            self.device = 'mems'
            self.load_data(device = 'mems', method = 'f')
        
        self.x_fit = np.zeros(self.x.shape)
        self.y_fit = np.zeros(self.y.shape)
        self.rr_bg_removed = np.zeros(self.r.shape)
        self.x_bg_removed = np.zeros(self.x.shape)
        self.y_bg_removed = np.zeros(self.y.shape)

#------------------------------------------------------------------------------

    def load_data(self, **kwargs):
        "This is a docstring"
        if kwargs['device'].lower() == 'mems':
            if kwargs['method'].lower() == 'f':
                self.f, self.x, self.y, self.r, \
                self.mc_temp, self.mct_temp, self.mct_cap = \
                np.loadtxt("{}/".format(sweep.mems_freq_path) +
                    sweep.mems_sweep_name.format(self.idn),
                    skiprows=1,usecols = (0,1,2,3,4,5,6),unpack = True)
                self.f = self.f 
         
            elif kwargs['method'].lower == 'ex':
                self.ex, self.x, self.y, self.r,\
                self.mc_temp, self.mct_temp, self.diode_temp = \
                np.loadtxt("{}/".format(sweep.mems_ex_path) +
                        sweep.mems_sweep_Name.format(self.idn),
                        usecols = (0,1,2,3,4,5,6),unpack = True)
       
        elif kwargs['device'].lower() == 'tf':    
            if kwargs['method'] == 'f':
                self.f, self.x, self.y, self.r,\
                self.mc_temp,  self.mct_temp, self.diode_temp = \
                np.loadtxt("{}/".format(sweep.tf_freq_path) +
                        sweep.tf_sweep_name.format(self.idn),
                        usecols = (0,1,2,3,4,5,6),skiprows=1,unpack = True)
       
            if kwargs['method'] == 'ex':
                self.ex, self.x, self.y, self.r, \
                self.mc_temp, self.mct_temp, self.diode_temp = \
                np.loadtxt("{}/".format(sweep.tf_ex_path) +
                        sweep.tf_sweep_name.format(self.idn),
                        usecols = (0,1,2,3,4,5,6),unpack = True)
        # This converst all of the outputs (volts) into micro volts and also
        # converts rms values to peak voltage amplitudes
        self.x = self.x * 1e6 * np.sqrt(2)
        self.y = self.y * 1e6 * np.sqrt(2)         
        self.r = self.r * 1e6 * np.sqrt(2) 

        self.xdis = self.displacement(self.x)
        self.ydis = self.displacement(self.y)
        self.rdis = self.displacement(self.r)
       
        self.xvel = self.velocity(self.x,self.f)
        self.yvel = self.velocity(self.y,self.f)
        self.rvel = self.velocity(self.r,self.f)
            
               #Calculate Temperature statistics
        self.avg_mc_temp = np.average(self.mc_temp)
        self.std_mc_temp = np.std(self.mc_temp)
       
        self.avg_mct_temp = np.average(self.mct_temp)
        self.std_mct_temp = np.std(self.mct_temp)

#        self.avg_diode_temp = np.average(self.diode_temp)
#       self.std_diode_temp = np.std(self.diode_temp)

        # Load in excitation from log file
        log_path = '/home/vlu/Documents/Research/4He/Data/CD160112/MEMS/LOG.txt'
#        self.ex = np.loadtxt(log_path, usecols= (10,), \
#               skiprows = int(self.idn)- 16)
#        self.excitation = self.ex[0]
 
        
#------------------------------------------------------------------------------

    def scale_data(self):
        self.x_scaled = self.x / (self.excitation**2)
        self.y_scaled = self.y / (self.excitation**2)
        self.r_scaled = self.r / (self.excitation**2)
        if self.x_bg_removed.any():
            self.x_bg_removed_scaled = self.x_bg_removed / (self.excitation**2)

        if self.y_bg_removed.any():
            self.y_bg_removed_scaled = self.y_bg_removed / (self.excitation**2)

        if self.r_bg_removed.any():
            self.r_bg_removed_scaled = self.y_bg_removed / (self.excitation**2)

        if self.rr_bg_removed.any():
            self.rr_bg_removed_scaled = self.rr_bg_removed / \
            (self.excitation**2)
#------------------------------------------------------------------------------

    def displacement(self,mu_volt):
        # beta*amp*vb
        inprop = 1.42e-9*6.82e11*10
        return mu_volt/inprop

#-----------------------------------------------------------------------------

    def velocity(self,mu_volt,freq):
        return self.displacement(mu_volt)*freq*2*np.pi*1e-4

#------------------------------------------------------------------------------

    def check_fit(self):
        #if self.x_fit
        if self.x_fit.any() and self.y_fit.any():
            f, sup = plt.subplots(1,2) #4x4 plot 
            plt.subplots_adjust(wspace=0.5, hspace=0.5) #Sets space between plots
            pylab.rcParams['figure.figsize'] = (11.0, 11.0) #Sets size of plot
            sup[0].plot(self.f, self.x, 'k*', self.f, self.x_fit, 'g-')
            sup[1].plot(self.f, self.y, 'k*', self.f, self.y_fit, 'g-')
        elif self.x_fit.any():
            res = self.x - self.x_fit
            ares = np.average(res)
            f, sup = plt.subplots(1,2) #4x4 plot 
            plt.subplots_adjust(wspace=0.5, hspace=0.5) #Sets space between plots
            pylab.rcParams['figure.figsize'] = (11.0, 11.0) #Sets size of plot
            sup[0].plot(self.f, self.x, 'k*', self.f, self.x_fit, 'g-')
            sup[1].plot(self.f, res, 'k*', self.f, [ares for i in \
                range(0,len(self.f))], 'g-')
            sup[0].plot(self.f, self.bg_0 + self.bg_1*self.f, 'm-')
        elif self.y_fit.any():
            plt.plot(self.f, self.y, 'k*', self.f, self.y_fit, 'g-')
            plt.plot(self.f, self.bg_0 + self.bg_1*self.f, 'm-')
        elif self.rrFit.any():
            plt.plot(self.f, self.rr_bg_removed, 'k*', self.f, self.rrFit, 'g-')
            plt.plot(self.f, self.bg_0 + self.bg_1*self.f, 'm-') 
        plt.show()
#------------------------------------------------------------------------------

    def quadPlot(sweeps, **kwargs): #sweeps should be a list
       
       # Initialize Sub Plots
        f, sup = plt.subplots(2,2) #4x4 plot 
        plt.subplots_adjust(wspace=0.5,hspace=0.5) #Sets space between plots
        pylab.rcParams['figure.figsize'] = (11.0, 11.0) #Sets size of plot
       
       #If quick exists then the function will load sweeps and plot them.
        if 'quick' in kwargs:
            if kwargs['quick']:
                leading_zero = ''
                kwargs['style'] = 'raw'
                for i in range(0,len(sweeps)):
                    if sweeps[i] < 10:
                        leading_zero = '00'
                    elif sweeps[i] < 100:
                        leading_zero = '0'
                    sweeps[i] = sweep(leading_zero+str(sweeps[i]), device = kwargs['device'], 
                                    method = 'f')    
        
        if not kwargs or kwargs['style'] == 'raw':
            for i in sweeps:
                sup[0,0].plot(i.f, i.x, "*-", label=i.idn)
                sup[0,1].plot(i.f, i.y, "*-")
                sup[1,0].plot(i.f, i.r, "*-")
                sup[1,1].plot(i.x, i.y, "*-")
         
        elif kwargs['style'] == 'bgr':
             for i in sweeps:
                sup[0,0].plot(i.f, i.x_bg_removed, "*-", label=i.idn)
                sup[0,1].plot(i.f, i.y_bg_removed, "*-")
                sup[1,0].plot(i.f, i.r_bg_removed, "*-")
                sup[1,1].plot(i.x_bg_removed, i.y_bg_removed, "*-")
 
        elif kwargs['style'] == 'raw_scaled':
             for i in sweeps:
                sup[0,0].plot(i.f, i.x_scaled, "*-", label=i.idn)
                sup[0,1].plot(i.f, i.y_scaled, "*-")
                sup[1,0].plot(i.f, i.r_scaled, "*-")
                sup[1,1].plot(i.x_scaled, i.y_scaled, "*-")      
        
        elif kwargs['style'] == 'bgr_scaled':
             for i in sweeps:
                sup[0,0].plot(i.f, i.x_bg_removed_scaled, "*-", label=i.idn)
                sup[0,1].plot(i.f, i.y_bg_removed_scaled, "*-")
                sup[1,0].plot(i.f, i.r_bg_removed_scaled, "*-")
                sup[1,1].plot(i.x_bg_removed_scaled, i.y_bg_removed_scaled, "*-")
       
        sup[0,0].set_title("X")
        sup[0,0].set_xlabel("Frequency (KHz)")
        sup[0,0].set_ylabel("Output (uV)")
        
        sup[0,1].set_title("Y")
        sup[0,1].set_xlabel("Frequency (KHz)")
        sup[0,1].set_ylabel("Output (uV)")

        sup[1,0].set_title("R")
        sup[1,0].set_xlabel("Frequency (KHz)")
        sup[1,0].set_ylabel("Output (uV)")
        
        sup[1,1].set_title("X-Y")
        sup[1,1].set_xlabel("Output (uV)")
        sup[1,1].set_ylabel("Output (uV)")
        
        sup[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5),
                    fancybox = True)
        plt.axis('equal')
        plt.show()

#------------------------------------------------------------------------------
    def sg(self):
        df = (np.max(self.f)-np.min(self.f) )/4.0
        f0 = self.f[np.argmax( np.abs(self.x))]
        A = (np.max(self.x) - np.min(self.x))
        return A, f0, df

    def quick(sweep_number):
        sw = sweep(str(sweep_number))
        sw.lrtzFit([0,0,0,90,0,0],'x',smart='')
        sw.print_attributes()
        sw.check_fit()
        del sw
  
    def quicktf(sweep_number):
        sw = sweep(str(sweep_number),device='tf',method='f')
        sw.lrtzFit([0,0,0,90,0,0],'x',smart='')
        sw.print_attributes()
        sw.check_fit()
        del sw       

    def lrtzFit(self, guess, what, **kwargs):
        dvar = self
        if self.device == 'mems':
            Fit_Functions = mems_fitting
        elif self.device == 'tf':
            Fit_Functions = tf_fitting
        
        if not not kwargs:
            guess[0], guess[1], guess[2] = self.sg()
       ## Fitting to X or Y requires a phase which is an extra fitting parameter
        if what.lower() == 'x':
            func = Fit_Functions.lrtzX
            popt, pcov = \
                curve_fit(Fit_Functions.lrtzX, self.f, self.x, p0 = guess, maxfev = 4000)
            self.A = popt[0]
            self.f0 = popt[1]
            self.df = popt[2]
            self.ph = popt[3]
            self.bg_0 = popt[4]
            self.bg_1 = popt[5]
            
            self.var_A = pcov[0,0]
            self.var_f0 = pcov[1,1]
            self.var_df = pcov[2,2]
            self.var_ph = pcov[3,3]
            self.var_bg_0 = pcov[4,4]
            self.var_bg_1 = pcov[5,5]

            self.x_bg_removed = self.x - self.bg_0 - (self.bg_1 * self.f)
            self.generate_fit_curve('x')
        elif what.lower() == 'y':
            popt, pcov = \
                curve_fit(Fit_Functions.lrtzY, self.f, self.y, p0 = guess, maxfev = 4000)
            self.A = popt[0]
            self.f0 = popt[1]
            self.df = popt[2]
            self.ph = popt[3]
            self.bg_0 = popt[4]
            self.bg_1 = popt[5]
            
            self.var_A = pcov[0,0]
            self.var_f0 = pcov[1,1]
            self.var_df = pcov[2,2]
            self.var_ph = pcov[3,3]
            self.var_bg_0 = pcov[4,4]
            self.var_bg_1 = pcov[5,5]
         
            self.y_bg_removed = self.y - self.bg_0 - (self.bg_1 * self.f)
            self.generate_fit_curve('y')
        elif what.lower() == 'rr':
            # Inorder to fit /r, the data set needs to be generated by fitting
            # and removing the background from x and y. This checks to see if
            # this has already been done and if not does it.
            if self.rr_bg_removed.all() == 0:
                self.lrtzFit(guess,'both')
            
            #If you want to fitting rr requires no phase; if the phased is
            # guessed it is simply removed. Guessing the is necessary if you
            # need to generate rr by fitting x and y.
            if len(guess) == 6:
                del guess[3]
            
            dvar = self.rr_bg_removed
            func = Fit_Functions.lrtzRR
            popt, pcov = \
                curve_fit(Fit_Functions.lrtzRR, self.rr_bg_removed, dvar, p0 = guess, maxfev = 4000)
            self.A = popt[0]
            self.f0 = popt[1]
            self.df = popt[2]
            self.bg_0 = popt[3]
            self.bg_1 = popt[4]
            
            self.var_A = pcov[0,0]
            self.var_f0 = pcov[1,1]
            self.var_df = pcov[2,2]
            self.var_bg_0 = pcov[3,3]
            self.var_bg_0 = pcov[4,4]
            self.generate_fit_curve('rr')
        elif what.lower() == 'both':
            self.lrtzFit(guess, 'y')
            self.lrtzFit(guess, 'x')
            self.r_bg_removed = np.sqrt((self.x_bg_removed**2) + (self.y_bg_removed**2))
            self.rr_bg_removed = (self.x_bg_removed**2) + (self.y_bg_removed**2)

#------------------------------------------------------------------------------

    def generate_fit_curve(self, what):
        if self.device == 'mems':
            Fit_Functions = mems_fitting
        elif self.device == 'tf':
            Fit_Functions = tf_fitting
        if what == 'x':
            self.x_fit = Fit_Functions.lrtzX(self.f, self.A, self.f0, self.df,
                                            self.ph, self.bg_0, self.bg_1)
        elif what == 'y':
            self.y_fit = Fit_Functions.lrtzY(self.f, self.A, self.f0, self.df,
                                            self.ph, self.bg_0, self.bg_1)
        elif what == 'rr':
            self.rrFit = Fit_Functions.lrtzRR(self.f, self.A, self.f0, self.df,
                                            self.bg_0, self.bg_1)
        elif what == 'xy' or what == 'yx':
            self.x_fit = Fit_Functions.lrtzX(self.f, self.A, self.f0, self.df,
                                            self.ph, self.bg_0, self.bg_1)
            self.y_fit = Fit_Functions.lrtzY(self.f, self.A, self.f0, self.df,
                                            self.ph, self.bg_0, self.bg_1)

#------------------------------------------------------------------------------

    def print_attributes(self):
        idn = 'sweep number: ' + self.idn
        temp = "Temp: {0:.1f} +/- {1:.1f} mK".format(self.avg_mc_temp,self.std_mc_temp)
        amp = 'A = {0:.1f} +/- {1:.1f} µVrms'.format(self.A,self.var_A)
        f0 = 'f0 = {0:.3f} +/- {1:.3f} Hz'.format(self.f0,self.var_f0)
        df = 'df = {0:.3f} +/- {1:.3f} Hz'.format(self.df,self.var_df)
        ph = 'ph = {0:.2f} +/ {1:.2f} deg'.format(self.ph,self.var_ph)
        bg_0 = 'bg_0 = {0:.3f} +/ {1:.3f} µVrms'.format(self.bg_0,self.var_bg_0)
        bg_1 = 'bg_1 = {0:.3f} +/- {1:.3f} µVrms/Vpp'.format(self.bg_1,self.var_bg_1)
        print(idn + '\n' + temp + '\n' + amp + '\n' + f0 + '\n' + df + '\n' +
            ph + '\n' + bg_0 + '\n' + bg_1)

#------------------------------------------------------------------------------

    def addToFile(self):
        header=""
        if not sweep.fitFileLoc:
            print("No fit file specified")
            return
        if not os.path.isfile(sweep.fitFileLoc):
            f = open(sweep.fitFileLoc, "w+")
            header="sweep Number\tAverageTemperature\tTemperature\Deviation\t\
              Amplitude\tResonance Frequency\tResonance Width\tPhase\t\
              First Order bg\tZeroth Order bg\n\n" 
        else:
            f = open(sweep.fitFileLoc, "a+") 
        f.write("{}{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(header,
            self.idn, str(self.avgTemp), str(self.stdTemp), str(self.A),
            str(self.f0), str(self.df), str(self.ph), str(self.bg_1),
            str(self.bg_0)))
        f.close()


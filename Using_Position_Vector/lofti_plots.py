import argparse
import numpy as np
from scipy import stats
from numpy import tan, arctan, sqrt, cos, sin, arccos
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#import matplotlib.style
#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

################################## User input ##################################
system = 'GKTau'
directory = system+"_ofti_output_2019-02-26/"

size = 12

# Change this to True if the lofti fitter failed to collect all the results
# into one file at the end of the run:
Collect_into_one_file = True

### Input distance and observation date:

d_star,d_star_err = 129.89184864417751,0.7553743009048957

date = 2015.5

################################# Definitions ################################

def eccentricity_anomaly(E,e,M):
    '''Eccentric anomaly function'''
    return E - (e*np.sin(E)) - M

def solve(f, M0, e, h):
    ''' Newton-Raphson solver for eccentricity anomaly
    from https://stackoverflow.com/questions/20659456/python-implementing-a-numerical-equation-solver-newton-raphson
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
    Returns: nextE (float): converged solution for eccentric anomaly
    '''
    if M0 / (1.-e) - np.sqrt( ( (6.*(1-e)) / e ) ) <= 0:
        E0 = M0 / (1.-e)
    else:
        E0 = (6. * M0 / e) ** (1./3.)
    lastE = E0
    nextE = lastE + 10* h  # "different than lastX so loop starts OK
    number=0
    while (abs(lastE - nextE) > h) and number < 1001:  # this is how you terminate the loop - note use of abs()
        newY = f(nextE,e,M0) # just for debug... see what happens
        lastE = nextE
        nextE = lastE - newY / (1.-e*np.cos(lastE))  # update estimate using N-R
        number=number+1
        if number >= 1000:
            nextE = float('NaN')#This truncates the calculation if a solution hasn't been reached by 1000 iter.
    return nextE



########### Stats ################
def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    From: https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/master/hpd.py
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max

def write_stats(params,params_names,filename):
    k = open(filename, 'w')
    string = 'Parameter    Mean    Std    68% Min Cred Int    95% Min Cred Int'
    k.write(string + "\n")
    for i in range(len(params)):
        # 68% CI:
        sorts = np.sort(params[i])
        frac=0.683
        ci68 = calc_min_interval(sorts,(1-frac))
        frac=0.954
        ci95 = calc_min_interval(sorts,(1-frac))
        k = open(filename, 'a')
        string = params_names[i] + '    ' + str(np.mean(params[i])) + '    ' + str(np.std(params[i])) + '    ' +\
          str(ci68) + '    ' + str(ci95)
        k.write(string + "\n")
    k.close()


########### Plots ################
def plot_1d_hist(params,names,filename,bins,
                     tick_fs = 20,
                     label_fs = 25,
                     label_x_x=0.5,
                     label_x_y=-0.25,
                     figsize=(30, 5.5)
                ):
    plt.ioff()
    # Bin size fo 1d hists:
    bins=bins
    fig = plt.figure(figsize=figsize)
    for i in range(len(params)):
        ax = plt.subplot2grid((1,len(params)), (0,i))
        plt.hist(params[i],bins=bins,edgecolor='none',alpha=0.8)
        plt.tick_params(axis='both', left=False, top=False, right=False, bottom=True, \
                labelleft=False, labeltop=False, labelright=False, labelbottom=True, labelsize = tick_fs)
        plt.xticks(rotation=45)
        plt.xlabel(names[i],fontsize=label_fs)
        ax.get_xaxis().set_label_coords(label_x_x,label_x_y)
        if i == 3:
            plt.xticks((-100, -50, 0, 50, 100), ('250', '310', '0', '50', '100'))

    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close(fig)
    return fig

def plot_orbits(a1,T1,to1,e1,i1,w1,O1, filename, obsdate, plane='xy', 
                    ticksize = 10, 
                    labelsize = 12,
                    Xlabel = 'Dec (")', 
                    Ylabel = 'RA (")',
                    Zlabel = 'Z (")',
                    figsize = (7.5, 6.), 
                    axlim = 8,  
                    cmap='viridis',
                    colorlabel = 'Phase',
                    color = True,
                    colorbar = True,
               ):
    ''' Plot orbits in RA/Dec given a set of orbital elements
        Inputs:  Array of orbital elements to plot
            a [as]: semi-major axis in as
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            filename (string): filename for written out plot
            obsdate [decimal yr]: observation date
        Optional Args:
            ticksize, labelsize (int): fontsize for tick marks and axis labels
            ylabel, xlabel (string): axis labels
            figsize (tuple): figure size (width, height)
            axlim (int) [arcsec]: axis limits
            cmap (str): colormap
            colorlabel (str): label for colorbar
            color: True plots in color, False plots orbits in bw
            colorbar: True renders colorbar, False omits colorbar.  If color is set to False, 
                colorbar must also be set to False.
        Returns: figure
    '''
    fig = plt.figure(figsize=figsize)
    plt.scatter(0,0,color='orange',marker='*',s=300,zorder=10)
    plt.xlim(-axlim,axlim)
    plt.ylim(-axlim,axlim)
    plt.gca().invert_xaxis()
    plt.gca().tick_params(labelsize=ticksize)
    majorLocator   = MultipleLocator(10)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(5)
    plt.grid(ls=':')

    for a,T,to,e,i,w,O in zip(a1,T1,to1,e1,i1,w1,O1):
        times = np.linspace(obsdate,obsdate+T,5000)
        X,Y,Z = np.array([]),np.array([]),np.array([])
        E = np.array([])
        for t in times:
            n = (2*np.pi)/T
            M = n*(t-to)
            nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip([M],[e])]
            E = np.append(E,nextE)
        r1 = a*(1.-e*cos(E))
        f1 = sqrt(1.+e)*sin(E/2.)
        f2 = sqrt(1.-e)*cos(E/2.)
        f = 2.*np.arctan2(f1,f2)
        # orbit plane radius in as:
        r = (a*(1.-e**2))/(1.+(e*cos(f)))
        X1 = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
        Y1 = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
        Z1 = r * sin(w+f) * sin(i)
        X,Y,Z = np.append(X,X1),np.append(Y,Y1),np.append(Z,Z1)
        if plane == 'xy':
            if color == True:
                plt.scatter(Y,X,c=((times-obsdate)/T),cmap=cmap,s=3,lw=0)
            else:
                plt.plot(Y,X, color='black',alpha=0.4)
            plt.ylabel(Xlabel,fontsize=labelsize)
            plt.xlabel(Ylabel,fontsize=labelsize)
        if plane == 'xz':
            if color == True:
                plt.scatter(X,Z,c=((times-obsdate)/T),cmap=cmap,s=3,lw=0)
            else:
                plt.plot(X,Z, color='black',alpha=0.4)
            plt.ylabel(Zlabel,fontsize=labelsize)
            plt.xlabel(Xlabel,fontsize=labelsize)
        if plane == 'yz':
            if color == True:
                plt.scatter(Y,Z,c=((times-obsdate)/T),cmap=cmap,s=3,lw=0)
            else:
                plt.plot(Y,Z, color='black',alpha=0.4)
            plt.ylabel(Zlabel,fontsize=labelsize)
            plt.xlabel(Ylabel,fontsize=labelsize)
    if colorbar == True:
        plt.colorbar().set_label(colorlabel)
    plt.savefig(filename+'.pdf', format='pdf')
    plt.savefig(filename+'.jpg', format='jpg', dpi=300)
    plt.savefig(filename+'.png', format='png', dpi=300)
    plt.close(fig)
    return fig


def plot_orbits3d(a1,T1,to1,e1,i1,w1,O1, filename, obsdate, plane='xy', 
                    ticksize = 10, 
                    labelsize = 12,
                    Xlabel = 'Dec (")', 
                    Ylabel = 'RA (")',
                    Zlabel = 'Z (")',
                    figsize = (7.5, 6.), 
                    axlim = 4,  
                    cmap='viridis',
                    colorlabel = 'Phase',
                    color = True,
                    colorbar = True,
               ):
    ''' Plot orbits in RA/Dec given a set of orbital elements
        Inputs:  Array of orbital elements to plot
            a [as]: semi-major axis in as
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            filename (string): filename for written out plot
            obsdate [decimal yr]: observation date
        Optional Args:
            ticksize, labelsize (int): fontsize for tick marks and axis labels
            ylabel, xlabel (string): axis labels
            figsize (tuple): figure size (width, height)
            axlim (int) [arcsec]: axis limits
            cmap (str): colormap
            colorlabel (str): label for colorbar
            color: True plots in color, False plots orbits in bw
            colorbar: True renders colorbar, False omits colorbar.  If color is set to False, 
                colorbar must also be set to False.
        Returns: figure
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0,0,0,color='orange',marker='*',s=300,zorder=10)
    plt.xlim(axlim,-axlim)
    plt.ylim(-axlim,axlim)
    ax.set_zlim(-axlim,axlim)
    #plt.zlim(-axlim,axlim)
    ax.set_ylabel('Dec (")')
    ax.set_xlabel('RA (")')
    ax.set_zlabel('Z (")')
    plt.gca().tick_params(labelsize=ticksize)
    majorLocator   = MultipleLocator(10)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(5)
    plt.grid(ls=':')

    for a,T,to,e,i,w,O in zip(a1,T1,to1,e1,i1,w1,O1)[0:25]:
        times = np.linspace(obsdate,obsdate+T,4000)
        X,Y,Z = np.array([]),np.array([]),np.array([])
        E = np.array([])
        for t in times:
            n = (2*np.pi)/T
            M = n*(t-to)
            nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip([M],[e])]
            E = np.append(E,nextE)
        r1 = a*(1.-e*cos(E))
        f1 = sqrt(1.+e)*sin(E/2.)
        f2 = sqrt(1.-e)*cos(E/2.)
        f = 2.*np.arctan2(f1,f2)
        # orbit plane radius in as:
        r = (a*(1.-e**2))/(1.+(e*cos(f)))
        X1 = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
        Y1 = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
        Z1 = r * sin(w+f) * sin(i)
        X,Y,Z = np.append(X,X1),np.append(Y,Y1),np.append(Z,Z1)
        ax.scatter(Y,X,Z,c=((times-obsdate)/T),cmap=cmap,s=3,lw=0)
    #if colorbar == True:
    #    plt.colorbar().set_label(colorlabel)
    plt.savefig(filename+'.pdf', format='pdf')
    plt.savefig(filename+'.jpg', format='jpg', dpi=300)
    plt.savefig(filename+'.png', format='png', dpi=300)
    return fig

################################# Begin script ##########################

files = directory+system+"_accepted"

if Collect_into_one_file == True:
    q = open(files, 'w')
    ## Prepare fitter output:
    # Collect into one file:
    for i in range(size):
        # Collect all the outputs into one file:
        dat = np.loadtxt(open(files+'_'+str(i),"rb"),delimiter='   ',ndmin=2)
        a,T,to,e,i,w,O,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]
        q = open(files, 'a')
        for a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,c,A,dice):
            string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1]])
            q.write(string + "\n")
        q.close()

    # Reperform accept/reject step with the min chi-squared from all processes:
    dat = np.loadtxt(open(files,"rb"),delimiter='   ',ndmin=2)
    a,T,to,e,i,w,O,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]

    chi_min = np.min(c)
    print 'Minimum chi^2 found: ',chi_min
    q = open(files, 'w')
    for a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,c,A,dice):
        delta_chi1 = -(c1-chi_min)/2.0
        AA = np.exp(delta_chi1)
        if AA > dice1:
            string = '   '.join([str(p) for p in [a1,T1,to1,e1, i1, w1,O1,c1,AA,dice1]])
            q.write(string + "\n")

    q.close()

dat = np.loadtxt(open(files,"rb"),delimiter='   ',ndmin=2)
num=dat.shape[0]

# Read in final parameter arrays:
a,T,to,e,i_deg,w_deg,O_deg,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]
i,w,O = np.radians(i_deg),np.radians(w_deg),np.radians(O_deg)
print a.shape

a_au=a*d_star
periastron = (1.-e)*a_au

### Set what parameters to plot in the histograms:
a_au2 = a_au[np.where(a_au<10000)]
to2 = to[np.where(to>-100000)]

#a_au2,to2 = a_au,to

w_temp = w_deg
for j in range(len(w_deg)):
    if w_temp[j] > 180:
        w_temp[j] = w_temp[j] - 360.
    else:
        pass

plot_params = [a_au2,e,i_deg,w_temp,O_deg,to2,periastron]
plot_params_names = [r"$a \; (AU)$",r"$e$",r"$ i \; (deg)$",r"$ \omega \; (deg)$",r"$\Omega \; (deg)$",r"$T_0 \; (yr)$",\
                         r"$a\,(1-e) \; (AU)$"]

######################### Do the things ###################################

# Write out stats:
print 'Writing out stats'
stats_name = directory+system+'_stats'
write_stats(plot_params,plot_params_names,stats_name)

# Plot 1-D histograms
print 'Making histograms'
output_name = directory + system+"_hists.pdf"
plot_1d_hist(plot_params,plot_params_names,output_name,50)

# Plot 2-D orbits
print 'Plotting orbits'
# Select random orbits from sample:
index = np.random.choice(range(0,dat.shape[0]),replace=False,size=100)
a1,T1,to1,e1,i1,w1,O1 = a[index],T[index],to[index],e[index],i[index],w[index],O[index]


# Plot those orbits:
# RA/Dec plane:
print 'XY plane'
output_name = directory + system+"_orbits"
plot_orbits(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = 35)

# X/Z plane:
print 'XZ plane'
output_name = directory + system+"_orbits_xz"
plot_orbits(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = 35, plane = 'xz')

# Y/Z plane:
print 'YZ plane'
output_name = directory + system+"_orbits_yz"
plot_orbits(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = 35, plane = 'yz')

# 3D
print '3D'
output_name = directory + system+"_orbits_3d"
plot_orbits3d(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = 35)




print 'Done'


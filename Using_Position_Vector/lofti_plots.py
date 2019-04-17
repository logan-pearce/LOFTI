'''
############################ LOFTI Plots ##############################
                     written by Logan Pearce, 2019
#######################################################################
    Take the output of the LOFTI algorithm and generate statistics, histograms,
    and nice plots for the accepted orbital configurations

# Requires:
#   python packages astropy, numpy, scipy, maplotplotlib, mpl_toolkits
#   output files from OFTI algorithm, either individual process outputs or
#   colelcted into one file.
#
# Input:
#   Directory containing OFTI output
#   Number of processes used to conduct the fit
#
# Output:
#      stats: Text file of statistics for each parameter
#      hists: PDF of 1d histograms of orbital parameter distributions
#      orbits: PDF/JPGs of the three 2-d projections of 100 randomly selected orbits
#      orbits_3d: PDF/JPG of a 3d plot of 25 randomly selected orbits
#
# usage: lofti_plots.py [-h] [-s SIZE] [-f COLLECT_INTO_ONE_FILE]
                      system directory

positional arguments:
  system                name of the system under study to use for labeling
                        output files
  directory             the path to the directory containing lofti results

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  Number of processes. default = 6
  -f COLLECT_INTO_ONE_FILE, --collect_into_one_file COLLECT_INTO_ONE_FILE
                        Set to True if the lofti fitting process did not terminate on its own. Default =
                        False
  -m PLT_STYLE, --plt_style PLT_STYLE
                        Name of matplotlib style sheet to use in plots.
                        Default = mpl default
  -d DATE, --date DATE
                        Observation date for plots.  Default = 2015.5 (Gaia DR2 date)
  -a AXLIM, --axlim AXLIM
                        Axlim for 2d plots (arcsec).  Default = 6 

example:
    python lofti_plots.py GKTau GKTau_ofti_output_2019-03-19 -s 48  <- Generate plots
         for the DS Tuc system in the directory GKTau_ofti_output_2019-03-19, which was fit using
         48 parallel processes, and has already been consolidated into one file by the OFTI script
    python lofti_plots.py GKTau GKTau_ofti_output_2019-03-19 -s 48 -f True <- Generate plots
         for the DS Tuc system in the directory GKTau_ofti_output_2019-03-19, which was fit using
         48 parallel processes, but which has not been collected into one file by the terminating OFTI script.
'''

import argparse
import numpy as np
from scipy import stats
from numpy import tan, arctan, sqrt, cos, sin, arccos
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#import matplotlib.style
#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import astropy.units as u


################################## User input ##################################

# Pull out arguments:
parser = argparse.ArgumentParser()
# Required positional arguments:
parser.add_argument("system", help="name of the system under study to use for labeling output files", type=str)
parser.add_argument("directory", help="the path to the directory containing lofti results", type=str)
# Optional positional arguments"
parser.add_argument("-s","--size", help="Number of processes. default = 6",type=str)
parser.add_argument("-f","--collect_into_one_file", help="Set to True if the lofti fitting process did\
     not terminate on its own. Default = False",type=str)
parser.add_argument("-m","--plt_style", help="Name of matplotlib style sheet to use in plots.  Default = mpl default",type=str)
parser.add_argument("-d","--date", help="Observation date.  Enter reference date for astrometry fits. \
                     Default = 2015.5 (Gaia DR2 date)",type=float)
parser.add_argument("-a","--axlim", help="Axlim (arcsec)",type=float)

args = parser.parse_args()

if args.plt_style:
    plt.style.use(args.plt_style)

directory = args.directory+'/'
system = args.system

if args.size:
    size = np.int_(args.size)
else:
    size = 6
    print 'Using default size = 6'

# Change this to True if the lofti fitter failed to collect all the results
# into one file at the end of the run:
Collect_into_one_file = False

if args.collect_into_one_file:
    Collect_into_one_file = True

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
def mode(array):
    n, bins = np.histogram(array, 10000, density=True)
    max_bin = np.max(n)
    bin_inner_edge = np.where(n==max_bin)[0]
    bin_outer_edge = np.where(n==max_bin)[0]+1
    # value in the middle of the highest bin:
    mode=(bins[bin_outer_edge] - bins[bin_inner_edge])/2 + bins[bin_inner_edge]
    return mode[0]

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
    string = 'Parameter    Mean    Std    Mode    68% Min Cred Int    95% Min Cred Int'
    k.write(string + "\n")
    k.close()
    for i in range(len(params)):
        # 68% CI:
        sorts = np.sort(params[i])
        frac=0.683
        ci68 = calc_min_interval(sorts,(1-frac))
        # 95% CI:
        frac=0.954
        ci95 = calc_min_interval(sorts,(1-frac))
        # Mode:
        m = mode(params[i])
        # Write it out:
        k = open(filename, 'a')
        string = params_names[i] + '    ' + str(np.mean(params[i])) + '    ' + str(np.std(params[i])) + '    ' +\
          str(m) + '    ' + str(ci68) + '    ' + str(ci95)
        k.write(string + "\n")
        k.close()

def to_si(mas,mas_yr,d):
    '''Convert from mas -> km and mas/yr -> km/s
        Input: 
         mas (array) [mas]: distance in mas
         mas_yr (array) [mas/yr]: velocity in mas/yr
         d (float) [pc]: distance to system in parsecs
        Returns:
         km (array) [km]: distance in km
         km_s (array) [km/s]: velocity in km/s
    '''
    km = ((mas*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = ((mas_yr*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = (km_s.value)*(u.km/u.yr).to(u.km/u.s)
    return km.value,km_s

def calc_XYZ(a,T,to,e,i,w,O,date):
    ''' Compute projected on-sky position only of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point. 
        Inputs:
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
        Returns: X, Y, and Z coordinates [as] where +X is in the reference direction (north) and +Y is east, and +Z
            is towards observer
    '''
    n = (2*np.pi)/T
    M = n*(date-to)
    nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    E = np.array(nextE)
    #E = solve(eccentricity_anomaly, M,e, 0.001)
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    # orbit plane radius in as:
    r = (a*(1.-e**2))/(1.+(e*cos(f)))
    X = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
    Y = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
    Z = r * sin(w+f)*sin(i)
    return X,Y,Z

def calc_velocities(a,T,to,e,i,w,O,date,dist):
    ''' Compute 3-d velocity of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  Uses my eqns derived from Seager 
        Exoplanets Ch2.
        Inputs:
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            m_tot [Msol]: total system mass
        Returns: X dot, Y dot, Z dot three dimensional velocities [km/s]
    '''
    # convert to km:
    a_km = to_si(a*1000.,0.,dist)
    a_km = a_km[0]
    
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    E = np.array(nextE)
    #E = solve(eccentricity_anomaly, M,e, 0.001)
    r1 = a*(1.-e*cos(E))
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    
    # Compute velocities:
    rdot = ( (n*a_km) / (np.sqrt(1-e**2)) ) * e*sin(f)
    rfdot = ( (n*a_km) / (np.sqrt(1-e**2)) ) * (1 + e*cos(f))
    Xdot = rdot * (cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i)) + \
           rfdot * (-cos(O)*sin(w+f) - sin(O)*cos(w+f)*cos(i))
    Ydot = rdot * (sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i)) + \
           rfdot * (-sin(O)*sin(w+f) + cos(O)*cos(w+f)*cos(i))
    Zdot = ((n*a_km) / (np.sqrt(1-e**2))) * sin(i) * (cos(w+f) + e*cos(w))
    
    Xdot = Xdot*(u.km/u.yr).to((u.km/u.s))
    Ydot = Ydot*(u.km/u.yr).to((u.km/u.s))
    Zdot = Zdot*(u.km/u.yr).to((u.km/u.s))
    return Xdot,Ydot,Zdot

def calc_accel(a,T,to,e,i,w,O,date,dist):
    ''' Compute 3-d acceleration of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  
        Inputs:
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
        Returns: X ddot, Y ddot, Z ddot three dimensional velocities [m/s/yr]
    '''
    # convert to km:
    a_km = to_si(a*1000.,0.,dist)[0]
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    E = np.array(nextE)
    #E = solve(eccentricity_anomaly, M,e, 0.001)
    # r and f:
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    r = (a_km*(1-e**2))/(1+e*cos(f))
    # Time derivatives of r, f, and E:
    Edot = n/(1-e*cos(E))
    rdot = e*sin(f)*((n*a_km)/(sqrt(1-e**2)))
    fdot = ((n*(1+e*cos(f)))/(1-e**2))*((sin(f))/sin(E))
    # Second time derivatives:
    Eddot = ((-n*e*sin(f))/(1-e**2))*fdot
    rddot = a_km*e*cos(E)*(Edot**2) + a_km*e*sin(E)*Eddot
    fddot = Eddot*(sin(f)/sin(E)) - (Edot**2)*(e*sin(f)/(1-e*cos(E)))
    # Positional accelerations:
    Xddot = (rddot - r*fdot**2)*(cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i)) + \
            (-2*rdot*fdot - r*fddot)*(cos(O)*sin(w+f) + sin(O)*cos(w+f)*cos(i))
    Yddot = (rddot - r*fdot**2)*(sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i)) + \
            (2*rdot*fdot + r*fddot)*(sin(O)*sin(w+f) + cos(O)*cos(w+f)*cos(i))
    Zddot = sin(i)*((rddot - r*fdot**2)*sin(w+f) + (2*rdot*fdot+ r*fddot*cos(w+f)))
    return Xddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr)), Yddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr)), \
                    Zddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr))

################################ Plots ##################################
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
        plt.colorbar().set_label(colorlabel, fontsize=labelsize)
    plt.savefig(filename+'.pdf', format='pdf')
    plt.savefig(filename+'.png', format='png', dpi=300)
    plt.close(fig)
    return fig


def plot_orbits3d(a1,T1,to1,e1,i1,w1,O1, filename, obsdate, plane='xy',
                    num_orbits = 25,
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

    for a,T,to,e,i,w,O in zip(a1,T1,to1,e1,i1,w1,O1)[0:num_orbits]:
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
    plt.savefig(filename+'.png', format='png', dpi=300)
    return fig

def plot_observables_hist(a,T,to,e,i,w,O,date,dist, filename,
                    bins = 50,
                    ticksize = 10, 
                    labelsize = 12,
                    figsize = (7.5, 6.)
                    ):
    ''' Plot histograms of posteriors of velocities and acceleration in the X, Y, and Z directions
        Inputs:
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
        Returns: 
            Histograms in:
            X, Y, Z positions in plane of the sky [mas],
            X dot, Y dot, Z dot three dimensional velocities [km/s]
            X ddot, Y ddot, Z ddot 3d accelerations in [m/s/yr]
    '''
    ddot = calc_accel(a,T,to,e,i,w,O,2015.5,44.)
    dot = calc_velocities(a,T,to,e,i,w,O,2015.5,44.)
    pos = calc_XYZ(a,T,to,e,i,w,O,2015.5)
    plt.ioff()
    # Bin size fo 1d hists:
    bins=bins

    # Plot X:
    fig = plt.figure(figsize=figsize)
    plt.hist(pos[0],bins=50)
    plt.xlabel(r'$X$ [mas]')
    plt.tight_layout()
    plt.savefig(filename+'_X.pdf', format='pdf')
    plt.close(fig)
    # Plot Y:
    fig = plt.figure(figsize=figsize)
    plt.hist(pos[1],bins=50)
    plt.xlabel(r'$Y$ [mas]')
    plt.tight_layout()
    plt.savefig(filename+'_Y.pdf', format='pdf')
    plt.close(fig)
    # Plot Z:
    fig = plt.figure(figsize=figsize)
    plt.hist(pos[2],bins=50)
    plt.xlabel(r'$Z$ [mas]')
    plt.tight_layout()
    plt.savefig(filename+'_Z.pdf', format='pdf')
    plt.close(fig)
    # Plot Xdot:
    fig = plt.figure(figsize=figsize)
    plt.hist(dot[0],bins=50)
    plt.xlabel(r'$\dot{X}$ [km/s]')
    plt.tight_layout()
    plt.savefig(filename+'_xdot.pdf', format='pdf')
    plt.close(fig)
    # Plot Ydot:
    fig = plt.figure(figsize=figsize)
    plt.hist(dot[1],bins=50)
    plt.xlabel(r'$\dot{Y}$ km/s]')
    plt.tight_layout()
    plt.savefig(filename+'_ydot.pdf', format='pdf')
    plt.close(fig)
    # Plot Zdot:
    fig = plt.figure(figsize=figsize)
    plt.hist(dot[2],bins=50)
    plt.xlabel(r'$\dot{Z}$ [km/s]')
    plt.tight_layout()
    plt.savefig(filename+'_zdot.pdf', format='pdf')
    plt.close(fig)
    # Plot Xddot:
    fig = plt.figure(figsize=figsize)
    plt.hist(ddot[0],bins=50)
    plt.xlabel(r'$\ddot{X}$ [m/s/yr]')
    plt.tight_layout()
    plt.savefig(filename+'_xddot.pdf', format='pdf')
    plt.close(fig)
    # Plot Yddot:
    fig = plt.figure(figsize=figsize)
    plt.hist(ddot[1],bins=50)
    plt.xlabel(r'$\ddot{Y}$ [m/s/yr]')
    plt.tight_layout()
    plt.savefig(filename+'_yddot.pdf', format='pdf')
    plt.close(fig)
    # Plot Zddot:
    fig = plt.figure(figsize=figsize)
    plt.hist(ddot[2],bins=50)
    plt.xlabel(r'$\ddot{Z}$ [m/s/yr]')
    plt.tight_layout()
    plt.savefig(filename+'_zddot.pdf', format='pdf')
    plt.close(fig)
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
    '''
    q = open(files, 'w')
    for a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,c,A,dice):
        delta_chi1 = -(c1-chi_min)/2.0
        AA = np.exp(delta_chi1)
        if AA > dice1:
            string = '   '.join([str(p) for p in [a1,T1,to1,e1, i1, w1,O1,c1,AA,dice1]])
            q.write(string + "\n")

    q.close()'''


dat = np.loadtxt(open(files,"rb"),delimiter='   ',ndmin=2)
num=dat.shape[0]

# Read in final parameter arrays:
a,T,to,e,i_deg,w_deg,O_deg,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]
i,w,O = np.radians(i_deg),np.radians(w_deg),np.radians(O_deg)

### Input distance and observation date:

d_star,d_star_err = 139.69, 1.32
print 'I have distance to system as:',d_star,d_star_err
yn = raw_input('Is that right? Enter y for yes, n for no.')
if yn == 'n':
    d_star = float(raw_input('Enter d_star: '))

if args.date:
    date = float(args.date)
else:
    date = 2015.5
    
print 'I have obs date to system as:',date
yn = raw_input('Is that right? Enter y for yes, n for no.')
if yn == 'n':
    date = float(raw_input('Enter observation date: '))

a_au=a*d_star
periastron = (1.-e)*a_au

### Set what parameters to plot in the histograms:
a_au2 = a_au[np.where(a_au<400)]
to2 = to[np.where(a_au<400)]

w_temp = w_deg.copy()
for j in range(len(w_deg)):
    if w_temp[j] > 180:
        w_temp[j] = w_temp[j] - 360.
    else:
        pass

plot_params = [a_au2,e,i_deg,w_deg,O_deg,to2,periastron] 
plot_params_names = [r"$a \; (AU)$",r"$e$",r"$ i \; (deg)$",r"$ \omega \; (deg)$",r"$\Omega \; (deg)$",r"$T0 \; (yr)$",\
                         r"$a\,(1-e) \; (AU)$"]

######################### Do the things ###################################

# Write out stats:
print 'Writing out stats'
stats_name = directory+system+'_stats'
write_stats([a_au2,e,i_deg,w_temp,O_deg,to2,periastron],plot_params_names,stats_name)

# Plot 1-D histograms
print 'Making histograms'
output_name = directory + system+"_hists.pdf"
plot_1d_hist([a_au2,e,i_deg,w_temp,O_deg,to2,periastron],plot_params_names,output_name,50,tick_fs = 25,
                     label_fs = 30)

print 'Plotting observable posteriors'
os.system('mkdir '+directory+'observable_posteriors')
output_name = directory + 'observable_posteriors/' + system
plot_observables_hist(a,T,to,e,i,w,O,date,d_star,output_name)

# Plot 2-D orbits
print 'Plotting orbits'
# Select random orbits from sample:
index = np.random.choice(range(0,dat.shape[0]),replace=False,size=100)
a1,T1,to1,e1,i1,w1,O1 = a[index],T[index],to[index],e[index],i[index],w[index],O[index]

if args.axlim:
    axlim = np.int_(args.axlim)
else:
    axlim = 6

# Plot those orbits:
# RA/Dec plane:
print 'XY plane'
output_name = directory + system+"_orbits"
plot_orbits(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = axlim, ticksize = 15, 
                    labelsize = 20)

# X/Z plane:
print 'XZ plane'
output_name = directory + system+"_orbits_xz"
plot_orbits(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = axlim, plane = 'xz')

# Y/Z plane:
print 'YZ plane'
output_name = directory + system+"_orbits_yz"
plot_orbits(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = axlim, plane = 'yz')

# 3D
print '3D'
output_name = directory + system+"_orbits_3d"
plot_orbits3d(a1,T1,to1,e1,i1,w1,O1, output_name, date, axlim = axlim, num_orbits = 20)

print 'Done'


'''
############################ LOFTI Position Vector ##############################
                     written by Logan Pearce, 2019
#################################################################################
    Perform Logan's version of OFTI using one observation date and observations of 
    relative position, velocity, or acceleration

# Requires:
#   python packages astropy, numpy, mpi4py
#
# Input:
#   Manually enter the observations for the system into the script
#
# Output:
#      system_accepted: all orbits accepted by fit in the order:
#            a,T,to,e,i_deg,w_deg,O_deg,chi-squared,probability,dice roll
#
# usage: mpiexec -n number_of_processes python lofti_phasespace_DSTuc_mpi.py

example:
    mpiexec -n 12 python lofti_pv.py  <- run LOFTI on my laptop using 12 parallel processes
    ibrun python lofti_phasespace.py <- sbatch command to run LOFTI on TACC.  For Lonestar 5, set nodes=1
            and cores = 48
'''




import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.time import Time
from datetime import date, datetime
import os
import time as tm

## MPI:
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
## define the communicator:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ncor = size

deg_to_mas = 3600000.
mas_to_deg = 1./3600000.

### 
accept_min = 2200

######################### Observations ###########################
############# Input system observations here #####################
name = 'GKTau'

now = str(date.today())
# Make a directory to store all results:
if rank == 0:
    os.system('mkdir '+name+'_ofti_output_'+now)
directory = name+'_ofti_output_'+now
comm.barrier()

d = 2015.5
# A = GK Tau
# B = GI Tau

plxa, plxaerr = 7.736181763296462, 0.0434716242332522
plxb, plxberr = 7.662587421336056, 0.04604393474172906
# Mass reference GK Tau: https://arxiv.org/pdf/1706.03505.pdf
# GI Tau: https://iopscience.iop.org/article/10.3847/1538-4357/aa9e52/pdf
mA, mAerr = 0.79, 0.07
mB, mBerr = 0.6, 0.3

# Astrometry from Gaia DR2:
# (RA/Dec are reported in degrees, errors in mas)
RAa, RAaerr = 68.39404779762734, 0.04225973382885226*mas_to_deg
RAb, RAberr = 68.39194796115807, 0.04693034099138683*mas_to_deg
Deca, Decaerr = 24.351538070211536, 0.02838685021460219*mas_to_deg
Decb, Decberr = 24.354652459003447, 0.030860001768019497*mas_to_deg

# Proper motions (in mas/yr):
pmRAa, pmRAaerr = 7.752527330204136, 0.09501610992230507
pmRAb, pmRAberr = 5.882953921562138, 0.10768919956109295
pmDeca, pmDecaerr = -20.48313007512066, 0.06976340757304786
pmDecb, pmDecberr = -20.550925312475705, 0.07597187724061871


# Set to True is there are RV measurements to contribute to fit:
RV = False

# Radial velocity (km/s) from Gaia solution:
# NO RV for this system
 # Enter radial velocity + error for both objects:
#rva, rvaerr = np.array([]),np.array([])
#rvb, rvberr = np.array([]),np.array([])
#rv_rel,rv_err = rvb-rva, np.sqrt(rvaerr**2+rvberr**2)

# RV observation dates:
#rvbjd = np.array([])
#rvbjd = Time(rvbjd, format='jd')
#d_rv = rvbjd.decimalyear

######## Nothing else needs to be modified below this line #############
########################################################################

############ Compute observables constraints: ##################
# Compute Delta RA and Delta Dec:
raa_array = np.random.normal(RAa, RAaerr, 10000)
rab_array = np.random.normal(RAb, RAberr, 10000)
deca_array = np.random.normal(Deca, Decaerr, 10000)
decb_array = np.random.normal(Decb, Decberr, 10000)
ra_array = (rab_array*deg_to_mas - raa_array*deg_to_mas) * np.cos(np.radians(np.mean([Deca,Decb])))
deltaRA,deltaRA_err = np.mean(ra_array), np.std(ra_array)
dec_array = ((decb_array - deca_array)*u.deg).to(u.mas).value
deltaDec,deltaDec_err = np.mean(dec_array),np.std(dec_array)
# Compute vel of B relative to A
pmRA, pmRAerr = (pmRAb - pmRAa), np.sqrt( pmRAberr**2 + pmRAaerr**2 )
#*cos(np.radians(np.mean([Deca,Decb]))) <- no cos term because it's in angular separation units
pmDec, pmDecerr = pmDecb - pmDeca, np.sqrt( pmDecaerr**2 + pmDecberr**2 )

def distance(parallax,parallax_error):
    '''Computes distance from Gaia parallaxes using the Bayesian method of Bailer-Jones 2015.
    Input: parallax [mas], parallax error [mas]
    Output: distance [pc], 1-sigma uncertainty in distance [pc]
    '''
    # Compute most probable distance:
    L=1350 #parsecs
    # Convert to arcsec:
    parallax, parallax_error = parallax/1000., parallax_error/1000.
    # establish the coefficients of the mode-finding polynomial:
    coeff = np.array([(1./L),(-2),((parallax)/((parallax_error)**2)),-(1./((parallax_error)**2))])
    # use numpy to find the roots:
    g = np.roots(coeff)
    # Find the number of real roots:
    reals = np.isreal(g)
    realsum = np.sum(reals)
    # If there is one real root, that root is the  mode:
    if realsum == 1:
        gd = np.real(g[np.where(reals)[0]])
    # If all roots are real:
    elif realsum == 3:
        if parallax >= 0:
            # Take the smallest root:
            gd = np.min(g)
        elif parallax < 0:
            # Take the positive root (there should be only one):
            gd = g[np.where(g>0)[0]]
    
    # Compute error on distance from FWHM of probability distribution:
    from scipy.optimize import brentq
    rmax = 1e6
    rmode = gd[0]
    M = (rmode**2*np.exp(-rmode/L)/parallax_error)*np.exp((-1./(2*(parallax_error)**2))*(parallax-(1./rmode))**2)
    lo = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), 0.001, rmode)
    hi = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), rmode, rmax)
    fwhm = hi-lo
    # Compute 1-sigma from FWHM:
    sigma = fwhm/2.355
            
    return gd[0],sigma

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

# Distance
#db,da = distance(plxb,plxberr),distance(plxa,plxaerr)
da = distance(plxa,plxaerr)
d_star,d_star_err = np.mean([da[0]]),np.mean([da[1]])


# Positions in km:
h,g = to_si(np.array([deltaRA,deltaDec]),np.array([pmRA,pmDec]),d_star)
deltaRA_km, deltaDec_km = h[0],h[1]
pmRA_kms,pmDec_kms = g[0],g[1]
delta_err_km,pm_err_kms = to_si(np.array([ra_array,dec_array]),np.array([np.random.normal(pmRA, pmRAerr, 10000),\
                                                                         np.random.normal(pmDec, pmDecerr, 10000)]),\
                                d_star)
deltaRA_err_km,deltaDec_err_km = np.std(delta_err_km[0]),np.std(delta_err_km[1])
pmRA_err_kms, pmDec_err_kms = np.std(pm_err_kms[0]),np.std(pm_err_kms[1])

# masses:
maarray, mbarray = np.random.normal(mA,mAerr,1000),np.random.normal(mB,mBerr,1000)
muarray = (maarray*mbarray)/(maarray+mbarray)
m_red, m_red_err = np.mean(muarray),np.std(muarray)

m_tot_array = maarray + mbarray
m_tot, m_tot_err = mA + mB, np.sqrt((mAerr**2) + (mBerr**2))

# Separation and PA for use in scale and rotate:
def to_polar(RAa,RAb,Deca,Decb):
    ''' Converts RA/Dec [deg] of two binary components into separation and position angle of B relative 
        to A [mas, deg]
    '''
    dRA = (RAb - RAa) * np.cos(np.radians(np.mean([Deca,Decb])))
    dRA = (dRA*u.deg).to(u.mas)
    dDec = (Decb - Deca)
    dDec = (dDec*u.deg).to(u.mas)
    r = np.sqrt( (dRA ** 2) + (dDec ** 2) )
    p = (np.degrees( np.arctan2(dDec.value,-dRA.value) ) + 270.) % 360.
    p = p*u.deg
    return r, p

raa_array = np.random.normal(RAa, RAaerr, 10000)
rab_array = np.random.normal(RAb, RAberr, 10000)
deca_array = np.random.normal(Deca, Decaerr, 10000)
decb_array = np.random.normal(Decb, Decberr, 10000)
rho_array,pa_array = to_polar(raa_array,rab_array,deca_array,decb_array)
rho, rhoerr  = np.mean(rho_array).value, np.std(rho_array).value
pa,paerr = np.mean(pa_array).value,np.std(pa_array).value

# RV at Gaia obs date:
if RV == True and float(rv_rel.shape[0])>1:
    def line(x,m,b):
        return m*x+b
    from scipy.optimize import curve_fit
    fit, pcov = curve_fit(line, d_rv,rv_rel,sigma=rv_err)
    vz_kms = fit[0]*d + fit[1]
    vz_kms_err = np.sqrt(pcov[0,0])
elif RV == True:
    vz_kms,vz_kms_err = rv_rel[0],rv_err[0]

if rank == 0:
    print 'Finished computing constraints:'
    print 'Delta RA, err in mas:',deltaRA,deltaRA_err
    print 'Delta Dec, err in mas:',deltaDec,deltaDec_err
    print 
    print 'pmDec, err in km/s:',pmDec_kms,pmDec_err_kms
    print 'pmRA, err in km/s:',pmRA_kms,pmRA_err_kms
    if RV == True:
        print 'rv at Gaia obs date in km/s:',vz_kms,vz_kms_err
    print
    print 'Starting OFTI run'

############################### OFTI ###############################

from numpy import tan, arctan, sqrt, cos, sin, arccos

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

def draw_priors(number):
    """Draw a set of orbital elements from proability distribution functions.
        Input: number - number of orbits desired to draw elements for
        Returns:
            a [as]: semi-major axis - set at 100 AU inital value
            T [yr]: period
            const: constant defining orbital phase of observation
            to [yr]: epoch of periastron passage
            e: eccentricity
            i [rad]: inclination in radians
            w [rad]: arguement of periastron
            O [rad]: longitude of nodes - set at 0 initial value
            m1 [Msol]: total system mass in solar masses
            dist [pc]: distance to system
    """
    #m1 = np.random.normal(m_tot,m_tot_err,number)
    #dist = np.random.normal(d_star,d_star_err,number)
    m1 = m_tot
    dist = d_star
    # Fixing and initial semi-major axis:
    a_au=100.0
    a_au=np.linspace(a_au,a_au,number)
    T = np.sqrt((np.absolute(a_au)**3)/np.absolute(m1))
    a = a_au/dist #semimajor axis in arcsec

    # Fixing an initial Longitude of ascending node in radians:
    O = np.radians(0.0)  
    O=[O]*number

    # Randomly generated parameters:
    #to = Time of periastron passage in years:
    const = np.random.uniform(0.0,1.0,number)
    #^ Constant that represents the ratio between (reference epoch minus to) over period.  Because we are scaling
    #semi-major axis, period will also scale, and epoch of periastron passage will change as a result.  This ratio
    #will remain constant however, so we can use it scale both T and to appropriately.
    to = d-(const*T)

    # Eccentricity:
    e = np.random.uniform(0.0,1.0,number)
    # Inclination in radians:
    cosi = np.random.uniform(-1.0,1.0,number)  #Draws sin(i) from a uniform distribution.  Inclination
    # is computed as the arccos of cos(i):
    i = np.arccos(cosi)
    # Argument of periastron in degrees:
    w = np.random.uniform(0.0,360.0,number)
    w = np.radians(w) #convert to radians for calculations
    return a,T,const,to,e,i,w,O,m1,dist

####### Scale and rotate to Gaia epoch:
def scale_and_rotate(X,Y):
    ''' Generates a new semi-major axis, period, epoch of peri passage, and long of peri for each orbit
        given the X,Y plane of the sky coordinates for the orbit at the date of the reference epoch
    '''
    r_model = np.sqrt((X**2)+(Y**2))
    rho_rand = np.random.normal(rho/1000.,rhoerr/1000.) #This generates a gaussian random to 
    #scale to that takes observational uncertainty into account.  #convert to arcsec
    #rho_rand = rho/1000. 
    a2 = a*(rho_rand/r_model)  #<- scaling the semi-major axis
    #New period:
    a2_au=a2*dist #convert to AU for period calc:
    T2 = np.sqrt((np.absolute(a2_au)**3)/np.absolute(m1))
    #New epoch of periastron passage
    to2 = d-(const*T2)

    # Rotate:
    # Rotate PA:
    PA_model = (np.degrees(np.arctan2(X,-Y))+270)%360 #corrects for difference in zero-point
    #between arctan function and ra/dec projection
    PA_rand = np.random.normal(pa,paerr) #Generates a random PA within 1 sigma of observation
    #PA_rand = pa
    #New omega value:
    O2=[]
    for PA_i in PA_model:
        if PA_i < 0:
            O2.append((PA_rand-PA_i) + 360.)
        else:
            O2.append(PA_rand-PA_i)
    # ^ This step corrects for the fact that the arctan gives the angle from the +x axis being zero,
    #while for RA/Dec the zero angle is +y axis.  

    #Recompute model with new rotation:
    O2 = np.array(O2)
    O2 = np.radians(O2)
    return a2,T2,to2,O2

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
            dist [pc]: distance to system
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


def calc_OFTI(a,T,to,e,i,w,O,d,dist):
    '''Does the same as calc_orbit but includes the OFTI scale and rotate step, and returns
       scaled and rotated orbital parameters
    '''
    X1,Y1,Z1 = calc_XYZ(a,T,to,e,i,w,O,d)
    a2,T2,to2,O2 = scale_and_rotate(X1,Y1)
    X2,Y2,Z2 = calc_XYZ(a2,T2,to2,e,i,w,O2,d)
    X2,Y2,Z2 = (X2*u.arcsec).to(u.mas).value, (Y2*u.arcsec).to(u.mas).value, (Z2*u.arcsec).to(u.mas).value
    Xdot,Ydot,Zdot = calc_velocities(a2,T2,to2,e,i,w,O2,d,dist)
    Xddot,Yddot,Zddot = calc_accel(a2,T2,to2,e,i,w,O2,d,dist)
    i,w,O2 = np.degrees(i),np.degrees(w),np.degrees(O2)
    return X2,Y2,Z2,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2


########### Perform initial run to get initial chi-squared:
# Draw random orbits:
a,T,const,to,e,i,w,O,m1,dist = draw_priors(10000)
# Compute positions and velocities:
X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2 = calc_OFTI(a,T,to,e,i,w,O,d,dist)
# Compute chi squared:
dr = (deltaRA - Y)/(deltaRA_err)
dd = (deltaDec - X)/(deltaDec_err)
dxdot = (pmDec_kms - Xdot)/(pmDec_err_kms)
dydot = (pmRA_kms - Ydot)/(pmRA_err_kms)
if RV == True:
    dzdot = (vz_kms - Zdot)/(vz_kms_err)
    chi = dr**2 + dd**2 + dxdot**2 + dydot**2 + dzdot**2
else:
    chi = dr**2 + dd**2 + dxdot**2 + dydot**2
chi_min = np.nanmin(chi)
print 'Chi-min:',chi_min

##### Start loop #####
start=tm.time()
num = 0
loop_count = 0
while num <= accept_min:
    # Draw random orbits:
    a,T,const,to,e,i,w,O,m1,dist = draw_priors(10000)
    # Compute position and velocities:
    X,Y,Z,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2 = calc_OFTI(a,T,to,e,i,w,O,d,dist)
    # Compute chi squared:
    dr = (deltaRA - Y)/(deltaRA_err)
    dd = (deltaDec - X)/(deltaDec_err)
    dxdot = (pmDec_kms - Xdot)/(pmDec_err_kms)
    dydot = (pmRA_kms - Ydot)/(pmRA_err_kms)
    if RV == True:
        dzdot = (vz_kms - Zdot)/(vz_kms_err)
        chi = dr**2 + dd**2 + dxdot**2 + dydot**2 + dzdot**2
    else:
        chi = dr**2 + dd**2 + dxdot**2 + dydot**2

    # Accept/reject:
    delta_chi = -(chi-chi_min)/2.0
    A = np.exp(delta_chi)
    rand = np.random.uniform(0.0,1.0,10000)  #The random "dice roll" to determine acceptable probability
    accepted = np.where(A > rand)
    
    # Write to file:
    parameters = np.zeros((10,10000))
    parameters[0,:],parameters[1,:],parameters[2,:],parameters[3,:],parameters[4,:],parameters[5,:], \
      parameters[6,:],parameters[7,:],parameters[8,:],parameters[9,:] = a2,T2,to2,e,i,w,O2,chi,A,rand
    parameters=np.transpose(parameters)
    k = open(directory+'/'+name+'_accepted_'+str(rank), 'a')
    for params in parameters[accepted]:
        string = '   '.join([str(p) for p in params])
        k.write(string + "\n")
    k.close()

    ############### Update chi-min ###################
    new_min =  min(chi)
	#determines the minimum chi from this loop
    if new_min < chi_min:
		chi_min = new_min
		#print 'Found new chi min: ',chi_min
		found_new_chi_min = 'yes'
		#kk = open(directory+'/'+name+'_chimin_tracking', 'a')
		#kk.write(str(chi_min)+ "\n")
		#kk.close()
    else:
        found_new_chi_min = 'no'
	#if this minimum chi is less than the previously assigned chi_min, update the chi_min variable
	#to the new value, and write it out to this file. 
	
    if found_new_chi_min == 'yes' and num!=0: 
		############## Recalculate old accepted orbits with new chi-min for acceptance #######
		dat = np.loadtxt(open(directory+'/'+name+"_accepted_"+str(rank),"rb"),delimiter='   ',ndmin=2)
		a,T,to,e,i,w,O,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]
		q = open(directory+'/'+name+'_accepted_'+str(rank), 'w')
		for a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,c,A,dice):
			delta_chi1 = -(c1-chi_min)/2.0
			AA = np.exp(delta_chi1)
			if AA > dice1:
				string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,c1,AA,dice1]])
				q.write(string + "\n")
			else:
				pass
		q.close()
	
		dat2 = np.loadtxt(open(directory+'/'+name+"_accepted_"+str(rank),"rb"),delimiter='   ',ndmin=2)
		num=dat2.shape[0]
    else:
		pass
	
	#This step is only for counting the total number of accepted orbits.  There's probably a better way but you know.
    mod2 = loop_count%10
    if mod2 == 0:
		dat2 = np.loadtxt(open(directory+'/'+name+"_accepted_"+str(rank),"rb"),delimiter='   ',ndmin=2)
		num=dat2.shape[0]
		print 'Loop count rank ',rank,': ',loop_count
		print "Rank ",rank," has found ",num,"accepted orbits"
		print 'Chi-min:',chi_min
	
    loop_count = loop_count + 1  #Iterate the counter
    found_new_chi_min = 'no' #reset the re-evaluator for the next loop
        
	#################################### End loop ###########################################
comm.barrier()
if rank == 0:
    print 'Found enough orbits, finishing up...'
    for i in range(size):
        # Collect all the outputs into one file:
        dat = np.loadtxt(open(directory+'/'+name+"_accepted_"+str(i),"rb"),delimiter='   ',ndmin=2)
        a,T,to,e,i,w,O,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]
        q = open(directory+'/'+name+'_accepted', 'a')
        for a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,c,A,dice):
            delta_chi1 = -(c1-chi_min)/2.0
            AA = np.exp(delta_chi1)
            if AA > dice1:
                string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,c1,AA,dice1]])
                q.write(string + "\n")
            else:
                pass
        q.close()
    # Reperform the acceptance step one last time with the lowest chi-min off all processes:
    dat = np.loadtxt(open(directory+'/'+name+"_accepted","rb"),delimiter='   ',ndmin=2)
    a,T,to,e,i,w,O,c,A,dice = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8],dat[:,9]
    chi_min = np.min(c)
    print 'Minimum chi^2 found: ',chi_min
    q = open(directory+'/'+name+'_accepted', 'w')
    for a1,T1,to1,e1,i1,w1,O1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,c,A,dice):
		delta_chi1 = -(c1-chi_min)/2.0
		AA = np.exp(delta_chi1)
		if AA > dice1:
			string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,c1,AA,dice1]])
			q.write(string + "\n")
		else:
			pass
    q.close()

    dat2 = np.loadtxt(open(directory+'/'+name+"_accepted","rb"),delimiter='   ',ndmin=2)
    num=dat2.shape[0]
else:
    pass		

if rank == 0:
    # Remove the individual files and keep just the aggregate:
    os.system('rm '+directory+'/*accepted_*')
    
if rank == 0:	
    print '.........done'
    print 'found total ',num,' orbits'
    stop=tm.time()
    time=stop-start
    from datetime import date
    
    print 'This operation took',time,'seconds'
    print 'and',time/3600.,'hours'
    
    #Write out to a log file:
    from datetime import date
    stop=tm.time()
    time=stop-start
    z = open(directory+'/'+name+'_log', 'a')
    string = str(date.today())
    string += ' took '
    string += str(time/3600.)
    string += ' hours to get '
    string += str(num)
    string += ' accepted orbits; '
    string += ' Tested '
    string += str(loop_count*10000*size)
    string += ' total permutations '
    z.write(string + "\n")
    z.close()
else:
    pass

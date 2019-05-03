###############################################################################
#                                                                             #
#                   The Orbits of Long Period Exoplanets:                     #
#                                  LOFTI:                                     #
#                      Logan's Orbits for the Impatient                       #
#                                                                             #
#                        Written by Logan A. Pearce (2017)                    #
#               Method Developed by Sarah Blunt et. al. (2017)                #
###############################################################################
#
# OFTI is a rejection sampling algorithm for fitting orbital elements to observation for systems
# with small orbit fractions observed.  The method was developed by Blunt et. al. and published
# in 2017.  Briefly, OFTI works by initially fixing two orbital parameters - semi-major axis and longitude
# of periastron - and randomly generating a value from prior probability distributions for the other 4
# parameters.  OFTI then scales and rotates the randomly generated orbit to best match observation, and
# performs a chi-squared determination to decide if the random orbit is accepted.  This version performs
# the calculation on batches of 10,000 orbits at a time.  This version also keeps a running track of the
# minimum chi-squared the algorithm has encountered, and periodically re-examines previously accepted
# orbits to determine if they should still be accepted, so it is regurlarly refining it acceptance criterion.
#
#
# MPI version is written to run using parallel processing on TACC Lonestar 5.
#
# Inputs:
#  - manual input of observation data (position angle and separation), errors, and observation epochs
# 
# Outputs:
#  - a directory named "name_ofti_output" containing:
#     - a text file of the parameters of each accepted orbit in the order:
#           - Semi-major axis (a) in arcsec
#           - Period (T) in years
#           - Time of periastron passage (to)
#           - Eccentricity (e)
#           - Inclination (i) in degrees
#           - Arguement of periastron (w) in degrees
#           - Longitude of periastron (O) in degrees
#           - Mass of the central star (m_star) in solar masses
#           - Distance to central star (d_star) from Earth in parsecs
#           - Calculated chi^2 value
#           - The probability of this orbit describing observations
#           - The random "dice roll" used in the acceptance criterion
#     - a text file containing the running track of the minimum chi^2 throughout the run
#     - a log file containing the details of each run
#
# From the terminal, execute as follows:
# mpiexec -n 6 python lofti_GSC6214_mpi.py


import numpy as np
from numpy import tan, arctan, sqrt, cos, sin, arccos
from numpy import random
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time as tm
import os
from datetime import date, datetime
from astropy.time import Time
##MPI STUFF
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

#######################################################################################################
############################################ Definitions  #############################################
#######################################################################################################

def derivative(f, x, h):
      return (f(x+h) - f(x-h)) / (2.0*h)

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

#Eccentricity anomaly equation for the numerical solver:
def eccentricity_anomaly(E,e,M):
    return E - (e*sin(E)) - M

##define the communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ncor = size
#######################################################################################################
#################################### Observation input: ###############################################
#######################################################################################################
#                                                                                                     #
#       In the r_obs array, enter the observed average separation for each                            #
#            epoch in chronological order in milliarcseconds                                          #
#       In the rerr array, enter the corresponding standard deviation on separation for each          #
#            epoch in milliarcseconds                                                                 #
#       In the PA_obs and terr arrays enter the observed position angle and std dev in degrees        #
#       In the d array, enter the observation dates as a decimal fraction of a year.                  #
#            ex: June 17th, 2008 = 2008.46                                                            #
#       For m_star and m_star_err, enter the star's mass and std dev in solar masses                  #
#       For d_star and d_star_err, enter the star's distance from Earth and std dev in parsecs        #
#       For system name, enter a string that will be used to label all output documents               #
#       For accept_min, enter the number of accepted orbits desired, which will terminate             #
#            the fitter                                                                               #
#                                                                                                     #
#######################################################################################################

# Stellar mass in solar masses:
m_star, m_star_err = 1.3,0.06

obs_dates = Time(['2013-08-07','2014-07-28','2014-11-30','2015-08-11',\
                  '2015-08-11'],scale='utc')
d = obs_dates.decimalyear

# Sep in mas
r_obs = np.array([1842.75,1842.80,1841.51,1841.24,1841.08])
rerr = np.array([0.1,0.18,0.20,0.14,0.21])
# pa in degrees
PA_obs = np.array([253.266,253.279,253.284,253.247,253.274])
terr = np.array([0.028,0.008,0.005,0.007,0.013])


# Stellar distance in parsecs:
d_star,d_star_err = 35.7, 0.5

# System name:
name = 'K444_test_astr'

# Number of accepted orbits desired:
accept_min = 2200


# NOTHING ELSE NEEDS TO BE CHANGED BELOW THIS LINE:
######################################################################################################
####################### Initial 10000 orbit loop to get initial chi_min: #############################
######################################################################################################

# Compute distance:
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

#d_star,d_star_err = distance(plx,plx_err)
if rank == 0:
    print 'I found the distance to be:',d_star,'+/-',d_star_err

now = str(date.today())

# Make a directory to store all results:
if rank == 0:
    os.system('mkdir '+name+'_ofti_output_'+now)
directory = name+'_ofti_output_'+now
comm.barrier()

if rank ==0:
    print 'Generating initial orbit sample...'
#Convert to RA/Dec:
r_obs,rerr = r_obs/1000, rerr/1000 #convert to arcseconds

PA_obs_rad = np.radians(PA_obs)
dec = r_obs*np.cos(PA_obs_rad)
ra = r_obs*np.sin(PA_obs_rad)

# Take the reference epoch as the one with the smallest total error:
total_error = np.sqrt(rerr**2 + terr**2)
ref = np.where(total_error == np.min(total_error))[0]


######################### Draw priors  #############################
# Define primary star statistics:
m1 = np.random.normal(m_star,m_star_err,10000) #draws stellar mass from gaussian dist around model mass (from GSC discovery paper)
dist = np.random.normal(d_star,d_star_err,10000)

# Fixing and initial semi-major axis:
a_au=100.0
a_au=np.linspace(a_au,a_au,10000)
T = np.sqrt((np.absolute(a_au)**3)/np.absolute(m1))
a = a_au/dist #semimajor axis in arcsec

# Fixing an initial Longitude of ascending node in radians:
O = np.radians(0.0)  
O=[O]*10000

# Randomly generated parameters:
	
#to = Time of periastron passage in years:
const = random.uniform(0.0,1.0,10000)
#^ Constant that represents the ratio between (reference epoch minus to) over period.  Because we are scaling
#semi-major axis, period will also scale, and epoch of periastron passage will change as a result.  This ratio
#will remain constant however, so we can use it scale both T and to appropriately.
to = d[ref]-(const*T)

#Eccentricity:
e = random.uniform(0.0,1.0,10000)
	
#Inclination in radians:
cosi = random.uniform(-1.0,1.0,10000)  #Draws sin(i) from a uniform distribution.  Inclination
#is computed as the arccos of sin(i):
i = np.arccos(cosi)
#Argument of periastron in degrees:
w = random.uniform(0.0,360.0,10000)
w = np.radians(w) #convert to radians for calculations
	

################# Determine PA/Sep at obs times ###########################
n = (2*np.pi)/T
r_model = []
true_anom = []
for date in d:
    M = n*(date-to)
    nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    E = np.array(nextE)
    r1 = a*(1.-e*cos(E))
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    r_model.append(r1)
    true_anom.append(f)

r_model = np.array(r_model)
#r_model = np.transpose(r_model)
true_anom = np.array(true_anom)
#true_anom = np.transpose(true_anom)

X1=r_model*((cos(O)*cos(w+true_anom))-(sin(O)*sin(w+true_anom)*cos(i)))
Y1=r_model*((sin(O)*cos(w+true_anom))+(cos(O)*sin(w+true_anom)*cos(i)))
# ^ Using constants to project orbit onto plane of the sky, from Seager textbook.  Y=RA, X=Dec
#This method arrives at the exact same answer as the Thiele-Innes constants (Lucy 2014)

########################## Scale a ########################
r_model_proj = np.sqrt((X1**2)+(Y1**2))
r_rand = np.random.normal(r_obs[ref],rerr[ref]) #This generates a gaussian random to 
#scale to that takes observational uncertainty into account.  Using the middle obs epoch.
a2 = a*(r_rand/r_model_proj[ref])  #<- scaling the semi-major axis

#New period:
a2_au=a2*dist #convert to AU for period calc:
T2 = np.sqrt((np.absolute(a2_au)**3)/np.absolute(m1))
#New epoch of periastron passage
to2 = d[ref]-(const*T2)

#New model data points:
n2 = (2.*np.pi)/T2
r_model2 = []
true_anom2 = []
for date in d:
    M = n2*(date-to2)
    nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
    E = np.array(nextE)
    r1 = a2*(1.-e*cos(E))
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    r_model2.append(r1)
    true_anom2.append(f)
r_model2 = np.array(r_model2)
true_anom2 = np.array(true_anom2)

#Recompute projection:

X2=r_model2*((cos(O)*cos(w+true_anom2))-(sin(O)*sin(w+true_anom2)*cos(i)))
Y2=r_model2*((sin(O)*cos(w+true_anom2))+(cos(O)*sin(w+true_anom2)*cos(i)))
	
################# Calculate new rotation ###########################
PA_model_proj = np.arctan2(X2,-Y2)
PA_model_proj2 = (np.degrees(PA_model_proj)+270)%360 #corrects for difference in zero-point
#between arctan function and ra/dec projection
PA_rand = np.random.normal(PA_obs[ref],terr[ref]) #Generates a random PA within 1
#sigma of observation
#New omega value:
O2=[]
for PA in PA_model_proj2[ref]:
    if PA < 0:
        O2.append((PA_rand-PA) + 360.)
    else:
        O2.append(PA_rand-PA)
# ^ This step corrects for the fact that the arctan gives the angle from the +x axis being zero,
#while for RA/Dec the zero angle is +y axis.  

#Recompute model with new rotation:
O2 = np.array(O2)
O2 = np.radians(O2)

X4=r_model2*((cos(O2)*cos(w+true_anom2))-(sin(O2)*sin(w+true_anom2)*cos(i)))
Y4=r_model2*((sin(O2)*cos(w+true_anom2))+(cos(O2)*sin(w+true_anom2)*cos(i)))

################## Determine Chi^2 between obs and pred  ###############
PA_model2 = np.arctan2(X4,-Y4)
PA_model = (np.degrees(PA_model2)+270.) % 360
PA_model=np.transpose(PA_model)  #<- This is necessary due to the 10000 x 4 array shape
	
r_model = np.sqrt((X4**2)+(Y4**2))
r_model=np.transpose(r_model)
	
chi=[]
for r,PA in zip(r_model,PA_model): 
	chi2 = ((r_obs-r)/(rerr))**2 + ((PA_obs-PA)/(terr))**2
	chi.append(np.sum(chi2))
# ^ Computing the chi-squared for each observation from the model and summing them

chi=np.array(chi)  #converts the number to a numpy array for further operations, doesn't change the value
chi_min = min(chi)  #<--- Initial minimum chi square value to subtract from all future trials

if rank ==0:
    print "Found initial chi-min: ", chi_min
#kk = open(directory+'/'+name+'_chimin_tracking', 'a') #Write out the initial chi-min
#kk.write(str(chi_min)+ "\n")
#kk.close()
delta_chi = -(chi-chi_min)/2.0
A = np.exp(delta_chi)  #Probability computation
rand = random.uniform(0.0,1.0,10000)  #The random "dice roll" to determine acceptable probability
accepted = np.where(A > rand) #This creates a list of indicies where the probability was greater than the dice roll

parameters = np.array([a2,T2,to2,e,np.degrees(i),np.degrees(w),np.degrees(O2),m1,dist,chi,A,rand])
parameters=np.transpose(parameters)  #Create an array of all accepted orbital parameters

#Writes accepted orbits to a file
k = open(directory+'/'+name+'_accepted_'+str(rank), 'a')
for params in parameters[accepted]:
	string = '   '.join([str(p) for p in params])
	k.write(string + "\n")
k.close()

##################################################################################
############################ Looping trials ######################################
start=tm.time()

from datetime import date, datetime
if rank==0:
    z = open(directory+'/'+name+'_log', 'a')
    string = str(date.today())
    string += '  Started run at  '
    string += str(datetime.now())
    z.write(string + "\n")
    z.close()

acc=0
num = 0
count = 0
loop_count=1
while num <= accept_min:  
    ######################### Generate priors  #############################
    m1 = np.random.normal(m_star,m_star_err,10000) 
    dist = np.random.normal(d_star,d_star_err,10000)
    a_au=100.0
    a_au=np.linspace(a_au,a_au,10000)
    T = np.sqrt((np.absolute(a_au)**3)/np.absolute(m1))
    a = a_au/dist 
    O = np.radians(0.0)  
    O=[O]*10000
    const = random.uniform(0.0,1.0,10000)
    to = d[ref]-(const*T)
    e = random.uniform(0.0,1.0,10000)
    cosi = random.uniform(-1.0,1.0,10000) 
    i = np.arccos(cosi)
    w = random.uniform(0.0,360.0,10000)
    w = np.radians(w) 
    

	################# Determine PA/Sep at obs times ###########################
    n = (2*np.pi)/T
    r_model = []
    true_anom = []
    for date in d:
        M = n*(date-to)
        nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
        r1 = a*(1.-e*cos(E))
        f1 = sqrt(1.+e)*sin(E/2.)
        f2 = sqrt(1.-e)*cos(E/2.)
        f = 2.*np.arctan2(f1,f2)
        r_model.append(r1)
        true_anom.append(f)
    r_model = np.array(r_model)
    true_anom = np.array(true_anom)

    X1=r_model*((cos(O)*cos(w+true_anom))-(sin(O)*sin(w+true_anom)*cos(i)))
    Y1=r_model*((sin(O)*cos(w+true_anom))+(cos(O)*sin(w+true_anom)*cos(i)))

    ########################## Scale a ########################
    r_model_proj = np.sqrt((X1**2)+(Y1**2))
    r_rand = np.random.normal(r_obs[ref],rerr[ref])
    a2 = a*(r_rand/r_model_proj[ref])
    a2_au=a2*dist 
    T2 = np.sqrt((np.absolute(a2_au)**3)/np.absolute(m1))
    to2 = d[ref]-(const*T2)
    #New model data points:
    n2 = (2.*np.pi)/T2
    r_model2 = []
    true_anom2 = []
    for date in d:
        M = n2*(date-to2)
        nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
        r1 = a2*(1.-e*cos(E))
        f1 = sqrt(1.+e)*sin(E/2.)
        f2 = sqrt(1.-e)*cos(E/2.)
        f = 2.*np.arctan2(f1,f2)
        r_model2.append(r1)
        true_anom2.append(f)
    r_model2 = np.array(r_model2)
    true_anom2 = np.array(true_anom2)
    X2=r_model2*((cos(O)*cos(w+true_anom2))-(sin(O)*sin(w+true_anom2)*cos(i)))
    Y2=r_model2*((sin(O)*cos(w+true_anom2))+(cos(O)*sin(w+true_anom2)*cos(i)))
	################# Calculate new rotation ###########################
    PA_model_proj = np.arctan2(X2,-Y2)
    PA_model_proj2 = (np.degrees(PA_model_proj)+270)%360
    PA_rand = np.random.normal(PA_obs[ref],terr[ref])

    O2=[]
    for PA in PA_model_proj2[ref]:
        if PA < 0:
            O2.append((PA_rand-PA) + 360.)
        else:
            O2.append(PA_rand-PA)
    O2 = np.array(O2)
    O2=np.radians(O2)
    X4=r_model2*((cos(O2)*cos(w+true_anom2))-(sin(O2)*sin(w+true_anom2)*cos(i)))
    Y4=r_model2*((sin(O2)*cos(w+true_anom2))+(cos(O2)*sin(w+true_anom2)*cos(i)))

    ################## Determine Chi^2 between obs and pred  ###############
    PA_model2 = np.arctan2(X4,-Y4)
    PA_model = (np.degrees(PA_model2)+270.) % 360
    PA_model = np.transpose(PA_model) 
    r_model = np.sqrt((X4**2)+(Y4**2))
    r_model=np.transpose(r_model)
    
    chi=[]
    for r,PA in zip(r_model,PA_model):
        chi2 = ((r_obs-r)/(rerr))**2 + ((PA_obs-PA)/(terr))**2
        chi.append(np.sum(chi2))

    chi=np.array(chi)
    delta_chi = -(chi-chi_min)/2.0
    A = np.exp(delta_chi)  
    rand = random.uniform(0.0,1.0,10000)
    accepted = np.where(A > rand)
    parameters = np.array([a2,T2,to2,e,np.degrees(i),np.degrees(w),np.degrees(O2),m1,dist,chi,A,rand])
    parameters=np.transpose(parameters)
    #Writes accepted orbits to a file
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
		print 'Found new chi min: ',chi_min
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
		############## but only if it has accepted at least one orbit ########################
		dat = np.loadtxt(open(directory+'/'+name+"_accepted_"+str(rank),"rb"),delimiter='   ',ndmin=2)
		a,T,to,e,i,w,O,m1,dist = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8]
		c,A,dice = dat[:,9],dat[:,10],dat[:,11]
		q = open(directory+'/'+name+'_accepted_'+str(rank), 'w')
		for a1,T1,to1,e1,i1,w1,O1,m11,dist1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,m1,dist,c,A,dice):
			delta_chi1 = -(c1-chi_min)/2.0
			AA = np.exp(delta_chi1)
			if AA > dice1:
				string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,m11,dist1,c1,AA,dice1]])
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
	
    loop_count = loop_count + 1  #Iterate the counter
    found_new_chi_min = 'no' #reset the re-evaluator for the next loop
    
	#################################### End loop ###########################################
comm.barrier()
if rank == 0:
    print 'Found enough orbits, finishing up...'
    for i in range(size):
        # Collect all the outputs into one file:
        dat = np.loadtxt(open(directory+'/'+name+"_accepted_"+str(i),"rb"),delimiter='   ',ndmin=2)
        a,T,to,e,i,w,O,m1,dist = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8]
        c,A,dice = dat[:,9],dat[:,10],dat[:,11]
        q = open(directory+'/'+name+'_accepted', 'a')
        for a1,T1,to1,e1,i1,w1,O1,m11,dist1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,m1,dist,c,A,dice):
            delta_chi1 = -(c1-chi_min)/2.0
            AA = np.exp(delta_chi1)
            if AA > dice1:
                string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,m11,dist1,c1,AA,dice1]])
                q.write(string + "\n")
            else:
                pass
        q.close()
    # Reperform the acceptance step one last time with the lowest chi-min off all processes:
    dat = np.loadtxt(open(directory+'/'+name+"_accepted","rb"),delimiter='   ',ndmin=2)
    a,T,to,e,i,w,O,m1,dist = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8]
    c,A,dice = dat[:,9],dat[:,10],dat[:,11]
    chi_min = np.min(c)
    print 'Minimum chi^2 found: ',chi_min
    q = open(directory+'/'+name+'_accepted', 'w')
    for a1,T1,to1,e1,i1,w1,O1,m11,dist1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,m1,dist,c,A,dice):
		delta_chi1 = -(c1-chi_min)/2.0
		AA = np.exp(delta_chi1)
		if AA > dice1:
			string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,m11,dist1,c1,AA,dice1]])
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
	

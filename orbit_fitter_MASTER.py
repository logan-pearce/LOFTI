import numpy as np
from numpy import tan, arctan, sqrt, cos, sin, arccos
from numpy import random
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time as tm
import os
from datetime import date

#Newton-Raphson solver for solving for the eccentricity anomaly (E)
#from https://stackoverflow.com/questions/20659456/python-implementing-a-numerical-equation-solver-newton-raphson:
def derivative(f, x, h):
      return (f(x+h) - f(x-h)) / (2.0*h)

def solve(f, E0, e, h):
    from numpy import cos
    lastE = E0
    nextE = lastE + 10* h  # "different than lastX so loop starts OK
    number=0
    while (abs(lastE - nextE) > h) and number < 1001:  # this is how you terminate the loop - note use of abs()
        newY = f(nextE,e,E0) # just for debug... see what happens
        #print "f(", nextE, ") = ", newY     # print out progress... again just debug
        lastE = nextE
        nextE = lastE - newY / (1.-e*cos(lastE))  # update estimate using N-R
        number=number+1
        if number >= 1000:
            nextE = float('NaN')
    return nextE

#Eccentricity anomaly equation for the numerical solver
def eccentricity_anomaly(E,e,M):
    return E - (e*sin(E)) - M
############################ Observations ###############################
r_obs =np.array([2.2020489,2.20220384,2.2029651,2.1973678,2.1926688,2.1931015])#In arcsec
rerr = np.array([0.0002579,0.0007237,0.0004729,0.0001299,0.0006768,0.00023159])
rerr[0],rerr[1],rerr[2],rerr[3] = np.sqrt(rerr[0]**2+0.001**2),np.sqrt(rerr[1]**2+0.001**2),\
    np.sqrt(rerr[2]**2+0.001**2),np.sqrt(rerr[3]**2+0.001**2) #Adding 1mas in error quadrature to scatter error for 2008-2014 epochs

PA_obs = np.array([175.6196,175.5894,175.5138,175.4416,175.41233,175.3811]) #In deg
terr = np.array([0.0036,0.0067,0.0072,0.0095,0.0091,0.0071])

d = np.array([2008.46,2009.41,2011.42,2014.58,2016.46,2017.49])

#Convert to RA/Dec:
PA_obs_rad = np.radians(PA_obs)
dec = r_obs*np.cos(PA_obs_rad)
ra = r_obs*np.sin(PA_obs_rad)

ref = (r_obs.shape[0])/2 #compute the index of the middle observation epoch as the reference epoch for all
#future calcs - the one we want to scale to to minimize chi-squared.  For even numbers of observations,
#it selects the later of the two middles

######################################################################################################
####################### Initial 10000 orbit loop to get initial chi_min: #############################

######################### Generate priors  #############################
#### Parameters #####
#Primary mass in solar masses:
#m1=0.6
m1 = np.random.normal(0.6,0.1,10000) #draws stellar mass from gaussian dist around model mass (from GSC discovery paper)
	
dist = np.random.normal(145,14.0,10000)

a_au=100.0
a_au=np.linspace(a_au,a_au,10000)
T = np.sqrt((np.absolute(a_au)**3)/np.absolute(m1))
a = a_au/dist #semimajor axis in arcsec
	
#to = Time of periastron passage in years:
c = random.uniform(0.0,1.0,10000)
#^ Constant that represents the ratio between (reference epoch minus to) over period.  Because we are scaling
#semi-major axis, period will also scale, and epoch of periastron passage will change as a result.  This ratio
#will remain constant however, so we can use it scale both T and to appropriately.
to = d[ref]-(c*T)

#Eccentricity:
e = random.uniform(0.0,1.0,10000)
	
#Inclination in radians:
cosi = random.uniform(-1.0,1.0,10000)  #Draws sin(i) from a uniform distribution.  Inclination
#is computed as the arccos of sin(i):
i = np.arccos(cosi)
#Argument of periastron in degrees:
w = random.uniform(0.0,360.0,10000)
w = np.radians(w) #convert to radians for calculations
	
#Longitude of ascending node in radians:
O = np.radians(0.0)  #start with a fixed value of Omega of 0.  Will be rotated later.
O=[O]*10000
	
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
to2 = d[ref]-(c*T2)

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
O2=np.radians(O2)

X4=r_model2*((cos(O2)*cos(w+true_anom2))-(sin(O2)*sin(w+true_anom2)*cos(i)))
Y4=r_model2*((sin(O2)*cos(w+true_anom2))+(cos(O2)*sin(w+true_anom2)*cos(i)))

################## Determine Chi^2 between obs and pred  ###############
PA_model2 = np.arctan2(X4,-Y4)
PA_model = np.degrees(PA_model2)+270.
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
print "Found initial chi-min: ", chi_min
kk = open('GSC6214_chimin_tracking', 'a') #Write out the initial chi-min
kk.write(str(chi_min)+ "\n")
kk.close()
delta_chi = -(chi-chi_min)/2.0
A = np.exp(delta_chi)  #Probability computation
rand = random.uniform(0.0,1.0,10000)  #The random "dice roll" to determine acceptable probability
accepted = np.where(A > rand) #This creates a list of indicies where the probability was greater than the dice roll

parameters = np.array([a2,T2,to2,e,np.degrees(i),np.degrees(w),np.degrees(O2),m1,dist,chi,A,rand])
parameters=np.transpose(parameters)  #Create an array of all accepted orbital parameters

#Writes accepted orbits to a file
k = open('GSC6214_accepted', 'a')
for params in parameters[accepted]:
	string = '   '.join([str(p) for p in params])
	k.write(string + "\n")
k.close()

#Writes accepted orbits to a file
g = open('GSC6214_accepted_initial', 'a')
for params in parameters[accepted]:
	string = '   '.join([str(p) for p in params])
	g.write(string + "\n")
g.close()

##################################################################################
############################ Looping trials ######################################
start=tm.time()

from datetime import date, datetime

z = open('GSC6214_log', 'a')
string = str(date.today())
string += '  Started run at  '
string += str(datetime.now())
z.write(string + "\n")
z.close()

acc=0
num = 0
count = 0
loop_count=1
while num <= 1000:  #<- Change this number to be the total number of accepted orbits desired
    ######################### Generate priors  #############################
    #### Parameters #####
    #Primary mass in solar masses:
    m1 = np.random.normal(0.6,0.1,10000)
    dist = np.random.normal(145,14.0,10000)
    a_au=100.0
    a_au=np.linspace(a_au,a_au,10000)
    T = np.sqrt((np.absolute(a_au)**3)/np.absolute(m1))
    a = a_au/dist
    c = random.uniform(0.0,1.0,10000)
    to = d[ref]-(c*T)
    e = random.uniform(0.0,1.0,10000)
    cosi = random.uniform(-1.0,1.0,10000)
    i = np.arccos(cosi)
    w = random.uniform(0.0,360.0,10000)
    w = np.radians(w)
    O = np.radians(0.0)
    O=[O]*10000
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
    to2 = d[ref]-(c*T2)
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
    PA_model = np.degrees(PA_model2)+270.
    PA_model=np.transpose(PA_model) 
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
    k = open('GSC6214_accepted', 'a')
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
		kk = open('GSC6214_chimin_tracking', 'a')
		kk.write(str(chi_min)+ "\n")
		kk.close()
    else:
        found_new_chi_min = 'no'
	#if this minimum chi is less than the previously assigned chi_min, update the chi_min variable
	#to the new value, and write it out to this file. 
	
    if found_new_chi_min == 'yes' and num!=0: 
		############## Recalculate old accepted orbits with new chi-min for acceptance #######
		############## but only if it has accepted at least one orbit ########################
		dat = np.loadtxt(open("GSC6214_accepted","rb"),delimiter='   ',ndmin=2)
		a,T,to,e,i,w,O,m1,dist = dat[:,0],dat[:,1],dat[:,2],dat[:,3],dat[:,4],dat[:,5],dat[:,6],dat[:,7],dat[:,8]
		c,A,dice = dat[:,9],dat[:,10],dat[:,11]
		q = open('GSC6214_accepted', 'w')
		for a1,T1,to1,e1,i1,w1,O1,m11,dist1,c1,A1,dice1 in zip(a,T,to,e,i,w,O,m1,dist,c,A,dice):
			delta_chi1 = -(c1-chi_min)/2.0
			AA = np.exp(delta_chi1)
			if AA > dice1:
				string = '   '.join([str(p) for p in [a1,T1,to1,e1,i1,w1,O1,m11,dist1,c1,AA,dice1]])
				q.write(string + "\n")
			else:
				pass
		q.close()
	
		dat2 = np.loadtxt(open("GSC6214_accepted","rb"),delimiter='   ',ndmin=2)
		num=dat2.shape[0]
    else:
		pass
	
	#This step is only for counting the total number of accepted orbits.  There's probably a better way but you know.
    mod2 = loop_count%10
    if mod2 == 0:
		dat2 = np.loadtxt(open("GSC6214_accepted","rb"),delimiter='   ',ndmin=2)
		num=dat2.shape[0]
		print 'Loop count:',loop_count
		print "############## Found ",num,"accepted orbits #################"	
	
    loop_count = loop_count + 1  #Iterate the counter
    found_new_chi_min = 'no' #reset the re-evaluator for the next loop
    
	#################################### End loop ###########################################


		
	
print '.........done'
stop=tm.time()
time=stop-start
from datetime import date

print 'This operation took',time,'seconds'
print 'and',time/3600.,'hours'

#Write out to a log file:
from datetime import date
stop=tm.time()
time=stop-start
z = open('GSC6214_log', 'a')
string = str(date.today())
string += ' took '
string += str(time/3600.)
string += ' hours to get '
string += str(num)
string += ' accepted orbits; '
string += ' Tested '
string += str(loop_count*10000.)
string += ' total permutations '
z.write(string + "\n")
z.close()
	
	

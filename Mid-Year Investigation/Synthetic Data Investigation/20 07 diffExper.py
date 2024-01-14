import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
import pandas as pd
from scipy.optimize import least_squares
import scipy.stats as stats
import glob

#just wrote - dunno if it works?
def nearestNeigh(x, t, xArr, tArr, data):
    """
        x = position of interest
        t = time of interest
        xArr = array of evaluated positions
        tArr = array of evaluated times
        data = matrix of temperatures dataframe type
    """
    xi = np.abs(xArr-x).argmin() #x is list of x values 
    yi = np.abs(tArr-t).argmin() #y is list of y values
    return data.loc[yi,xi] #data table!

def cutDF(xObs,tObs, xData, tData, TdataDF):
    """
    Input:
        xObs = positions of interest
        tObs = times of interest
        xData = array of evaluated positions
        tData = array of evaluated times
        TdataDF = matrix of temperatures dataframe type
    Output:
        TdataCutDF = cut down DF based off the observational data df
    """
    xKeep = []
    tKeep =[]
    for xi in xObs:
        xKeep.append(xData[np.abs(xi-xData).argmin()])
    TdataCutxDF=TdataDF.filter(items=xKeep, axis=0)
    for ti in tObs:
        tKeep.append(tData[np.abs(ti-tData).argmin()])
    TdataCutDF=TdataCutxDF.filter(items=tKeep, axis=1)
    return TdataCutDF #data table!


def manyNeigh(xApoi,tApoi, xArr, tArr, data):
    """
        returns an array of points of interest from arrays of points of interest
        xApoi = x array of POSITION interest points
        t = t array of TIME interest points
        xArr = array of evaluated positions
        tArr = array of evaluated times
        data = matrix of temperatures dataframe type

        output:
        TArr = array of returned temps NEED THIS TO BE A DATAFRAME NOT AN ARRAY

    """
    TArr=[]
    for x,t in xApoi,tApoi:
        TArr.append(nearestNeigh(x, t, xArr, tArr, data))
    return TArr

#from column function
def makeColSol(x,t,T):
    x=x[1]
    t=t[:,1]
    colSol=[0.0,0.0,0.0]
    for i in range(len(t)):
        for j in range(len(x)):
            colSol = np.vstack((colSol,[x[j],t[i],T[i,j]]))
    #colSol = np.delete(colSol, (0), axis=0)
    return np.delete(colSol, (0), axis=0)

class Boundary(object):
    """
    Class that stores useful information on boundary or initial conditions for a PDE mesh.

    Attributes:
        condition (str): Type of boundary condition e.g. dirichlet, neumann, robin.
        function (function): Function used to calculate the boundary value at a specific point.
    """
    def __init__(self, condition, function):

        # set the type of boundary condition
        self.condition = condition

        # set the boundary function/equation
        self.function = function



class MeshXT(object):
    """
    Class that stores useful information about the current XT mesh and associated boundary conditions.

    Includes methods for plotting the PDE solution associated with this mesh and calculating the error relative
    to the analytic solution, if it is known.

    Attributes:
        nx (int): Number of mesh points along the x dimension.
        nt (int): Number of mesh points along the t dimension.
        x (numpy array): Mesh coordinates along the x dimension.
        t (numpy array): Mesh coordinates along the t dimension.
        dx (float): Mesh spacing along the x dimension.
        dt (float): Mesh spacing along the t dimension.
        boundaries (list): List of objects of Boundary class. Order is u(x0,t), u(x1,t), u(x,t0), du(x,t0)/dt, etc...
        solution (numpy array): PDE solution as calculated on the mesh.
        method (string): Name of the solution method e.g. explicit, crank-nicolson, galerkin, backward-euler.
        mae (float): mean absolute error of numerical/mesh solution relative to analytic/exact solution.

    Arguments:
        :param x: Lower and upper limits in x dimension.
        :param t: Lower and upper limits in t dimension.
        :param delta: Mesh spacing in x and t dimensions.
        :param boundaries: List of objects of the Boundary class, with same order as class attribute of same name.
        :type x: Numpy array with two elements.
        :type t: Numpy array with two elements.
        :type delta: Numpy array with two elements.
        :type boundaries: List.
    """

    def __init__(self, x, t, delta, boundaries):

        # define the integer number of mesh points, including boundaries, based on desired mesh spacing
        self.nx = math.floor((x[1] - x[0]) / delta[0]) + 1
        self.nt = math.floor((t[1] - t[0]) / delta[1]) + 1

        # calculate the x and y values/coordinates of mesh as one-dimensional numpy arrays
        self.x = np.linspace(x[0], x[1], self.nx)
        self.t = np.linspace(t[0], t[1], self.nt)

        # calculate the actual mesh spacing in x and y, should be similar or same as the dx and dy arguments
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

        # store the boundary and initial condition as a list of Boundary class objects
        self.boundaries = boundaries

        # initialise method name - useful for plot title. Updated by solver.
        self.method = None

        # initialise mean absolute error to be None type.
        self.mae = None

        # initialise the full PDE solution matrix
        self.solution = np.zeros((self.nx, self.nt))

        # apply dirichlet boundary conditions directly to the mesh solution
        if self.boundaries[0].condition == 'dirichlet':
            self.solution[0, :] = self.boundaries[0].function(self.x[0], self.t)

        elif self.boundaries[0].condition == 'neumann':
            self.solution= np.vstack([np.ones([1,self.nt]), self.solution])#self.boundaries[0].function(self.x[0], self.t)
            self.x = np.append(self.x,self.x[-1]+self.dx)

        if self.boundaries[1].condition == 'dirichlet':
            self.solution[-1, :] = self.boundaries[1].function(self.x[-1], self.t)

        elif self.boundaries[1].condition == 'neumann':
            self.solution= np.vstack([self.solution,np.zeros([1,self.nt])])#self.boundaries[0].function(self.x[0], self.t)
            self.x = np.append(self.x,self.x[-1]+self.dx)


        # apply initial conditions directly to the mesh solution
        self.solution[:, 0] = self.boundaries[2].function(self.x, self.t[0])

    def get_XYZ(self): #also changed this from fdValidation.py
        [X,Y] = np.meshgrid(self.x[:-1],self.t) 
        Z = np.array(list(map(list, self.solution.transpose())))
        col=makeColSol(X,Y,Z)
        return col

    def plot_solution(self, ntimes, save_to_file=False, save_file_name='figure.png'):
        """
        Plot the mesh solution u(x,t^n) at a fixed number of time steps.

        Arguments:
            :param ntimes: number of times at which to plot the solution, u(x,t_n).
            :param save_to_file: If True, save the plot to a file with pre-determined name.
            :param save_file_name: Name of figure to save.
            :type ntimes: Integer.
            :type save_to_file: Boolean.
            :type save_file_name: String.
        """
        #plotting commands
        timeSpaceTemp = np.linspace(0,self.nt-1,ntimes)
        timeSpace=[]
        for i in timeSpaceTemp:
            timeSpace.append(int(i))
        plt.figure()
        for j in timeSpace:
            plt.plot(self.x, self.solution.transpose()[j])
        #plt.text(0,1,'mae: '+str(self.mae))
        plt.xlabel("x m")
        plt.ylabel("Temp C")
        plt.legend(self.t[timeSpace], title = "Time s")
        plt.title("Explicit Numerical Solution $\Delta$t = 0.001")
        plt.show()
        
        #3d plot
        plt.figure()
        #timeSpaceTemp2 = np.linspace(0,self.nt-1,self.nt)

        #[X,Y,Z] = get_XYZ(self)
        #Z = np.array(list(map(list, self.solution.transpose())))
        timeSpaceTemp2 = np.linspace(0,self.nt-1,self.nt)
        [X,Y] = np.meshgrid(self.x,timeSpaceTemp2)
        Z = np.array(list(map(list, self.solution.transpose())))
        #col=makeColSol(X,Y,Z)

        ax = plt.axes(projection='3d')
        ax.plot_surface(X=X, Y=Y, Z=Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.xlabel("x m")
        plt.ylabel("Time ms")
        plt.title("Explicit Solution")
        #plt.zlabel("Temp C")
        plt.show()

#not currently being used
    def mean_absolute_error(self, exact):
        """
        Calculates the mean absolute error in the solution, relative to exact solution, for the final time step.

        Arguments:
            :param exact: The exact solution to the PDE.
            :type exact: Function.
        """
        #calculates mean absolute error by summing and using the absolute function then dividing by nx
        self.mae = (np.sum(np.absolute(self.solution[:,-1] - exact(self.x,self.t[-1]))))/self.nx   



class SolverHeatXT(object):
    """
    Class containing attributes and methods useful for solving the 1D heat equation.

    Attributes:
        method (string): Name of the solution method e.g. crank-nicolson, explicit, galerkin, backward-euler.
        r (float): ratio of mesh spacings, useful for evaluating stability of solution method.
        theta (float): Weighting factor used in implicit scheme e.g. Crank-Nicolson has theta=1/2.
        a (numpy array): Coefficient matrix in system of equations to solve for PDE solution.
        b (numpy array): Vector of constants in system of equations to solve for PDE solution at time t^n.

    Arguments:
        :param mesh: instance of the MeshXT class.
        :param alpha: reciprocal of thermal diffusivity parameter in the heat equation.
        :param method: Name of the solution method e.g. crank-nicolson, explicit, galerkin, backward-euler.
        :type mesh: Object.
        :type alpha: Float.
        :type method: String.
    """

    def __init__(self, mesh, alpha, method='crank-nicolson'):

        # set the solution method
        self.method = method
        mesh.method = method

        # set the ratio of mesh spacings used in solver equations
        self.r = mesh.dt / (alpha * mesh.dx * mesh.dx)
        #checks stability
        if self.r < 0 or self.r > 0.5:
            print("unstable, solution will not converge")
        else:
            print("solution looks stable! will converge!")
                

        # initialise theta variable used in the implicit methods
        self.theta = None

        # determine if solution method requires matrix equation and set A, b accordingly
        if self.method == 'explicit':
            self.a = None
            self.b = None
        else:
            self.a = np.zeros((mesh.nx-2, mesh.nx-2))
            self.b = np.zeros(mesh.nx-2)

    def solver(self, mesh):
        """
        Run the requested solution method. Default to Crank-Nicolson implicit method if user hasn't specified.

        Arguments:
            :param mesh: Instance of the MeshXT class.
        """

        # run the explicit solution method
        if self.method == 'explicit':
            self.explicit(mesh)


    def explicit(self, mesh):
        """
        Solve the 1D heat equation using an explicit scheme.

        Arguments:
            :param mesh: Instance of the MeshXT class.
        """
        #nested for loop. goes thru and accesses the shape around the point of interest.
        #applies the stencil and rearranges and solves.
        # then saves directly to the solution matrix (mesh.solution). Points that have just been calculated
        # are likely used in the next iteration.
        for i in range(1,mesh.nt):
            for j in range(1,mesh.nx): 
                mesh.solution[j,i] =self.r*(mesh.solution[j-1,i-1]+mesh.solution[j+1,i-1])+(1-2*self.r)*mesh.solution[j,i-1]
                if mesh.boundaries[1].condition == 'neumann' and j==mesh.nx-2:
                    mesh.solution[-1,:] =mesh.solution[-3,:]
        #deletes the imaginary row that we created earlier - note this would have to be dynamic for different uses
        #jsut a quick fix for now cause im working on other stuff
        #mesh.solution = np.delete(mesh.solution, -1, 0) 

#fucntion for the ting (objective)
#TObs should be passing x and t because where there are difference instances - we should be finding TData at those places
#from an interpolated function or something. I shouldnt we expecting exact values; thus probably not a good idea to pass mesh_explicit
#in this function - instead maybe pass the interpolated curve or something, unsure.
def func(param):
    
    [mesh_explicit,TObs]= stuff() #do i have t and x from TObs???
    # right now it uses the same observation data info - this is not realistic....
    solver_explicit = SolverHeatXT(mesh_explicit, alpha=param, method='explicit')

    # run the solver for each case
    solver_explicit.solver(mesh_explicit)

    #get x y z
    Tdata_df = pd.DataFrame(mesh_explicit.solution[:-1,:], index= np.linspace(0., 1., 21))

    #real test run
    # create an instance of the SolverHeatXT class for each solution method

    #get x y z
    #resd = TData-TObs
    rss = np.sum(np.square(TData-TObs))
    #rss=np.linalg.norm(TData-TObs)
    return rss

def func2(param):
    """
        TObs is a dataset now?eyes
    """
    [mesh_explicit,TObsDF]= stuff() #do i have t and x from TObs???
    # right now it uses the same observation data info - this is not realistic....
    solver_explicit = SolverHeatXT(mesh_explicit, alpha=param, method='explicit')

    # run the solver for each case
    solver_explicit.solver(mesh_explicit)

    #get x y z
    #TdataDF = pd.DataFrame(mesh_explicit.solution[:-1,:], index= np.linspace(0., 1., 21))
    TdataDF = pd.DataFrame(mesh_explicit.solution[:-1,:], index= mesh_explicit.x[:-1], columns =mesh_explicit.t)
    TdataCutDF = cutDF(TObsDF.index,TObsDF.columns, TdataDF.index, TdataDF.columns, TdataDF) #cutDF(xObs,tObs, xData, tData, TdataDF)
    
    #real test run
    # create an instance of the SolverHeatXT class for each solution method

    #get x y z
    #resd = TData-TObs
    #rss2 = np.sum(np.square(TObsDF-TdataCutDF.values))
    rss2 = np.sum(np.square(TObsDF.values-TdataCutDF.values))
    #rss=np.linalg.norm(TData-TObs)
    return rss2

#check if this works mans cause idkkkkk
def bootstrap(sample, trueMu):
    """
    does all the boring stats shit - need the sample set and the true parameter value
    Inputs:
        sample - a set of estimated parameters measured at the same location
        trueMu - just for all this artifical bullsh!t; we wont actually know this value but good for code/experiment vibes.
    """
    sample.sort()
    mu = sum(sample)/len(sample)
    sd =np.std(sample)
    plt.plot(sample, stats.norm.pdf(sample, mu, sd))
    #plot some vertical lines for the thermal diffusivity vals
    # x coordinates for the lines
    xcoords = [mu, trueMu]
    # colors for the lines
    colors = ['r','b']
    #labels
    labels = ['Estimated Param', 'True Param']
    for xc,c,l in zip(xcoords,colors,labels):
        plt.axvline(x=xc, label='{}'.format(l), c=c)
    plt.legend()
    plt.show()

#check if this works mans cause idkkkkk
def bootstrap2(sample, trueMu):
    """
    does all the boring stats shit - need the sample set and the true parameter value
    Inputs:
        sample - a set of estimated parameters measured at the same location
        trueMu - just for all this artifical bullsh!t; we wont actually know this value but good for code/experiment vibes.
    """
    sample.sort()
    mu = sum(sample)/len(sample)
    #sd =np.std(sample)
    plt.hist(sample, 15, density=True, facecolor='r')
    #plot some vertical lines for the thermal diffusivity vals
    # x coordinates for the lines
    xcoords = [mu, trueMu]
    # colors for the lines
    colors = ['r','b']
    #labels
    labels = ['Estimated Param', 'True Param']
    for xc,c,l in zip(xcoords,colors,labels):
        plt.axvline(x=xc, label='{}'.format(l), c=c)
    plt.legend()
    plt.show()
#check if this works mans cause idkkkkk
def bootstrapMulti(samples, trueMu):
    """
    does all the boring stats shit - need the sample set and the true parameter value
    Inputs:
        samples - a set of a set of estimated parameters measured at the same location
        trueMu - just for all this artifical bullsh!t; we wont actually know this value but good for code/experiment vibes.
    """
    mus=[trueMu]
    colours = ['k','b','g','c','m','y'] #manually change these for now
    for row,c in samples,colours[:-1]:
        row.sort()
        mu = sum(row)/len(row)
        mus.append(mu)
        #sd =np.std(row)
        plt.hist(row, 50, density=True, facecolor=c)

    #plot some vertical lines for the thermal diffusivity vals
    # x coordinates for the lines
    xcoords = mus
    # colors for the lines
    #labels
    labels = ['True Param','Estimated Param (start)','Estimated Param (middle)','Estimated Param (end)','Estimated Param (realistic)','Estimated Param (all points)'] #manually change these for now
    for xc,c,l in zip(xcoords,colors,labels):
        plt.axvline(x=xc, label='{}'.format(l), c=c)
    plt.legend()
    #plt.xlim(40, 160)
    #plt.ylim(0, 0.03)
    plt.show()
#check if this works mans cause idkkkkk
def bootstrapMulti2(samples, trueMu,t):
    """
    does all the boring stats shit - need the sample set and the true parameter value
    Inputs:
        samples - a set of a set of estimated parameters measured at the same location
        trueMu - just for all this artifical bullsh!t; we wont actually know this value but good for code/experiment vibes.
        t = type of data: binary, pessimistic (0) or optimistic (1)
    """
    colors = ['b','g','c','m','k']
    mus=[]
    sds=[]
    x=0
    for row in samples:
        row.sort()
        mu = sum(row)/len(row)
        mus.append(mu)
        sd =np.std(row)
        sds.append(sd)
        plt.hist(row, 15, density=True, facecolor=colors[x])
        x=x+1
    #plot some vertical lines for the thermal diffusivity vals
    # x coordinates for the lines
    mus.append(trueMu)
    # colors for the lines
     #manually change these for now
    #labels
    labels = ['Estimated Param (all points)','Estimated Param (end)','Estimated Param (middle)','Estimated Param (realistic)','True Param'] #manually change these for now
    for xc,c,l in zip(mus,colors,labels):
        plt.axvline(x=xc, label='{}'.format(l), c=c)
    plt.legend()
    if t==0:
        plt.title("Pessimistic Data")
    elif t ==1:
        plt.title("Optimistic Data")
    plt.show()
    return mus,sds


# define the boundary and initial condition functions
def left(x, t): return 30.0+0.*x+0.*t
#def right(x, t): return 0.*x+0.*t
def right_derivative(x, t): return 0.*x + 0.*t
def initial(x, t): return 0.*x+0.*t


# define the boundaries properties and create each BoundariesXY object
x0 = Boundary('dirichlet', left)
x1 = Boundary('neumann', right_derivative)
t0 = Boundary('initial', initial)

# create an instance of the MeshXY class for each solution method
mesh_explicit = MeshXT(x=[0., 1.0], t=[0., 0.1], delta=[0.05, 0.0001], boundaries=[x0, x1, t0]) #what units is this in??? does it matter
alpha=1.0/(math.pi**2)
solver_explicit = SolverHeatXT(mesh_explicit, alpha=alpha, method='explicit')

    # run the solver
solver_explicit.solver(mesh_explicit)

#dataset_df = pd.DataFrame(mesh_explicit.solution[:-1,:], index= np.linspace(0., 1., 21))
dataset_df = pd.DataFrame(mesh_explicit.solution[:-1,:], index= mesh_explicit.x[:-1], columns =mesh_explicit.t)
"""
real =[]
pos = [1.,3.,5.]
for el in pos:
    real.append(el*(1./6.))
mu, sigma = 0, 0.3
#real = np.array([0.15000000000000002,0.5,0.8500000000000001])
sample=[]
locations= [[0.5],[1.0],real,mesh_explicit.x[:-1]]
TObsCutDF= cutDF(locations[0],mesh_explicit.t[:], mesh_explicit.x[:],mesh_explicit.t[:],dataset_df) #finally works!

while len(sample)<51:
    noise = np.random.normal(mu, sigma, np.shape(TObsCutDF))
    TObsDF=TObsCutDF+noise

    param0 = 0.5
        #anon func - do i know how it works? no. does it work? eyes
    stuff = lambda: [mesh_explicit,TObsDF]
    res_1 = least_squares(func2, param0)
    print(res_1.x)
    sample.append(res_1.x[0])

np.savetxt('middle2.csv', sample, delimiter=',')

"""
"""
#save array to datafrome to pickle or something
dataset_df = pd.DataFrame(mesh_explicit.solution[:-1,:], index= np.linspace(0., 1., 21))

dataset_df.to_pickle("./MySol.pkl")
#to load unpickled_df = pd.read_pickle("./temp1.pkl")
"""
"""
real =[]
pos = [1.,3.,5.]
for el in pos:
    real.append(el*(1./6.))
#some artifically noisy data
mu, sigma = 0, 0.1

noise = np.random.normal(mu, sigma, np.shape(dataset_df))
TObsDF=dataset_df+noise #thisis what id have to adjust
TObsCutDF= TObsDF.filter(items=real, axis=0)
param0 = 0.5

#anon func - do i know how it works? no. does it work? eyes
stuff = lambda: [mesh_explicit,TObsCutDF]
res_1 = least_squares(func2, param0)
print(res_1.x)
print(res_1.cost)
print(res_1.optimality)


param0 = 0.5
#maybe change this to a function - maybe just change all these to function, why is everything so messy
sample=[]
while len(sample)<51:
    noise = np.random.normal(mu, sigma, np.shape(dataset_df))
    TObsDF=dataset_df+noise #thisis what id have to adjust
    #  creating a noise with the same dimension as the dataset (2,2) 
    #noise = np.random.normal(mu, sigma, len(T))
    #test column array
    #TObs=T+noise #thisis what id have to adjust
    #anon func - do i know how it works? no. does it work? eyes
    stuff = lambda: [mesh_explicit,TObsDF]
    res_1 = least_squares(func2, param0)
    sample.append(res_1.x[0])

sample = np.random.normal(alpha, 0.5, 50)

#bootstrap(sample, alpha)
bootstrapMulti2(sample, alpha)
"""
#working in terms of sequence

#alpha=1.0/(math.pi**2)

"""
real =[]
pos = [1.,3.,5.]
for el in pos:
    real.append(el*(1./6.))
mu, sigma = 0, 0.1 
samples=[]
locations= [[0.0],[0.5],[1.0],real,mesh_explicit.x[:-1]]
for x in locations:
    sample=[]
    while len(sample)<6:
        noise = np.random.normal(mu, sigma, np.shape(dataset_df))
        TObsDF=dataset_df+noise #thisis what id have to adjust
        TObsCutDF= TObsDF.filter(items=x, axis=0)
        param0 = 0.5
        #anon func - do i know how it works? no. does it work? eyes
        stuff = lambda: [mesh_explicit,TObsCutDF]
        res_1 = least_squares(func2, param0)
        print(res_1.x)
        sample.append(res_1.x[0])
    samples.append(sample)

bootstrapMulti2(samples, alpha)


#multiprocessing attempt 1
def experiments(x, mu, sigma, dataset_df):
    sample = []
    while len(sample)<6:
        noise = np.random.normal(mu, sigma, np.shape(dataset_df))
        TObsDF=dataset_df+noise #thisis what id have to adjust
        TObsCutDF= TObsDF.filter(items=x, axis=0)
        param0 = 0.5
       #anon func - do i know how it works? no. does it work? eyes
        stuff = lambda: [mesh_explicit,TObsCutDF]
        res_1 = least_squares(func2, param0)
        print(res_1.x)
        sample.append(res_1.x[0])
    return sample

if __name__ == '__main__':
    with Pool(5) as p:
        samples=[]
        real =[]
        pos = [1.,3.,5.]
        for el in pos:
            real.append(el*(1./6.))
        locations= [[0.0],[0.5],[1.0],real,mesh_explicit.x[:-1]]
        mu, sigma = 0, 0.1 
        samples = p.map(experiments, locations, mu, sigma, dataset_df)
    print(alpha)
"""

"""
    DAta analysis tings

samplesBest=[]
samplesWorst = []
filenames = glob.glob('*.csv')
for data in filenames:
    dataName =data.replace(".csv","")
    globals()[dataName] = np.loadtxt(fname=data, delimiter=',')
    if 'Worst' in data:
        samplesWorst.append(globals()[dataName])
    else:
        samplesBest.append(globals()[dataName])

musWorst,sdsWorst = bootstrapMulti2(samplesWorst,alpha,t=0)
musBest,sdsBest = bootstrapMulti2(samplesBest,alpha,t=1)
FBest, pBest = stats.f_oneway(all2,end,middle2,real)
FWorst, pWorst = stats.f_oneway(allWorst,endWorst,middleWorst,realWorst)
print(FBest, pBest)
print(FWorst, pWorst)
print(musWorst)
print(sdsWorst)
print(musBest)
print(sdsBest)
"""
print(alpha)
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d
from ColumnFunction import makeColSol
import pandas as pd


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

    def get_XYZ(self):
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
        """
        timeSpaceTemp = np.linspace(0,self.nt-1,ntimes)
        timeSpace=[]
        for i in timeSpaceTemp:
            timeSpace.append(int(i))
        plt.figure()
        for j in timeSpace:
            plt.plot(self.x[:-1], self.solution.transpose()[j])
        #plt.text(0,1,'mae: '+str(self.mae))
        plt.xlabel("x m")
        plt.ylabel("Temp C")
        plt.legend(self.t[timeSpace], title = "Time s")
        plt.title("Explicit Numerical Solution $\Delta$t = 0.001")
        plt.show()
        """
        #3d plot
        plt.figure()
        #timeSpaceTemp2 = np.linspace(0,self.nt-1,self.nt)

        #[X,Y,Z] = get_XYZ(self)
        #Z = np.array(list(map(list, self.solution.transpose())))
        #timeSpaceTemp2 = np.linspace(0,self.nt-1,self.nt)
        [X,Y] = np.meshgrid(self.x[:-1],self.t)
        Z = np.array(list(map(list, self.solution.transpose())))
        #col=makeColSol(X,Y,Z)
        plt.contourf(X, Y, Z, 20, cmap='RdGy')

        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=15)
        plt.xlabel("x (m)", fontsize=20)
        plt.ylabel("Time (s)", fontsize=20)
        plt.tick_params('both',labelsize=15)
        plt.title("FD Explicit Solution",fontsize = 20)
        #plt.zlabel("Temp C")
        plt.show()
    


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
        mesh.solution = np.delete(mesh.solution, -1, 0) 


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
mesh_explicit = MeshXT(x=[0., 0.5], t=[0., 0.1], delta=[0.02, 0.00001], boundaries=[x0, x1, t0])

# create an instance of the SolverHeatXT class for each solution method
solver_explicit = SolverHeatXT(mesh_explicit, alpha=1.0/(math.pi**2), method='explicit')

# run the solver for each case
solver_explicit.solver(mesh_explicit)

mesh_explicit.plot_solution(ntimes=5, save_to_file=False, save_file_name='explicit.png')
#get x y z

#test column array
#col = mesh_explicit.get_XYZ() #good
#save array to datafrome to pickle or something
#dataset_df = pd.DataFrame({'x': col[:, 0], 'time': col[:, 1], 'Temp': col[:, 2]})

#dataset_df.to_pickle("./temp2.pkl")
#to load unpickled_df = pd.read_pickle("./temp1.pkl")
# plot the solution to the PDE on the mesh
#mesh_explicit.plot_solution(ntimes=5, save_to_file=False, save_file_name='explicit.png')

#scipy.io.savemat('./fdSol.mat', mdict={'fdSol': col})
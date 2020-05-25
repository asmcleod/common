import numpy
import numpy as np
import numbers
import time
from scipy.fftpack import ifft
from common import misc
from common import numerics as num
from common import baseclasses
from common.baseclasses import AWA
from common.log import Logger
from mpmath import fp
from mpmath.calculus.quadrature import TanhSinh, GaussLegendre, QuadratureRule    
from cmath import *
import pdb

#--- Inversion utilities

class InverseLaplaceXform(object):
  """Adapted from:
laplace.py with mpmath
appropriate for high precision

Talbot suggested that the Bromwich line be deformed into a contour that begins
and ends in the left half plane, i.e., z \to \infty at both ends.
Due to the exponential factor the integrand decays rapidly
on such a contour. In such situations the trapezoidal rule converge
extraordinarily rapidly.
For example here we compute the inverse transform of F(s) = 1/(s+1) at t = 1

>>> error = Talbot(1,24)-exp(-1)
>>> error
  (3.3306690738754696e-015+0j)

Talbot method is very powerful here we see an error of 3.3e-015
with only 24 function evaluations

Created by Fernando Damian Nieuwveldt      
email:fdnieuwveldt@gmail.com
Date : 25 October 2009

Adapted to mpmath and classes by Dieter Kadelka
email: Dieter.Kadelka@kit.edu
Date : 27 October 2009

Reference
L.N.Trefethen, J.A.C.Weideman, and T.Schmelzer. Talbot quadratures
and rational approximations. BIT. Numerical Mathematics,
46(3):653 670, 2006."""

  def test_F(s): return 1/(1+s) #Should invert to *exp(-s*t)*

  def __init__(self,F=test_F,shift=0.0,N=24):
    self.F = F
    # test = Talbot() or test = Talbot(F) initializes with testfunction F

    self.shift = shift
    # Shift contour to the right in case there is a pole on the 
    #   positive real axis :
    # Note the contour will not be optimal since it was originally devoloped 
    #   for function with singularities on the negative real axis For example
    #   take F(s) = 1/(s-1), it has a pole at s = 1, the contour needs to be 
    #   shifted with one unit, i.e shift  = 1. 
    # But in the test example no shifting is necessary
 
    self.N = N
    # with double precision this constant N seems to best for the testfunction
    #   given. For N = 22 or N = 26 the error is larger (for this special
    #   testfunction).
    # With laplace.py:
    # >>> test.N = 500
    # >>> print test(1) - exp(-1)
    # >>> -2.10032517928e+21
    # Huge (rounding?) error!
    # with mp_laplace.py
    # >>> mp.dps = 100
    # >>> test.N = 500
    # >>> print test(1) - exp(-1)
    # >>> -5.098571435907316903360293189717305540117774982775731009465612344056911792735539092934425236391407436e-64

    #self.__call__=numpy.frompyfunc(self.__call__,1,1)

  def __call__(self,t):
      
    if t is 0 or \
        (hasattr(t,'__len__') and 0 in t):
        raise ValueError('Inverse transform cannot be calculated for t=0.')
          
    # Initiate the stepsize
    h = 2*numpy.pi/self.N
 
    ans =  0.0
    # parameters from
    # T. Schmelzer, L.N. Trefethen, SIAM J. Numer. Anal. 45 (2007) 558-571
    c1 = 0.5017
    c2 = 0.6407
    c3 = 0.6122
    c4 = 0+0.2645j
    
  # The for loop is evaluating the Laplace inversion at each point theta i
  #   which is based on the trapezoidal rule
    for k in range(self.N):
      theta = -numpy.pi + (k+0.5)*h
      z = self.shift + self.N/t*(c1*theta/numpy.tan(c2*theta) - c3 + c4*theta)
      dz = self.N/t * (-c1*c2*theta/numpy.sin(c2*theta)**2 + c1/numpy.tan(c2*theta)+c4)
      ans += numpy.exp(z*t)*self.F(z)*dz
          
    return ((h/(2j*numpy.pi))*ans)#.real #NOTE: it seems that there is in general no erroneous imaginary part for real argument, so make Talbot work for complex arguments

def InvertIntegralOperator(A,smoothing=1):
    
    #No need to do fancy smoothing if none is requested
    A=numpy.matrix(A)
    if smoothing==0: return A.getI()
    
    #Damping term (damps second derivative of solution by an amount gamma)
    N=A.shape[0]
    diag=6*numpy.eye(N)
    diag[0,0]=diag[-1,-1]=1
    diag[1,1]=diag[-2,-2]=5
    off_diag1=-4*numpy.eye(N)
    off_diag1[0,0]=off_diag1[-1,-1]=-2
    off_diag2=numpy.eye(N)
    H=numpy.matrix(numpy.roll(off_diag2,2,axis=0)+\
                   numpy.roll(off_diag1,1,axis=0)+\
                   diag+\
                   numpy.roll(off_diag1,1,axis=1)+\
                   numpy.roll(off_diag2,2,axis=1))
    H[N-2:N,0:2]=H[0:2,N-2:N]=0 #Don't couple non-adjacent ends
    
    #Effective inverse of matrix operator with smoothing
    gamma=smoothing*_get_minimal_smoothing(N)
    Ainv=(A.T*A+gamma**2*H).getI()*A.T
    
    return Ainv
    
def _get_minimal_smoothing(N):
    """An heuristic function for the "most optimal"
    (in some sense) smoothing parameter for inverting
    an integral operator of quadrature dimension *N*."""
    
    return 9.5e-11*N**1.71

#--- Matrix utilities

def orthogonalize(U, eps=1e-15):
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns 
    will be 0.
    
    Args:
        U (numpy.array): A d x n matrix with columns that need to be orthogonalized.
        eps (float): Threshold value below which numbers are regarded as 0 (default=1e-15).
    
    Returns:
        (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
            not linearly independent, then the last n-r cols are zeros.
    
    Examples:
    ```python
    >>> import numpy as np
    >>> import gram_schmidt as gs
    >>> gs.orthogonalize(np.array([[10., 3.], [7., 8.]]))
    array([[ 0.81923192, -0.57346234],
       [ 0.57346234,  0.81923192]])
    >>> gs.orthogonalize(np.array([[10., 3., 4., 8.], [7., 8., 6., 1.]]))
    array([[ 0.81923192 -0.57346234  0.          0.        ]
       [ 0.57346234  0.81923192  0.          0.        ]])
    
    
    Couresy:
        Anmol Kabra, Cornell
        https://gist.github.com/anmolkabra/b95b8e7fb7a6ff12ba5d120b6d9d1937 
    """
    
    import numpy as np
    import numpy.linalg as la
    
    n = len(U[0])
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    V = U.T
    for i in range(n):
        prev_basis = V[0:i]     # orthonormal basis before V[i]
        coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if la.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.   # set the small entries to 0
        else:
            V[i] /= la.norm(V[i])
    return V.T


#--- Numerical Quadrature Utilities

def clencurt(N):
  """ Computes the Clenshaw Curtis nodes and weights """
  if N == 1:
    x = 0
    w = 2
  else:
    n = N - 1
    C = np.zeros((N,2))
    k = 2*(1+np.arange(np.floor(n/2)))
    C[::2,0] = 2/np.hstack((1, 1-k*k))
    C[1,1] = -n
    V = np.vstack((C,np.flipud(C[1:n,:])))
    F = np.real(ifft(V, n=None, axis=0))
    x = F[0:N,1]
    w = np.hstack((F[0,0],2*F[1:n,0],F[n,0]))
  
  return x,w

class ClenshawCurtis(QuadratureRule):
    
    buffer=1e-2
    
    def calc_nodes(self,degree,prec):
        
        x,w=clencurt(N=degree)
        smallest_diff=1-x[-2]
        x*=1-self.buffer*smallest_diff
        
        return list(zip(x,w))
    
CC=ClenshawCurtis(fp)
TS=TanhSinh(fp)
GL=GaussLegendre(fp)
prec=4
#GL quadrature seems to work A LOT better, 
#it's as though the middle values of the kernel
#are important, and TS under-samples them...

def GetQuadrature(N=72,xmin=1e-3,xmax=numpy.inf,\
                  quadrature=GL,**kwargs):
    
    global xs,weights
    
    if hasattr(quadrature,'calc_nodes'):
        
        #deg=int(numpy.floor(numpy.log(N)/numpy.log(2)))
        
        if quadrature is GL: deg=numpy.log(2/3.*N)/numpy.log(2)
        elif quadrature is TS: deg=numpy.log(N)/numpy.log(2)-1
        elif quadrature is CC: deg=N
        else: deg=N
        
        #Degree always rounds up to provide at least the desired number `N` of points
        deg=int(numpy.ceil(deg))
        
        #The above formulas are just heuristics for the degree necessary for at least N samples
        #If it's not the case, might have to run once more...
        nodes=[]
        while len(nodes)<N:
            nodes=quadrature.calc_nodes(deg,prec); deg+=1
        
        nodes=quadrature.transform_nodes(nodes,a=xmin,b=xmax)
        
        xs,weights=list(zip(*nodes))
        xs,weights=misc.sort_by(xs,weights)
        xs=numpy.array(xs)
        weights=numpy.array(weights)
        #@bug: mpmath 0.17 has a bug whereby TS weights are overly large
        if quadrature is TS: weights*=3.8/numpy.float(len(weights))
        
    else:
        span=xmax-xmin
        if span==numpy.inf:
            raise ValueError('Infinite bounds are not supported for linear quadrature.')
            
        if quadrature=='linear' or quadrature==None:
            xs=numpy.linspace(xmin,xmax,N)
            weights=numpy.array([span/float(N)]*int(N))
            
        elif quadrature=='exponential':
            exkwargs=misc.extract_kwargs(kwargs,beta=1)
            beta=exkwargs['beta']
            M=N-1
            xs=xmin+span/float(beta)*numpy.log(M/(M-numpy.arange(M)*(1-numpy.exp(-beta))))
            xs=numpy.array(list(xs)+[span])
            weights=numpy.diff(xs)
            weights=numpy.array(list(weights)+[weights[-1]])
        
        elif quadrature=='double_exponential':
            exkwargs=misc.extract_kwargs(kwargs,beta=1)
            beta=exkwargs['beta']
            if N%2==0: N+=1
            M=N-1
            pref=2/float(M)*(numpy.exp(beta/2.)-1)
            
            js_lower=numpy.arange(numpy.ceil(M/2.))
            xs_lower=span*(1/2.-1/float(beta)*numpy.log(numpy.exp(beta/2.)-pref*js_lower))
            
            js_upper=numpy.arange(numpy.ceil(M)-numpy.ceil(M/2.))
            xs_upper=span*(1/2.+1/float(beta)*numpy.log(1+pref*js_upper))
            
            xs=xmin+numpy.array(list(xs_lower)+list(xs_upper)+[span])
            weights=numpy.diff(xs)
            weights=numpy.array(list(weights)+[weights[-1]])
        
        elif quadrature=='simpson':
            
            if N%2==0: N+=1
            xs=xmin+numpy.linspace(0,span,N)
            weights=numpy.zeros((N,))+2
            weights[(numpy.arange(N)%2)==0]=4
            weights[0]=1; weights[-1]=1
            weights*=span/float(N-1)/3.
    
    return xs,weights

#---Locally Weighted Scatterplot Smoothing
#Courtesy Ali Yahya
#Email github@ali01.com
#Location Stanford, CA
import math
from bisect import insort_left
from numpy.linalg.linalg import norm

def LOESS(X,Y,new_x=None,smoothing=0.05):
    
  if not isinstance(X, np.ndarray) \
     and hasattr(X,'__len__'): X=np.array(X)
     
  if not isinstance(Y, np.ndarray) \
     and hasattr(Y,'__len__'): Y=np.array(Y)
     
  if new_x is None: new_x=X
  elif not isinstance(new_x, np.ndarray) \
       and hasattr(new_x,'__len__'): new_x=np.array(new_x)
       
  return np.array([loess_query(x,np.mat(X).T,Y,alpha=smoothing).squeeze() \
                   for x in new_x])

def loess_query(x_query, X, y, alpha):
    
  y = np.mat(y).T
  x_query = np.array(x_query)

  if alpha <= 0 or alpha > 1:
    raise ValueError('ALPHA must be between 0 and 1')

  # inserting constant ones into X and X_QUERY for intercept term
  X = np.insert(X, obj=0, values=1, axis=1)
  x_query = np.insert(x_query, obj=0, values=1)

  # computing weights matrix using a tricube weight function
  W = weights_matrix(x_query, X, alpha)

  # computing theta from closed form solution to locally weighted linreg
  theta = (X.T * W * X).I * X.T * W * y

  # returning prediction
  return np.matrix.dot(theta.A.T, x_query)


def weights_matrix(x_query, X, alpha):
  if isinstance(x_query, np.matrix):
    x_query = x_query.A

  m = len(X)                # number of data points
  r = int(round(alpha * m)) # size of local region
  W = np.identity(m)        # skeleton for weights matrix

  sorted_list = []
  for i,row in enumerate(X):
    delta_norm = norm(row - x_query)
    insort_left(sorted_list, delta_norm)
    W[i][i] = delta_norm

  # computing normalization constant based on alpha
  h_i = 1 / sorted_list[r - 1]

  # normalizing weights matrix
  W = W * h_i

  # applying tricube weight function to weights matrix
  for i in range(0, len(W)):
    W[i][i] = (1 - (W[i][i] ** 3)) ** 3 if W[i][i] < 1 else 0

  return np.mat(W)

def ParameterFit(xs,ys,model_func,params0,\
                 limits=None,relative_error=False,\
                 verbose=False,error_exp=2,window_exp=.2,\
                 args=(),**kwargs):
    
    import numpy as np
    from scipy.optimize import leastsq
    Logger.raiseException('`model_func` must be a callable function of the fit parameters.',\
                          unless=hasattr(model_func,'__call__'), exception=TypeError)
    Logger.raiseException('`params0` must be a sequence of initial values for the model function parameters.',\
                          unless=(hasattr(params0,'__len__') and len(params0)), exception=TypeError)
    Logger.raiseException('`xs` and `ys` must be numeric sequences of identical length '+\
                          'greater than the number of fit parameters.',\
                          unless=(hasattr(xs,'__len__') and hasattr(ys,'__len__') and \
                                  len(xs)==len(ys) and len(xs)>len(params0)),\
                          exception=ValueError)
    
    #Coerce limits to the correct form#
    if not limits: limits=[]
    for i,limit_pair in enumerate(limits):
        if not limit_pair: limits[i]=[-np.inf,np.inf]
        else:
            for j,limit in enumerate(limit_pair):
                if limit is None: limit_pair[j]=(-1)**(j+1)*np.inf
    limits+=[[-np.inf,np.inf]]*(len(params0)-len(limits))
    
    #Define the form of the window function#
    all_window_lims=[]
    for limit_pair in limits:
        window_lims=[]
        for limit in limit_pair:
            if not np.isinf(limit): window_lims.append(limit)
        all_window_lims.append(window_lims)
        
    def window_func(params):
        
        window=1
        for i,param in enumerate(params):
            for window_lim in all_window_lims[i]:
                window*=np.abs((params0[i]-window_lim)/(param-window_lim))**window_exp
        
        return window
    
    #Define the form of the error function#
    if relative_error:
        Diff=lambda model,ys: np.abs((model-ys)/ys)**(error_exp/2.)
    else: Diff=lambda model,ys: np.abs(model-ys)**(error_exp/2.)
        
    #Define error function#
    def error_func(params,*args):
        
        model=model_func(xs,params,*args)
        window=window_func(params)
        
        return Diff(model,ys)*window
    
    return leastsq(error_func,params0,args=args,**kwargs)

#--- Image Modification Utilities

def smooth(x,window_len=4,window='blackman',mode='nearest',axis=-1):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    from scipy.ndimage import convolve1d

    ##Verify window length##
    if not isinstance(x,numpy.ndarray): x=numpy.array(x)
    Logger.raiseException('Input array must be larger than the window length along the axis specified.',\
                          unless=(x.shape[axis]>window_len),\
                          exception=ValueError)
    if window_len<3: return x

    ##Get Window##
    if isinstance(window,str):
        Logger.raiseException("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.",\
                              unless=(window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']),\
                              exception=ValueError)
        if window == 'flat': #moving average
            window=numpy.ones(numpy.round(window_len),'d')
        else:
            window=eval('numpy.'+window+'(window_len)')
        window=window/window.sum()

    ##Convolve##
    try: y=convolve1d(x,window,mode=mode,axis=axis)
    except TypeError:
        yreal=convolve1d(x.real,window,mode=mode,axis=axis)
        yimag=convolve1d(x.imag,window,mode=mode,axis=axis)
        y=yreal+1j*yimag
    
    #Make new array
    if isinstance(x,baseclasses.ArrayWithAxes):
        y=baseclasses.ArrayWithAxes(y)
        y.adopt_axes(x)
        y=x.__class__(y)
    
    return y

def CrossCorrelate(image1,image2=None,subtract_mean=True,normalize=True,window='blackman'):
    
    global s1,s2,m1,m2
    if not isinstance(image1,AWA): image1=AWA(image1)
    if image2 is None: image2=image1
    elif not isinstance(image2,AWA): image2=AWA(image2)
    
    m1=numpy.abs(numpy.mean(image1))
    m2=numpy.abs(numpy.mean(image2))
    
    tot1=numpy.abs(image1.integrate_axis(axis=0).integrate_axis(axis=1))
    tot2=numpy.abs(image2.integrate_axis(axis=0).integrate_axis(axis=1))
        
    #Remove DC power to apply window#
    s1=num.Spectrum(num.Spectrum(image1-m1,axis=-1,window=window),\
                   axis=0,window=window)
    s2=num.Spectrum(num.Spectrum(image2-m2,axis=-1,window=window),\
                   axis=0,window=window)
    if s1.shape is not s2.shape: s2=s1.get_coinciding_spectrum(s2)
             
    #Get autocorrelation function vs x,y component of displacement vector#
    Gxy=(s1*numpy.conj(s2)).get_inverse(axis=0).get_inverse(axis=-1).real
             
    #If we did not want mean subtracted, add its contribution back in
    if not subtract_mean:
        Gxy+=numpy.sqrt(tot1*tot2*m1*m2) #For autocorrelation, constant equals sum(image)*mean(image)
    
    #At infinity, an autocorrelation will therefore converge to mean image value
    if normalize: Gxy/=numpy.sqrt(tot1*tot2)
    
    Gxy=Gxy.sort_by_axes()
    xs,ys=Gxy.axes
    xs-=numpy.mean(xs)
    ys-=numpy.mean(ys)
    
    Gxy=numpy.roll(numpy.roll(Gxy,\
                              Gxy.shape[0]//2,\
                              axis=0),\
                   Gxy.shape[1]//2,\
                   axis=1)
    
    Gxy.set_axes([xs,ys])
    
    return Gxy

def RotationalAverageOfImage(image,Nthetas=200,NRs=200,Rmax=None,**kwargs):
    
    global on_polar,Xs,Ys
    from scipy.interpolate import griddata
    from scipy.integrate import trapz
    
    if not isinstance(image,baseclasses.AWA):
        Dx,Dy=image.shape
        image=baseclasses.AWA(image,axes=[numpy.linspace(-Dx/2.,Dx/2.,Dx),\
                                          numpy.linspace(-Dy/2.,Dy/2.,Dy)])
    xs,ys=image.axis_grids
    
    Rmax_sugg=numpy.min((numpy.max(xs),numpy.max(ys)))
    if not Rmax: Rmax=Rmax_sugg
    else: Rmax=numpy.min((Rmax_sugg,Rmax))
    
    Rs,angles=numpy.mgrid[0:Rmax:NRs*1j,0:2*numpy.pi:Nthetas*1j]
    R_axis=Rs[:,0]
    angle_axis=angles[0,:]
    
    Xs=Rs*numpy.cos(angles); Ys=Rs*numpy.sin(angles)
    on_polar=griddata(list(zip(xs.flatten(),ys.flatten())),\
                    image.flatten(),\
                    list(zip(Xs.flatten(),Ys.flatten())),\
                    fill_value=0,**kwargs)
    on_polar=on_polar.reshape((NRs,Nthetas))
    
    imageR=1/(2*numpy.pi)*baseclasses.AWA(trapz(x=angle_axis,y=on_polar,axis=1),\
                                          axes=[R_axis],axis_names=[r'$\rho$'])
    
    return imageR

def RotationalIntegralOfImage(image,Nthetas=200,NRs=200,Rmax=None,**kwargs):
    
    global on_polar,Xs,Ys
    from scipy.interpolate import griddata
    from scipy.integrate import trapz
    
    if not isinstance(image,baseclasses.AWA):
        Dx,Dy=image.shape
        image=baseclasses.AWA(image,axes=[numpy.linspace(-Dx/2.,Dx/2.,Dx),\
                                          numpy.linspace(-Dy/2.,Dy/2.,Dy)])
    xs,ys=image.axis_grids
    
    Rmax_sugg=numpy.min((numpy.max(xs),numpy.max(ys)))
    if not Rmax: Rmax=Rmax_sugg
    else: Rmax=numpy.min((Rmax_sugg,Rmax))
    
    Rs,angles=numpy.mgrid[0:Rmax:NRs*1j,0:2*numpy.pi:Nthetas*1j]
    R_axis=Rs[:,0]
    angle_axis=angles[0,:]
    
    Xs=Rs*numpy.cos(angles); Ys=Rs*numpy.sin(angles)
    on_polar=griddata(list(zip(xs.flatten(),ys.flatten())),\
                    image.flatten(),\
                    list(zip(Xs.flatten(),Ys.flatten())),\
                    fill_value=0,**kwargs)
    on_polar=on_polar.reshape((NRs,Nthetas))
    
    imageR=baseclasses.AWA(trapz(x=angle_axis,y=Rs*on_polar,axis=1),\
                           axes=[R_axis],axis_names=[r'$\rho$'])
    
    return imageR

def RadialAverageOfImage(image,Nthetas=200,NRs=200,**kwargs):
    
    global on_polar,Xs,Ys
    from scipy.interpolate import griddata
    from scipy.integrate import trapz
    
    if not isinstance(image,baseclasses.AWA): image=baseclasses.AWA(image)
    xs,ys=image.axis_grids
    
    Rmax=numpy.min((numpy.max(xs),numpy.max(ys)))
    Rs,angles=numpy.mgrid[0:Rmax:NRs*1j,0:2*numpy.pi:Nthetas*1j]
    R_axis=Rs[:,0]
    angle_axis=angles[0,:]
    
    Xs=Rs*numpy.cos(angles); Ys=Rs*numpy.sin(angles)
    on_polar=griddata(list(zip(xs.flatten(),ys.flatten())),\
                    image.flatten(),\
                    list(zip(Xs.flatten(),Ys.flatten())),**kwargs)
    on_polar=on_polar.reshape((NRs,Nthetas))
    
    image_theta=baseclasses.AWA(trapz(x=R_axis,y=Rs*on_polar,axis=0),\
                                axes=[angle_axis],axis_names=[r'$\theta$'])
    
    return image_theta

def PolynomialFitImage(image,order=3,\
                       full_output=False):
    
    image=AWA(image)
    assert image.ndim==2,'Image must be 2-dimensional.'
    
    X,Y=image.axis_grids
    B=image.flatten()
    
    #Generate bases of the 2-dimensional polynomial
    bases=[]
    for n in range(order+1):
        for i in range(n+1):
            bases.append(X**(n-i)*Y**i)
    A=np.array([basis.flatten() for basis in bases]).T
    
    coeffs, r, rank, s = np.linalg.lstsq(A, B)
    
    fit=np.sum([coeff*basis for coeff,basis \
                in zip(coeffs,bases)],axis=0)
    fit=AWA(fit); fit.adopt_axes(image)
    
    if full_output: return {'fit':fit,  'coefficients':coeffs,\
                            'R':r,      'rank':rank,            's':s}
    else: return fit

def FourierFilterImage(image,fmin=0,fmax=None,sharp=False,order=4,\
                       square=False,fminy=None,fmaxy=None,window=None):
    
    def high_pass(fabs,fmin):
        
        if not fmin: return 1
        else: return ((fabs/fmin)/numpy.sqrt(1+(fabs/fmin)**2))**order
        
    def low_pass(fabs,fmax):
        
        if not fmax: return 1
        return (1/numpy.sqrt(1+(fabs/fmax)**2))**order
    
    #Subtract average value before taking spectrum; will restore it later
    avg_val=image
    for i in range(2): avg_val=np.mean(avg_val,axis=-1)
    avg_val=np.array(avg_val) #just in case we have a single image, make number into array before resize
    avg_val.resize(avg_val.shape+(1,1)) #resize for broadcasting
    image=image-avg_val #make sure not to modify `image` in-place
    is_low_pass=True
    
    #Spectrum
    s=image
    for i in [-1,-2]: s=num.Spectrum(s,axis=i,window=window)
    fgrids=s.axis_grids[-2:]
    
    ##Rotationally symmetric filtering##
    if not square:
        if fmin!=0: is_low_pass=False
        
        fabs_grid=numpy.sqrt(numpy.sum([fgrid**2 for fgrid in fgrids],axis=0))
        if sharp: mask=(fabs_grid>=fmin)*(fabs_grid<=fmax)
        else: mask=high_pass(fabs_grid,fmin)*low_pass(fabs_grid,fmax)
        
    ##Square filtering##
    else:
        if fminy is None: fminy=fmin
        if fmaxy is None: fmaxy=fmax
        if fmin!=0 or fminy!=0: is_low_pass=False
        
        fx=numpy.abs(fgrids[0]); fy=numpy.abs(fgrids[1])
        if sharp: mask=(fx>=fmin)*(fx<=fmax)*(fy>=fminy)*(fy<=fmaxy)
        else: mask=high_pass(fx,fmin)*low_pass(fx,fmax)*\
                   high_pass(fy,fminy)*low_pass(fy,fmaxy)
    
    smasked=s*mask
    
    filtered=smasked
    for i in [-1,-2]: filtered=filtered.get_inverse(axis=i)
    
    filtered=filtered.astype(float)
    
    filtered=num.interpolating_resize(filtered,image.shape)
    if isinstance(image,AWA):
        filtered=AWA(filtered)
        filtered.adopt_axes(image)
        
    #Restore average value, provided we weren't doing high-pass
    if is_low_pass: filtered+=avg_val
        
    return filtered

class QuickConvolver(object):
    """Convolves an image with a kernel, using a pre-calculated kernel function.
    Useful for computing many convolutions using the same kernel.
    
    Convolution is performed using the Fourier convolution theorem.
    
    Note: Kernel should be a "density" function with invariant L2 norm over grid
    of `xs` and `ys`, for any choice of x and y pixel spacing `dx` and `dy`.
    A good example is the Gaussian function, with amplitude compatible with its
    full-width at half-maximum, according to its `ArrayWithAxes` axes or with input
    `size`.
    
    Supports constant-value or mirror-image padding of input image to mitigate edge effects.
    
    `kwargs` are passed through to `kernel_function` or `kernel_function_fourier`.
    
    EXAMPLE:
    
    Consider the image:
    >>> image=AWA(zeros((101,101)),axes=[linspace(-.5,.5,101)]*2)
    >>> image[50,50]=1
    
    Try the first convolution:
    >>> kernel_function=lambda x,y: 1/sqrt(x**2+y**2+1e-8**2)
    >>> qc1=QC(size=(1,1),shape=(101,101),pad_by=.5,kernel_function=kernel_function)
    >>> result1=qc1(image)
    >>> result1-=result1.min() #overall offset, while correct, should not be meaningful
    >>> result1[result1==result1.max()]=0 #point at center is controlled by 1e-8
    
    And the second convolution:
    >>> kernel_function_fourier=lambda kx,ky: 2*pi/sqrt(kx**2+ky**2+1e-8**2)
    >>> qc2=QC(size=(1,1),shape=(101,101),pad_by=.5,kernel_function_fourier=kernel_function_fourier)
    >>> result2=qc2(image)
    >>> result2-=result2.min() #overall offset is controlled by 1e-8
    
    And compare:
    >>> figure();result1.cslice[0].plot(plotter=semilogy)
    >>> result2.cslice[0].plot()
    >>> gca().set_xscale('symlog',linthreshx=1e-2)
    """
    
    def __init__(self,shape=None,
                 size=(1,1),
                 pad_by=0,
                 pad_with=0,
                 pad_mult=np.zeros((3,3))+1,
                 kernel_function_fourier=None,\
                 kernel_function=None,
                 kernel=None,
                 xs=None,ys=None,
                 **kwargs):
        
        assert isinstance(pad_with,numbers.Number) \
                or pad_with in ('mirror',),\
                'Argument `pad_with`=%s not understood!'%repr(pad_with)
        pad_mult=np.asarray(pad_mult,dtype=np.float64)
        assert pad_mult.shape==(3,3),\
                'Argument `pad_mult`=%s must be a 3x3 array!'%repr(pad_mult)
        self.pad_by=pad_by
        self.pad_with=pad_with
        self.pad_mult=pad_mult
        
        assert kernel_function is not None \
            or kernel_function_fourier is not None \
            or kernel is not None,\
                'One of `kernel_function_fourier`, `kernel_function`, '+\
                'or `kernel` must be provided!'
        
        self.kernel_function=kernel_function
        self.kernel_function_fourier=kernel_function_fourier
        
        if kernel_function_fourier is not None: 
            self.recompute_kernel_fourier(shape,size,**kwargs)
        elif kernel_function is not None:
            self.recompute_kernel(shape,size,**kwargs)
        else: self.set_kernel(kernel,xs,ys)
    
    def pad_const(self,arr,dN,const=0):
        
        s=arr.shape
        dNx,dNy=dN
        
        nw_tile=np.full((dNx,dNy),const)*self.pad_mult[0,0]
        n_tile=np.full((s[0],dNy),const)*self.pad_mult[0,1]
        ne_tile=np.full((dNx,dNy),const)*self.pad_mult[0,2]
        
        w_tile=np.full((dNx,s[1]),const)*self.pad_mult[1,0]
        e_tile=np.full((dNx,s[1]),const)*self.pad_mult[1,2]
        
        sw_tile=np.full((dNx,dNy),const)*self.pad_mult[2,0]
        s_tile=np.full((s[0],dNy),const)*self.pad_mult[2,1]
        se_tile=np.full((dNx,dNy),const)*self.pad_mult[2,2]
        
        return np.hstack((np.vstack((nw_tile,   n_tile,     ne_tile)),\
                          np.vstack((w_tile,    arr,        e_tile)),\
                          np.vstack((sw_tile,   s_tile,     se_tile))))
        
    def pad_mirror(self,arr,dN,sign=None):
        
        dNx,dNy=dN
        Nx,Ny=arr.shape
        
        nw_tile=arr[:dNx,:dNy]\
                    [::-1,::-1]*self.pad_mult[0,0]
        n_tile=arr[:,:dNy]\
                    [::1,::-1]*self.pad_mult[0,1]
        ne_tile=arr[Nx-dNx:,:dNy]\
                    [::-1,::-1]*self.pad_mult[0,2]
                    
        w_tile=arr[:dNx]\
                    [::-1,::1]*self.pad_mult[1,0]
        e_tile=arr[Nx-dNx:]\
                    [::-1,::1]*self.pad_mult[1,2]
        
        sw_tile=arr[:dNx,Ny-dNy:]\
                    [::-1,::-1]*self.pad_mult[2,0]
        s_tile=arr[:,Ny-dNy:]\
                    [::1,::-1]*self.pad_mult[2,1]
        se_tile=arr[Nx-dNx:,Ny-dNy:]\
                    [::-1,::-1]*self.pad_mult[2,2]
        
        return np.hstack((np.vstack((nw_tile,   n_tile,     ne_tile)),\
                          np.vstack((w_tile,    arr,        e_tile)),\
                          np.vstack((sw_tile,   s_tile,     se_tile))))
    
    @classmethod
    def remove_pad(cls,im,dN):
        
        Nx,Ny=im.shape
        dNx,dNy=dN
        result=im[dNx:Nx-dNx,\
                  dNy:Ny-dNy]
        
        return result
    
    def set_axes(self,shape,size):
        
        self.shape=shape
        
        if hasattr(self.pad_by,'__len__'):
            pad_by_x,pad_by_y=self.pad_by
        else: pad_by_x=pad_by_y=self.pad_by
        
        pad_by_x=np.min((pad_by_x,1))
        pad_by_y=np.min((pad_by_y,1))
        
        dNx=int(pad_by_x*shape[0])
        dNy=int(pad_by_y*shape[1])
        self.dN=(dNx,dNy)
        self.padded_shape=[N+2*dN for N,dN in zip(shape,\
                                                  (dNx,dNy))]
        
        padded_size=(size[0]*self.padded_shape[0]/float(shape[0]),\
                     size[1]*self.padded_shape[1]/float(shape[1]))
        
        #We could use `pad_by` to extend range, but using dN values
        # ensures the point spacing remains the same, so the unpadded
        # part of the axes will remain unchanged.
        self.xs=np.linspace(-padded_size[0]/2.,padded_size[0]/2.,
                            self.padded_shape[0])
        self.ys=np.linspace(-padded_size[1]/2.,padded_size[1]/2.,
                            self.padded_shape[1])
        
        self.dx=padded_size[0]/float(self.padded_shape[0])
        self.dy=padded_size[1]/float(self.padded_shape[1])
        
    def norm_fourier(self,kernel_fourier):
        
        norm=self.dx*self.dy #This is correct- convolution of two fields gives same result irrespective of grid coarseness
        
        return kernel_fourier*norm
    
    def set_kernel(self,kernel,xs=None,ys=None):
        
        #Infer size from axes, if possible
        if isinstance(kernel,AWA): xs,ys=kernel.axes
        else:
            assert xs is not ys is not None,\
                'If `kernel` is not an `ArrayWithAxes` instance, '+\
                'please explicitly provide `xs` and `ys` for array axes.'
            kernel=AWA(kernel,axes=[xs,ys])
            
        size=(np.max(xs)-np.min(xs),\
              np.max(ys)-np.min(ys))
            
        #Set uniform axes and inherit zero position from original axes
        self.set_axes(kernel.shape,size)
        self.xs+=np.mean(xs)
        self.ys+=np.mean(ys)
            
        #Interpolate to padded axes
        self.kernel=kernel.interpolate_axis(self.xs,axis=0,bounds_error=False,\
                                            extrapolate=False,fill_value=0)\
                          .interpolate_axis(self.ys,axis=1,bounds_error=False,\
                                            extrapolate=False,fill_value=0)
        
        #Roll the kernel so that its origin position lies at index (0,0) before FFT
        x_0pos=np.argmin(np.abs(self.xs))
        y_0pos=np.argmin(np.abs(self.ys))
        rolled_kernel=np.roll(np.roll(self.kernel,\
                                      -x_0pos,axis=0),\
                              -y_0pos,axis=1)
        self.rolled_kernel=rolled_kernel
        
        self.kernel_fourier=np.fft.fft(np.fft.fft(rolled_kernel,\
                                                  axis=0),\
                                       axis=1)
        self.kernel_fourier=self.norm_fourier(self.kernel_fourier)
    
    def recompute_kernel(self,shape,
                         size=(1,1),
                         kernel_function=None,\
                         **kwargs):
        
        if kernel_function: self.kernel_function=kernel_function
        else: kernel_function=self.kernel_function
        
        assert kernel_function is not None,\
            'No default `kernel_function`, provide one!'
        
        self.set_axes(shape,size)
        
        xs_grid=self.xs.reshape((len(self.xs),1))
        ys_grid=self.ys.reshape((1,len(self.ys)))
        
        kernel=kernel_function(xs_grid,ys_grid,**kwargs)
        self.kernel=AWA(kernel,axes=(self.xs,self.ys))
        
        #Roll the kernel so that its origin position lies at index (0,0) before FFT
        x_0pos=np.argmin(np.abs(self.xs))
        y_0pos=np.argmin(np.abs(self.ys))
        
        rolled_kernel=np.roll(np.roll(self.kernel,\
                                      -x_0pos,axis=0),\
                              -y_0pos,axis=1)
        self.rolled_kernel=rolled_kernel
        
        self.kernel_fourier=np.fft.fft(np.fft.fft(rolled_kernel,axis=0),axis=1)
        self.kernel_fourier=self.norm_fourier(self.kernel_fourier)
        
    def recompute_kernel_fourier(self,shape,
                                 size,
                                 kernel_function_fourier=None,\
                                 **kwargs):
        
        if kernel_function_fourier: self.kernel_function_fourier=kernel_function_fourier
        else: kernel_function_fourier=self.kernel_function_fourier
        
        assert kernel_function_fourier is not None,\
            'No default `kernel_function_fourier`, provide one!'
        
        self.set_axes(shape,size)
        Nx,Ny=self.padded_shape
        
        kxs=2*np.pi*np.fft.fftfreq(Nx,self.dx)
        kys=2*np.pi*np.fft.fftfreq(Ny,self.dy)
        
        kxs_grid=kxs.reshape((len(kxs),1))
        kys_grid=kys.reshape((1,len(kys)))
        
        self.kernel_fourier=kernel_function_fourier(kxs_grid,\
                                                    kys_grid,**kwargs)
        self.kernel=None
    
    def __call__(self,im):
        
        assert im.shape == self.shape,\
            'Shape of `im` must match that of kernel: %s.'%repr(self.shape)+\
            '  Otherwise, use method `set_axes` to reset the size and spatial range of the kernel.'
        
        self.im=im
        
        # Pad input if desired
        if self.pad_by:
            if self.pad_with=='mirror':
                im_padded=self.pad_mirror(im, self.dN)
            elif self.pad_with=='mirror_inv':
                im_padded=self.pad_mirror(im, self.dN,sign=-1)
            else:
                im_padded=self.pad_const(im, self.dN, self.pad_with)
            self.im_padded=im_padded
        else: self.im_padded=im
        
        im_fourier=np.fft.fft(np.fft.fft(self.im_padded,axis=0),axis=1)
        mult=im_fourier*self.kernel_fourier
        
        result = np.fft.ifft(np.fft.ifft(mult,axis=0),axis=1)
        
        #case to real if neither kernel nor image were complex
        if not np.iscomplexobj(self.kernel_fourier) and not np.iscomplexobj(im): result=result.real
        
        #Shave off padded zone of output
        if self.pad_by: result=self.remove_pad(result, self.dN)
        
        #Provide axes to match input
        if isinstance(im,AWA):
            result=AWA(result); result.adopt_axes(im)
        
        return result

############################################
#--- Image synchronization routine components #
############################################

def baryocentric_coords(pts,pt):
    """See e.g.: http://en.wikipedia.org/wiki/Barycentric_coordinate_system_%28mathematics%29"""
    
    xs,ys=list(zip(*pts))
    x,y=pt
    
    det=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
    l1=((ys[1]-ys[2])*(x-xs[2])+(xs[2]-xs[1])*(y-ys[2]))/float(det)
    l2=((ys[2]-ys[0])*(x-xs[2])+(xs[0]-xs[2])*(y-ys[2]))/float(det)
    
    return l1, l2, 1-l1-l2

class AffineXform(object):
    
    def __init__(self,pts1,pts2):
    
        xs1,ys1=list(zip(*pts1))
        self.xs1=numpy.array(xs1).astype(numpy.float)
        self.ys1=numpy.array(ys1).astype(numpy.float)
        
        xs2,ys2=list(zip(*pts2))
        self.xs2=numpy.array(xs2).astype(numpy.float)
        self.ys2=numpy.array(ys2).astype(numpy.float)
        
        centerx1=numpy.mean(xs1)
        centery1=numpy.mean(ys1)
        
        centerx2=numpy.mean(xs2)
        centery2=numpy.mean(ys2)
        
        pts1=pts1[:2] #select only the first two points now
        pts2=pts2[:2]
        
        pts1=pts1-numpy.array([centerx1,centery1]).reshape((1,2)) #row vectors
        pts2=pts2-numpy.array([centerx2,centery2]).reshape((1,2)) #row vectors
        
        Rmat1=numpy.matrix(pts1).T #each position vector makes a column
        Rmat2=numpy.matrix(pts2).T
        
        self.Amat=Rmat2*Rmat1.I
        
        centervec1=numpy.matrix([centerx1,centery1]).T
        centervec2=numpy.matrix([centerx2,centery2]).T
        
        self.bvec=centervec2-self.Amat*centervec1
        
    def __call__(self,x,y):
        
        rvec1=numpy.matrix([x,y]).T
        rvec2=self.Amat*rvec1+self.bvec
        
        return numpy.array(rvec2).squeeze()
    
    def test(self):
        
        from matplotlib.pyplot import figure,plot
        
        figure()
        plot(self.xs1,self.ys1,color='b',marker='o')
        plot(self.xs2,self.ys2,color='g',marker='o')
        
        centerx1=numpy.mean(self.xs1)
        centery1=numpy.mean(self.ys1)
        sx=numpy.std(self.xs1)
        sy=numpy.std(self.ys1)
        
        xs=numpy.linspace(-sx,sx,10)+centerx1
        ys=numpy.linspace(-sx,sx,10)+centery1
        
        for x1 in xs:
            for y1 in ys:
                plot([x1],[y1],marker='+',color='b')
                x2,y2=self(x1,y1)
                plot([x2],[y2],marker='+',color='g')

def AffineGridsFromFeaturePoints(ref_pt_pairs,\
                                    target_pt_pairs,
                                    xs=None,\
                                    ys=None,\
                                    Nxs=100,\
                                    Nys=100,\
                                    prefer_BC=False,\
                                    BC_attempts=5,\
                                    triangle_thresh=.05):
    """Map a grid from one (reference) image to affine grids in the other (target)
    images using a comparison of their feature point sets.
    
    This implementation uses a Delaunay triangulation in the reference (undistorted) space to assign a simplex to each
    point in the grid intended to map from the reference space to the target space.  This ensures that each
    grid point receives an affine transformation appropriate to its intended simplex."""
    
    from scipy.spatial import Delaunay
    global ref_Del
    
    def pts_make_triangle(pts,thresh=triangle_thresh):
        
        ((Ax,Bx,Cx),(Ay,By,Cy))=list(zip(*pts))
        area=numpy.abs((Ax*(By-Cy)+Bx*(Cy-Ay)+Cx*(Ay-By)))/2.
        
        return area>thresh
    
    # Build x- and y-values of grid if none are specified #
    if xs is None or ys is None:
        pts1_xs,pts1_ys=list(zip(*ref_pt_pairs))
        if xs is None:
            xmin,xmax=numpy.min(pts1_xs),numpy.max(pts1_xs)
            xs=numpy.linspace(xmin,xmax,Nxs)
        if ys is None:
            ymin,ymax=numpy.min(pts1_ys),numpy.max(pts1_ys)
            ys=numpy.linspace(ymin,ymax,Nys)
            
    #Make sure feature pts come in lists of tuples
    ref_pts=[tuple(pt) for pt in ref_pt_pairs]
    target_pts=[[tuple(pt) for pt in this_target_pt_pairs] \
                for this_target_pt_pairs in target_pt_pairs]
    Ntargets=len(target_pts)
    
    #Need also an array form of reference feature points for distance calculations
    ref_pts_arr=numpy.array(ref_pts)
    
    #Broadcast the axes into grids and then into xy pairs
    xgrid=numpy.array(xs).reshape((len(xs),1))
    ygrid=numpy.array(ys).reshape((1,len(ys)))
    xgrid=xgrid+0*ygrid; ygrid=ygrid+0*xgrid
    xy_pts=list(zip(xgrid.flatten(),ygrid.flatten()))
    
    #Find the simplices of the reference Delaunay triangulation which host the xy points
    Logger.write('Performing Delaunay triangulation of anchor points in the reference space...')
    ref_Del=Delaunay(ref_pt_pairs)
    simplex_inds=ref_Del.find_simplex(xy_pts)
    
    #Take these simplices into their respective feature point indices (sorted) in the reference space
    Logger.write('Identifying simplices of associated with grid points in the Delaunay triangulation...')
    all_simplex_pt_inds=[]
    k=0; t0=time.time()
    for xy_pt,simplex_ind in zip(xy_pts,simplex_inds):
        
        #If xy point `i` is baryocentric in a simplex, retrieve the simplex pts
        if simplex_ind!=-1: 
            try: simplex_pt_inds=ref_Del.simplices[simplex_ind]
            except AttributeError: simplex_pt_inds=ref_Del.vertices[simplex_ind] #`vertices` attribute is deprecated reference to `simplices`
        
        #The point is outside the convex hull, so get closest 3 points in
        #reference space which actually form a triangle
        else:
            #Get distance from this xy pt to each reference pt#
            xy_pt_arr=numpy.array(xy_pt).reshape((1,2))
            distances=numpy.sqrt(numpy.sum((ref_pts_arr-xy_pt_arr)**2,axis=-1))
            sorted_inds=numpy.argsort(distances)
            
            #Do an ordered check of these reference point indices in
            #groups of three until one group forms a triangle
            j=0; is_tri=False
            while not is_tri:
                try: simplex_pt_inds=sorted_inds[j:j+3]
                #Reached the end of the points, use the first available group and hope for best
                except IndexError: simplex_pt_inds=sorted_inds[0:3]; break
                
                #Do we have a triangle?
                tri_pts=[ref_pts_arr[ind] for ind in simplex_pt_inds]
                is_tri=pts_make_triangle(tri_pts)
                j+=1
        
        all_simplex_pt_inds.append(tuple(sorted(simplex_pt_inds)))
        k+=1
                
        #Progress
        t=time.time()
        if t-t0>1:
            Logger.write('\tProgress: %1.2f%% complete...'\
                                      %(100.*k/float(len(xy_pts))))
            t0=t
    
    #Loop through each target space
    Logger.write('Mapping grid points from reference space into target spaces...')
    all_target_grid_pts=[]
    for n in range(Ntargets):
        
        Logger.write('Mapping grid points into target space %i...'%(n+1))
        
        #Loop through each xy point and use corresponding simplex index trio for affine transform
        affine_xforms={}
        
        #Loop in x-direction
        target_grid_pts_along_x=[]
        k=0; t0=time.time()
        for i,x in enumerate(xs):
            
            #Loop in y-direction
            target_grid_pts_along_y=[]
            for j,y in enumerate(ys):
            
                #Get index of this xy point and associated indices of simplex points
                pt_ind=i*len(ys)+j #X changes most quickly
                simplex_pt_inds=all_simplex_pt_inds[pt_ind]
                
                #Try to request an existing affine transform
                try: affine_xform=affine_xforms[simplex_pt_inds]
                
                #Or make one and store it for continued use with these simplex points
                except KeyError:
                    #Get the explicit reference/target points from their indices
                    ref_pt_trio=[ref_pts[simplex_pt_ind] for simplex_pt_ind in simplex_pt_inds]
                    target_pt_trio=[target_pts[n][simplex_pt_ind] for simplex_pt_ind in simplex_pt_inds]
                    affine_xform=AffineXform(ref_pt_trio,target_pt_trio)
                    affine_xforms[simplex_pt_inds]=affine_xform
                    
                #Use the affine xform to map the xy pair
                target_x,target_y=affine_xform(x,y)
                #print target_x,target_y
                target_grid_pts_along_y.append((target_x,target_y))
                
                #Progress
                t=time.time()
                if t-t0>1:
                    progress=(n*len(xs)*len(ys)+i*len(ys)+j)/numpy.float(Ntargets*len(xy_pts))*100
                    Logger.write('\tProgress: %1.1f%%...'%progress)
                    t0=t
            
            target_grid_pts_along_x.append(target_grid_pts_along_y)
            k+=1
            
        # We have mapped all grid points into this target space now
        all_target_grid_pts.append(target_grid_pts_along_x)
    
    return {'ref_grid_pts':[[(x,y) for y in ys] for x in xs],\
            'grid_pts':all_target_grid_pts}

def InterpolateImageToAffineGrid(image,grid_pts,\
                                 image_xgrid=None,
                                 image_ygrid=None,\
                                 **kwargs):
    """Distort an image to a new one by interpolating to an affine grid of points."""
    
    global rbs
    from scipy.interpolate import RegularGridInterpolator,\
                                  SmoothBivariateSpline,\
                                  LinearNDInterpolator
    
    Nx,Ny=numpy.array(grid_pts).shape[:2] #Get the XY shape of the mapped grid
    interpolated=numpy.zeros((Nx,Ny))
    
    regular=False
    if image_xgrid is None or image_ygrid is None:
        regular=True
        if isinstance(image,AWA): xs,ys=image.axes
        else:
            xs=np.arange(image.shape[0])
            ys=np.arange(image.shape[1])
    
    Logger.write('Generating bivariate interpolator...')
    if regular: Interp=RegularGridInterpolator((xs,ys),\
                                               np.array(image),\
                                               **kwargs)
    else:
        Interp=LinearNDInterpolator(list(zip(image_xgrid.flatten(),\
                                        image_ygrid.flatten())),\
                                     image.T.flatten(),**kwargs)
    
    Logger.write('Interpolating image...')
    
    t0=time.time()
    grid_pts=list(grid_pts)
    for i in range(Nx):
        for j in range(Ny):
            grid_pt=grid_pts[i][j]
            if regular: args=(grid_pt,)
            else: args=grid_pt
            interpolated[i,j]=Interp(*args)
            t=time.time()
            if t-t0>1:
                progress=(i*Ny+1)/numpy.float(Nx*Ny)*100
                Logger.write('\tProgress: %1.1f%%'%progress)
                t0=t
            
    return interpolated


def Synchronize2DImages(images,within_pts=True,fontsize=14,prefer_BC=False,cmap='gray'):
    """Interactively synchronize a set of images by first identifying their feature points.
    
    Uses functions `AffineGridsFromFeaturePoints` and `InterpolateImageToAffineGrid`."""
    
    import copy
    import time
    import itertools
    from matplotlib import pyplot as plt
    from common.plotting import PointPicker
    
    Logger.raiseException('Input `images` must be a list of 2-D arrays of length 2 or greater.',\
                          unless=(len(images)>=2 and isinstance(images[0],numpy.ndarray) and \
                                  False not in [image.ndim==2 for image in images]),\
                          exception=TypeError)
                
    def inherit_lines_and_texts(ax,lines,texts):
        
        for l1 in lines:
            l2=ax.plot([0],[0])[0]
            l2.set_xdata(l1.get_xdata())
            l2.set_ydata(l1.get_ydata())
            l2.set_color(l1.get_color())
            l2.set_marker(l1.get_marker())
            
        for t1 in texts:
            x,y=t1.get_position()
            ax.text(x,y,t1.get_text(),fontsize=t1.get_fontsize(),\
                    bbox=dict(facecolor='white', alpha=0.25),
                         horizontalalignment='center',\
                         verticalalignment='bottom')
            
    cax=None
                
    ## Now define the script ##
    # Make sure the images are bare arrays and then make them vanilla AWAs #
    images=[baseclasses.AWA(numpy.array(image)) for image in images]
    all_pts=[]
    
    f=plt.figure()
    f.set_size_inches((15,7),forward=True)
    plt.subplots_adjust(wspace=.4)
    
    for i in range(len(images)):
        
        if i==1:
            # Get relevant data from "image 1" plot in panel 2 and copy into panel 1 #
            copied_clim=ax2.images[0].get_clim()
            lines=ax2.lines
            texts=ax2.texts
            
            ax1=plt.subplot(121)
            plt.sca(ax1); ax1.clear()
            images[i-1].plot(colorbar=False,cmap=cmap)
            ax1.autoscale(False)
            ax1.images[0].set_clim(copied_clim)
            #[ax1.add_line(line) for line in lines] #`ax.add_line` appears to be broken...`
            inherit_lines_and_texts(ax1,lines,texts)
            plt.title('Image %i'%i)
            plt.tight_layout()
            plt.colorbar()
            plt.draw()
        
        # Plot new image with appropriate color limits #
        image=images[i]
        mn=numpy.mean(image)
        std=numpy.std(image)
        
        if i==0: ax2=plt.subplot(122)
        else: ax2.clear(); plt.sca(ax2)
        image.plot(colorbar=False,cmap=cmap)
        ax2.autoscale(False)
        clim_min=mn-3*std
        clim_max=mn+3*std
        ax2.images[0].set_clim(clim_min,clim_max)
        plt.title('Image %i'%(i+1))
        if i==0:
            plt.tight_layout()
            cbar=plt.colorbar(); cax=cbar.ax
        else:
            plt.subplots_adjust(right=.9,wspace=.4)
            # This is supposed to induce the colorbar to now change on assignment of new color limits #
            # Does not induce updates, seems broken #
            cbar.update_normal(ax2.images[0])
        plt.draw()
        
        print('Now begin selecting feature points on image %i by clicking with the right mouse button. '%(i+1)+\
              'You can zoom into the plot in the usual fashion using the zoom tool and the left mouse button. '+\
              'Press [c] to edit the color scale.')
        if i==0:
            print('When a sufficient number of points have been selected to span the image, press [enter] to stop. '+\
                   'Note that the same points will have to be selected also for the next %i images.'%(len(images)-1))
            max_pts=None
        else:
            max_pts=len(all_pts[0])
            print('Make sure to select the same %i points in the same order as in image %i as displayed in the left panel.'%(max_pts,i))
        
        print()
        Picker=PointPicker(mousebutton=3,max_pts=max_pts,ax=ax2,cbar=cbar,fontsize=fontsize,verbose=True)
        all_pts.append(Picker.get_points())
        
        print('You have finished with image %i!\n'%(i+1))

    Logger.write('Synchronizing grids from the indicated feature points...')
    time.sleep(.5)
    grids=AffineGridsFromFeaturePoints(all_pts[0],all_pts[1:],\
                                              xs=images[0].axes[0],
                                              ys=images[0].axes[1],
                                              prefer_BC=prefer_BC)
    
    corrected_images=[images[0]]
    for i,image in enumerate(images[1:]):
        corrected_image=InterpolateImageToAffineGrid(image,grids['grid_pts'][i])
        corrected_images.append(corrected_image)
        
    return {'corrected_images':corrected_images,'grids':grids}

######################
#--- Peak Picking
######################

import numpy as np
from math import pi, log
import pylab
from scipy import fft, ifft
from scipy.optimize import curve_fit
    
    
def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200) 
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*tab)
    """
    
    def _datacheck_peakdetect(x_axis, y_axis):
        if x_axis is None:
            x_axis = list(range(len(y_axis)))
        
        if len(y_axis) != len(x_axis):
            raise ValueError
        
        #needs to be a numpy array
        y_axis = np.array(y_axis)
        x_axis = np.array(x_axis)
        return x_axis, y_axis
        
    def _peakdetect_parabole_fitter(raw_peaks, x_axis, y_axis, points):
        """
        Performs the actual parabole fitting for the peakdetect_parabole function.
        
        keyword arguments:
        raw_peaks -- A list of either the maximium or the minimum peaks, as given
            by the peakdetect_zero_crossing function, with index used as x-axis
        x_axis -- A numpy list of all the x values
        y_axis -- A numpy list of all the y values
        points -- How many points around the peak should be used during curve
            fitting, must be odd.
        
        return -- A list giving all the peaks and the fitted waveform, format:
            [[x, y, [fitted_x, fitted_y]]]
            
        """
        func = lambda x, k, tau, m: k * ((x - tau) ** 2) + m
        fitted_peaks = []
        for peak in raw_peaks:
            index = peak[0]
            x_data = x_axis[index - points // 2: index + points // 2 + 1]
            y_data = y_axis[index - points // 2: index + points // 2 + 1]
            # get a first approximation of tau (peak position in time)
            tau = x_axis[index]
            # get a first approximation of peak amplitude
            m = peak[1]
            
            # build list of approximations
            # k = -m as first approximation?
            p0 = (-m, tau, m)
            popt, pcov = curve_fit(func, x_data, y_data, p0)
            # retrieve tau and m i.e x and y value of peak
            x, y = popt[1:3]
            
            # create a high resolution data set for the fitted waveform
            x2 = np.linspace(x_data[0], x_data[-1], points * 10)
            y2 = func(x2, *popt)
            
            fitted_peaks.append([x, y, [x2, y2]])
            
        return fitted_peaks
    
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]

def zermuller_find_roots(f, xinit, ztol= 1.0e-5, ftol=1.0e-5, maxiter=1000, wantreal=False, nroots=1):
    
    def deflate(f,z, kroots, roots):
        """
        Arguments 
          f                 Input: complex<double> function whose root is desired
          z                 Input: test root
          kroots            Input: number of roots found so far
          roots             Input/Output: saved array of roots
     
        Return value
          Deflated value of f at z.
     
        Description
          This routine is local to zermuller.
          Basically, it divides the complex<double> function f by the product
                       (z - root[0]) ... (z - root[kroots - 1]).
          where root[0] is the first root, root[1] is the second root, ... etc.
        """
        undeflate = t = f(z)
        nroots = len(roots)
        for i in range(nroots):
            denom = z - roots[i]
            while (abs(denom) < 1e-8):# avoid division by a small number #
                denom += 1.0e-8
            t = t / denom
        return t, undeflate
    
    nmaxiter = 0
    retflag  = 0
    roots = []
    for j in range(nroots):
        #print "j=",  j
        x1  = xinit
        x0  = x1 - 1.0
        x2  = x1 + 1.0
 
        f0, undeflate  = deflate(f, x0, j, roots)
        f1, undeflate  = deflate(f, x1, j, roots)
        f2, undeflate  = deflate(f, x2, j, roots)
 
        h21 = x2 - x1
        h10 = x1 - x0
        f21 = (f2 - f1) / h21
        f10 = (f1 - f0) / h10
 
        for i in range(maxiter):
            #print "iter", i
            f210 = (f21 - f10) / (h21+h10) 
            b    = f21 + h21 * f210
            t    = b*b- 4.0 * f2 * f210
 
            if (wantreal) :         # force real roots ? #
               if (real(t) < 0.0):
                   t = 0.0
               else :
                   t =  real(t)
 
            Q = sqrt(t)
            D = b + Q
            E = b - Q
 
            if (abs(D) < abs(E)) :
                D = E
 
 
            if (abs(D) <= ztol) :      # D is nearly zero ? #
                xm = 2 * x2 - x1
                hm = xm - x2
            else :
                hm = -2.0 * f2 / D
                xm = x2 + hm
 
 
            # compute deflated value of function at xm.  #
            fm, undeflate = deflate(f, xm, j, roots)
 
 
            # Divergence control #
            absfm = abs(fm)
            absf2 = 100. * abs(f2)
            # Note: Originally this was a while() block but it
            #       causes eternal cycling for some polynomials.
            #       Hence, adjustment is only done once in our version.
            if (absf2 > ztol and absfm >= absf2) :
                hm    = hm * 0.5
                xm    = x2 + hm
                fm    = f(xm)
                absfm = abs(fm)
 
 
            # function or root tolerance using original function
            if (abs(undeflate) <= ftol or abs(hm) <= ztol) :
                if (i > nmaxiter) :
                    nmaxiter = i
                    retflag = 0
                    break
 
            # Update the variables #
            x0  = x1
            x1  = x2
            x2  = xm
 
            f0  = f1
            f1  = f2
            f2  = fm
 
            h10 = h21
            h21 = hm
            f10 = f21
            f21 = (f2 - f1) / h21
 
 
        if (i > maxiter) :
                nmaxiter = i
                retflag  = 2
                break
 
        xinit = xm
        #print "a root is ", xinit
        roots.append(xinit)
 
        # initial estimate should be far enough from latest root.
        xinit    = xinit + 0.85
 
    maxiter = nmaxiter
    return roots
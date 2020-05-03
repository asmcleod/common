###############################################
#ASM# module "numerics" from package "common" #ASM#
###############################################

"""
These are miscellaneous functions useful for a variety of python purposes.
"""

import copy
import types
import re
from .log import Logger
from . import misc
from . import baseclasses
from .baseclasses import expand_limits,get_array_slice
try: import numpy
except ImportError:
    Logger.raiseException('The <numpy> module is required for this suite, but it was not found in the python path.')
    raise ImportError

__module_name__=__name__

deg2rad=numpy.pi/180
number_types=(int,int,float,numpy.int32,numpy.int64,numpy.float,numpy.float32,numpy.float64,numpy.complex)
if 'float128' in dir(numpy): number_types+=(numpy.float128,)
if 'complex128' in dir(numpy): number_types+=(numpy.complex128,)

def sequence_cmp(a,b):
    
    if hasattr(a,'__len__') and not hasattr(b,'__len__'): return -1
    elif hasattr(b,'__len__') and not hasattr(a,'__len__'): return 1
    elif not hasattr(a,'__len__') and not hasattr(b,'__len__'): return cmp(a,b)
    else:
        lengths=(len(a),len(b))
        for i in range(max(lengths)):
            try: a_value=a[i]
            except IndexError: return 1
            try: b_value=b[i]
            except IndexError: return -1
            comparison=sequence_cmp(a_value,b_value)
            if comparison!=0: return comparison
        return 0

def factorial(n):
    """
    This function calculates the factorial of a number.
    """
    sum = 1.0
    for m in range(1, int(n)+1):
        sum = float(m)*sum
    return sum

def bin_array(arr,bins=200,weights=None,density=False):
    
    arr=arr.flatten()
    if weights is not None: weights=weights.flatten()
    h=numpy.histogram(arr,bins=bins,density=density,weights=weights)
    h[0][numpy.isnan(h[0])]=0
    h=baseclasses.ArrayWithAxes(h[0],axes=[h[1][:-1] \
                                           +numpy.diff(h[1])]).sort_by_axes()
    
    return h

def shear_array(a, strength=1, shear_axis=1, increase_axis=None,interpolate=False):
    
    Logger.raiseException('The dimension of `a` must be at least 2.',
                          unless=(a.ndim>=2), exception=TypeError)
    Logger.raiseException('`shear_axis` and `increase_axis` must be distinct.',
                          unless=(shear_axis!=increase_axis), exception=ValueError)
    
    shear_axis%=a.ndim
    if increase_axis is None:
        axes=list(range(a.ndim))
        axes.remove(shear_axis)
        increase_axis=axes[0]
    
    if interpolate:
        Logger.raiseException('The `interpolate` option is only enabled for \
                               an array `a` of dimension 2.',\
                               unless=a.ndim==2, exception=IndexError)
        arr=baseclasses.AWA(a)
        xs=arr.axes[shear_axis]
        get_ind=lambda y: (y,slice(None)) if shear_axis==1 else (slice(None),y)
        res = [arr[get_ind(y)].interpolate_axis((xs-strength*y)%arr.shape[shear_axis],\
                                       axis=0, extrapolate=True,\
                                       bounds_error=False) \
               for y in range(a.shape[increase_axis])]
        res=numpy.array(res)
        if shear_axis==0: res=res.transpose()
        
    else:
        indices = numpy.indices(a.shape)
        indices[shear_axis] -= strength * indices[increase_axis]
        indices[shear_axis] %= a.shape[shear_axis]
        res = a[tuple(indices)]
        
    return res

def levi_cevita(*indices):

    result = 1
    for i, x1 in enumerate(indices):
        for j in range(i+1, len(indices)):
            x2 = indices[j]
            if x1 > x2:
                result = -result
            elif x1 == x2:
                return 0
    return result


def rotation_matrix(angle,about_axis=None,optimize=False):
    """
    Returns the rotation matrix corresponding to rotation by
    angle *angle* (in degrees) about some axis vector
    *about_vector* (DEFAULT: Z-axis).
    """
    
    from numpy.linalg import norm
    
    ###Set conversion to radians###
    deg2rad=numpy.pi/180.
    angle=deg2rad*angle
    
    ###Set default about_axis###
    if about_axis is None: about_axis=[0,0,1]
    
    ###Or make sure provided one is a unit vector###
    else:
        about_axis=numpy.array(about_axis).astype(complex).flatten()
        about_axis/=norm(about_axis)
    
    #####Construct a rotation matrix = P+(I-P)*cos(theta)+Q*sin(theta)#####
    v=about_axis
    TensorProd=numpy.transpose(numpy.matrix(v))*numpy.matrix(v)
    CrossProd=numpy.matrix([[0,-v[2],v[1]],\
                            [v[2],0,-v[0]],\
                            [-v[1],v[0],0]])
    I=numpy.matrix(numpy.eye(3))
    
    return numpy.cos(angle)*I+numpy.sin(angle)*CrossProd+(1-numpy.cos(angle))*TensorProd

def rotate_vector(vector,angle,about_axis=None,\
                  optimize=False):
    """
    Returns *vector* rotated by angle *angle* (in degrees)
    about *about_axis* (DEFAULT: [0 0 1]).
    """
    
    vector=numpy.asarray(vector)
    vector=numpy.matrix(vector).T
    
    #Pad to three dimensions if necessary#
    padded=False
    if vector.shape==(2,1):
        vector=numpy.matrix(list(numpy.array(vector).squeeze())+[0]).T
        padded=True
    
    assert vector.shape==(3,1),\
        'Input `vector` must be a 2- or 3-coordinate iterable.'
    
    #vector=vector.T
    rotated_vector=(rotation_matrix(angle,about_axis,\
                                    optimize=optimize)*vector)
    
    result=numpy.array(rotated_vector).squeeze()
    if padded: result=result[:2]
    
    return result

def cross_product_matrix(vec):
    
    from numpy.linalg import norm
    
    vec=numpy.array(vec)
    vecNorm=numpy.float(norm(vec))
    vecHat=vec/vecNorm
    
    I=numpy.matrix(numpy.eye(3))
    projector=numpy.transpose(numpy.matrix(vecHat))*numpy.matrix(vecHat)
    R=rotation_matrix(90,about_axis=vecHat)
    
    return vecNorm*R*(I-projector)
    

def change_basis_matrix(from_basis=numpy.eye(3),\
                        to_basis=numpy.eye(3),\
                        optimize=False):
    
    if not optimize:
        try:
            ##Check that we have array type##
            from_basis=numpy.matrix(from_basis).squeeze()
            to_basis=numpy.matrix(to_basis).squeeze()
            
            ##Check that we have square matrices (arrays)##
            if not ((from_basis.ndim is 2) and (from_basis.shape[-1] is from_basis.shape[0])) \
               and not ((to_basis.ndim is 2) and (to_basis.shape[-1] is to_basis.shape[0])): raise ValueError
               
            ##Check that we have non-singular matrices (valid bases)##
            from_basis.I; to_basis.I
        
        except: Logger.raiseException('*from_basis* and *to_basis* should both be valid '+\
                                      'NxN matrix-representable bases whose rows are '+\
                                      'linearly independent basis vectors.', exception=numpy.LinAlgError)
        
    ##Convert to conventional representation##
    #Columns are basis vectors#
    from_basis=from_basis.T; to_basis=to_basis.T
    
    return to_basis.I*from_basis

def change_vector_basis(vector,from_basis=numpy.eye(3),\
                               to_basis=numpy.eye(3),\
                               optimize=False):
    """
    Returns *vector* rotated by angle *angle* (in degrees)
    about *about_axis* (DEFAULT: [0 0 1]).
    """
    
    if not optimize:
        try:
            #Make into a vector#
            vector=numpy.matrix(vector).squeeze()
            
            #Check shape#
            if not vector.ndim is 1 and vector.shape[0] in (2,3):
                raise ValueError
            
        except: Logger.raiseException('*vector* must be a matrix-representable list of 2 or 3 coordinates.')
    
    #Pad to three dimensions if necessary#
    if vector.shape[0] is 2: vector=numpy.matrix(list(vector)+[0])
    vector=vector.T
    result=(change_basis_matrix(from_basis,to_basis,optimize=optimize)*vector).T
    
    return numpy.array(result).squeeze()

def grid_axes(*axes):
    
    array_axes=[]
    for axis in axes:
        axis=numpy.array(axis)
        Logger.raiseException('Each provided axis must have a one dimensional array shape.',\
                         unless=((axis.ndim==1) and (0 not in axis.shape)),\
                         exception=ValueError)
        array_axes.append(axis)
    
    index_grids=list(numpy.mgrid.__getitem__([slice(0,len(axis)) for axis in array_axes]))
    
    return [array_axes[i][index_grids[i]] for i in range(len(array_axes))]

#TODO: remove legacy naming
expand_axes=grid_axes

def array_from_points(points,values,dtype=None,fill=numpy.nan):
    
    #####Check values#####
    error_msg='Both *points* and *values* must be iterables of the same (non-zero) length, '+\
              'the first a list of (tuple) points and the latter a list of corresponding values.'
    Logger.raiseException(error_msg,unless=(hasattr(points,'__len__') and hasattr(values,'__len__')),
                     exception=TypeError)
    Logger.raiseException(error_msg,unless=(len(points)==len(values) and len(points)>0),\
                     exception=IndexError)
    points=list(points)
    for i in range(len(points)):
        ####Give each point a length equal to the array shape, even if 1-D array####
        if not hasattr(points[i],'__len__'): points[i]=[points[i]]
        Logger.raiseException('Each point in *points* must be an (tuple) iterable of uniform length (i.e., each describing an array of a consistent shape).',\
                         unless=(len(points[i])==len(points[0])),\
                         exception=IndexError)
        
    if dtype is None: dtype=type(values[0])
    
    #####Get list of indices in all axes corresponding to ordered values#####
    ###Unzip into format [[axis1 values], [axis2 values], ...]###
    unsorted_axes=list(zip(*points))
    index_lists=[]
    sorted_axes=[]
    
    ####Iterate through each axis to get lists of indices corresponding to each axis####
    for i in range(len(unsorted_axes)):
        unsorted_axis_values=list(unsorted_axes[i])
        sorted_axis_values=copy.copy(unsorted_axis_values)
        sorted_axis_values.sort()
        sorted_axes.append(sorted_axis_values)
        
        index_list=[0]*len(sorted_axis_values)
        j=0
        index=0
        while j<len(sorted_axis_values):
            axis_value=sorted_axis_values[j]
            positions=misc.all_indices(unsorted_axis_values,axis_value)
            
            ###Fill in *index_list* at positions where *axis_value* occurs with the index value###
            for position in positions:
                index_list[position]=index
            
            ###Skip ahead so we don't double up on axis values we've already counted###
            index+=1
            j+=len(positions)
        
        index_lists.append(index_list)
    
    ###Zip back up into same format as *points*###
    index_points=list(zip(*index_lists))
        
    ####Remove non-unique values from *sorted_axes*####
    axes=[]
    for i in range(len(sorted_axes)):
        sorted_axis=sorted_axes[i]
        axis=[]
        for j in range(len(sorted_axis)):
            value=sorted_axis[j]
            if value not in axis: axis.append(value)
        axes.append(numpy.array(axis))
    
    #####Produce array and fill it in by indices#####
    ###Get array shape and make empty array###
    shape=[]
    for i in range(len(axes)):
        axis=axes[i]
        shape.append(len(axis))
    shape=tuple(shape)
    output_array=numpy.zeros(shape,dtype=dtype)
    output_array[:]=fill #fill with *fill*
    
    ###Fill in by indices###
    for i in range(len(index_points)):
        index=index_points[i]
        value=values[i]
        output_array[index]=value
        
    return baseclasses.ArrayWithAxes(output_array,axes=tuple(axes))

def operate_on_entries(func,*inputs,**kwargs):
    
    ##Extract special keyword arguments##
    exkwargs=misc.extract_kwargs(kwargs,out=None,\
                                        dtype=None)
    out=exkwargs['out']; dtype=exkwargs['dtype']
    
    ##Check function##
    Logger.raiseException('*func* must be a callable object operating on %i input arguments.'%len(inputs),\
                          unless=hasattr(func,'__call__'), exception=TypeError)
    
    ##Check input (and output if provided)##
    inputs=list(inputs)
    ##Verify that we have arrays##
    for i in range(len(inputs)):
        try: inputs[i]=numpy.array(inputs[i])
        except: Logger.raiseException('Each input to *func* must be castable as an array instance.',\
                                       exception=TypeError)
    ##Broadcast inputs to list of tuples##
    try: 
        if len(inputs)>=2:
            broadcast_inputs=numpy.broadcast(*inputs)
            shape=broadcast_inputs.shape
            linear_inputs=list(broadcast_inputs)
        else:
            shape=inputs[0].shape
            linear_inputs=list(zip(inputs[0].ravel()))
    except ValueError: Logger.raiseException('Each input to *func* must be of consistent shape '+\
                                             '(subject to array broadcasting rules).', exception=ValueError)
        
    ##Verify they have the same shape##
    if out is None:
        if dtype is None: dtype=object
        out=numpy.ndarray(shape,dtype=dtype)
    else:
        Logger.raiseException('*out* must be castable as an array instance with shape %s.'%repr(shape),\
                              unless=(isinstance(out,numpy.ndarray) and (out.shape==shape)),\
                              exception=IndexError)
        if dtype!=None: out=out.astype(dtype)
    
    ###Use operator on broadcast inputs###
    from numpy.lib.index_tricks import unravel_index
    linear_indices=range(numpy.product(shape)) #a single line of incrementing integers
    #We use linear iteration approach because it is (apparantly) the only reliable way
    #to allocate memory for each entry in sequence##
    for linear_index in linear_indices:
        indices=unravel_index(linear_index,shape)
        out[indices]=func(*linear_inputs[linear_index],**kwargs)
    
    return out

def expanding_resize(arr,desired_shape,append_dim=True):
    ####This function exists because I hate the default behavior of numpy.resize and array.resize, this resize function doesn't mix BETWEEN dimensions####
    
    #####Check inputs#####
    arr=numpy.array(arr)
    Logger.raiseException('*desired_shape* must be iterable.',\
                     unless=hasattr(desired_shape,'__len__'),\
                     exception=TypeError)
    Logger.raiseException('*desired_shape* cannot be an empty shape.',\
                     unless=(len(desired_shape)>0), exception=ValueError)
    arr=numpy.array(copy.copy(arr))
    Logger.raiseException('All dimensions of *arr* must be non-zero.',\
                     unless=(0 not in arr.shape), exception=ValueError)
    for item in desired_shape:
        message='Each element of *desired_shape* must be a positive integer.'
        Logger.raiseException(message, unless=(type(item)==int), exception=TypeError)
        Logger.raiseException(message, unless=(item>0), exception=ValueError)
    
    ####If we need to append_dim dimensions####
    ndim=len(desired_shape)
    while ndim>len(arr.shape):
        if append_dim: new_shape=arr.shape+(1,)
        else: new_shape=(1,)+arr.shape
        arr=numpy.reshape(arr,new_shape)
    ####if we need to remove dimensions####
    while ndim<len(arr.shape):
        if append_dim: drop_dim=-1
        else: drop_dim=0
        slicer=[None]*len(arr.shape); slicer[drop_dim]=0 #Retain first element in dimension to drop
        arr=get_array_slice(arr,slicer) #now the final dimension is safe to remove
        arr=numpy.reshape(arr,arr.shape[:drop_dim])
    
    for i in range(ndim):
        input_shape=arr.shape
        
        size=input_shape[i]
        desired_size=int(desired_shape[i])
        ndiff=desired_size-size
        
        ###If there's no difference, do nothing###
        if ndiff==0: continue
        
        ###If desired size is larger, we expand on either side###
        elif ndiff>0:
            bottom_slicer=[None]*ndim; bottom_slicer[i]=0
            bottom_edge=get_array_slice(arr,bottom_slicer)
            top_slicer=[None]*ndim; top_slicer[i]=input_shape[i]-1
            top_edge=get_array_slice(arr,top_slicer)
            
            ##Iteratively concatenate D=N-1 slices to either side of this axis to fill up to proper size##
            for j in range(ndiff):
                #Decide whether to concatenate at top or bottom of axis in this iteration#
                if j%2==0: to_concatenate=[bottom_edge,arr]
                else: to_concatenate=[arr,top_edge]
                arr=numpy.concatenate(to_concatenate,axis=i)
                
        ###If desired size is smaller, take interior slice###
        elif ndiff<0:
            bottom_slicer=[None]*ndim; bottom_slicer[i]=[1,None]
            top_slicer=[None]*ndim; top_slicer[i]=[None,-1]
            slicers=[bottom_slicer,top_slicer]
            
            ##Iteratively sub-select array, shaving off top and bottom slices##
            for j in range(numpy.abs(ndiff)):
                slicer=slicers[j%2]
                arr=get_array_slice(arr,slicer)
            
    return arr

def interpolating_resize(arr,desired_shape):
    ####This function exists because I hate the default behavior of numpy.resize and array.resize, this resize function doesn't mix BETWEEN dimensions####

    try: from scipy.interpolate import interp1d
    except ImportError:
        ##Replace interpolator with a home-cooked version?##
        ##Not yet.##
        Logger.raiseException('SciPy module unavailable!  Interpolation cannot be used, '+\
                         'an expanding resize will be used instead.', exception=False)
        return expanding_resize(arr,desired_shape)
        
    #####Check inputs#####
    Logger.raiseException('*arr* and *desired_shape* must both be iterables.',\
                     unless=(hasattr(arr,'__len__') and hasattr(desired_shape,'__len__')),\
                     exception=TypeError)
    Logger.raiseException('*desired_shape* cannot be an empty shape.',\
                     unless=(len(desired_shape)>0), exception=ValueError)
    arr=numpy.array(copy.copy(arr))
    Logger.raiseException('All dimensions of *arr* must be non-zero.',\
                     unless=(0 not in arr.shape), exception=ValueError)
    for item in desired_shape:
        message='Each element of *desired_shape* must be a positive integer.'
        Logger.raiseException(message, unless=(type(item)==int), exception=TypeError)
        Logger.raiseException(message, unless=(item>0), exception=ValueError)
    
    ####If we need to append dimensions####
    ndim=len(desired_shape)
    while ndim>len(arr.shape): arr=numpy.reshape(arr,arr.shape+(1,))
    ####if we need to remove dimensions####
    while ndim<len(arr.shape):
        slicer=[None]*len(arr.shape); slicer[-1]=0 #Take first element in last dimension
        arr=get_array_slice(arr,slicer) #now the final dimension is safe to remove
        arr=numpy.reshape(arr,arr.shape[:-1])
    
    for i in range(ndim):
        
        input_shape=arr.shape
        size=input_shape[i]
        desired_size=int(desired_shape[i])
        ndiff=desired_size-size
        
        ###If there's no difference, do nothing###
        if ndiff==0: continue
        
        ###If desired size is larger, we have to watch out if we can't interpolate###
        #Instead we just replicate with concatenation#
        elif size==1 and ndiff>0:
            
            slicer=[None]*ndim; slicer[i]=0
            slice=get_array_slice(arr,slicer)
            for j in range(ndiff): arr=numpy.concatenate((arr,slice),axis=i)
            
        ###Otherwise we're free to use interpolation###
        else:
            x=numpy.linspace(0,size,size)/float(size)
            x_new=numpy.linspace(0,desired_size,desired_size)/float(desired_size)
            #Swap axis to the end axis - the numpy folks need to fix their code for *interp1d*,
            #the returned array is shaped incorrectly if *axis=0* and *ndim>2*, 
            #only thing that works for sure is *axis=-1*.
            arr=arr.swapaxes(i,-1)
            interpolator=interp1d(x,arr,axis=-1)
            arr=interpolator(x_new)
            arr=arr.swapaxes(i,-1)
        
    return arr
    
def value_to_index(value,x):

    #####Check inputs#####
    try:
        x=numpy.array(x)
        if x.ndim>1: raise TypeError
    except TypeError:
        Logger.raiseException('Axis values list *x* must be a linear numeric array',\
                         exception=TypeError)
    
    #####Find most appropriate index#####
    return numpy.abs(x-value).argmin()

def slice_array_by_value(value,x,arr,axis=0,squeeze=True,get_closest=False):
    ##THis function is largely deprecated in favor of coordinate slicing on *ArrayWithAxes* instances#
    
    #####Find most appropriate index#####
    closest_index=value_to_index(value,x)
    axis=axis%arr.ndim
    limits=[None]*axis+[closest_index]+[None]*(arr.ndim-axis-1)
    
    #####Slice the array#####
    sliced_arr=get_array_slice(arr,limits,squeeze=squeeze)
    ##Reduce to number if it has no shape##
    if len(sliced_arr.shape)==0: return sliced_arr.tolist()
    
    if get_closest==True: return (x[closest_index],sliced_arr)
    else: return sliced_arr
    
#@TODO: remove this legacy re-name
index_array_by_value=slice_array_by_value

def remove_array_poles(arr,window=10,axis=-1):
    
    ndim=arr.ndim
    axis=axis%ndim
    newarr_slices=[]
    
    #Fill in first element
    arr_first=get_array_slice(arr,(0,1),axis=axis,squeeze=False)
    newarr_slices.append(arr_first)
    
    #Smooth everything else by windows in between
    arr_previous=arr_first
    for i in range(arr.shape[axis]-2):
        arr_windowed=get_array_slice(arr,(1+i,1+i+window),axis=axis,squeeze=False)
        
        #Compute difference along axis in window
        diff=(arr_previous-arr_windowed)**2
        for j in range(ndim):
            compress_axis=ndim-j-1
            if compress_axis==axis: continue
            diff=numpy.sum(diff,axis=compress_axis)
        
        #Find index in window of minimal difference
        index=numpy.arange(len(diff))[diff==diff.min()]
        index=index[0] #pick soonest index of minimum difference
        
        #Populate new array with element from *arr* at this window index
        arr_next=get_array_slice(arr_windowed,(index,index+1),axis=axis,squeeze=False)
        newarr_slices.append(arr_next)
        
        #Call this slice now the "previous" slice in next iteration, use it for diff reference
        arr_previous=arr_next
        
    #Fill in last element
    arr_previous=get_array_slice(arr,(arr.shape[axis]-1,arr.shape[axis]),axis=axis,squeeze=False)
    newarr_slices.append(arr_previous)
        
    #Make new array
    newarr=numpy.concatenate(newarr_slices,axis=axis)
    if isinstance(arr,baseclasses.ArrayWithAxes):
        newarr=baseclasses.ArrayWithAxes(newarr)
        newarr.adopt_axes(arr)
        newarr=arr.__class__(newarr)
        
    return newarr

def broadcast_items(*items):
    
    array_items=[numpy.array(item) for item in items]
        
    ndim_front=0
    ndim_behind=numpy.sum([array_item.ndim for array_item in array_items])
    
    broadcasted_arrays=[]
    for array_item in array_items:
        
        #If item had no shape (e.g. scalar) don't broadcast
        if array_item.ndim==0:
            broadcasted_arrays.append(array_item)
            continue
        
        ndim_behind-=array_item.ndim
        broadcasted_shape=(1,)*ndim_front\
                            +array_item.shape\
                            +(1,)*ndim_behind
        ndim_front+=array_item.ndim
        
        broadcasted=array_item.reshape(broadcasted_shape)
        broadcasted_arrays.append(broadcasted)
        
    #For each unsized array item, replace with original item#
    #(We don't want unsized arrays in the returned value)#
    for i in range(len(broadcasted_arrays)):
        if not broadcasted_arrays[i].ndim>=1:
            broadcasted_arrays[i]=items[i]
        
    return broadcasted_arrays

def differentiate(y,x=None,axis=0,order=1,interpolation='linear'):
    """
    Differentiate an array *y* with respect to array *x*
    *order* times.
    
    OUTPUTS:
        dy/dx(x')
    Here *dy/dx* is the computed derivative, and *x'* values
    correspond to axis coordinates at which the derivative
    is computed.  If interpolation is used, the primed
    coordinates are identical to the input axis coordinates,
    otherwise these are central-difference coordinates.
    
    INPUTS:
        *x:  values for the axis over which the derivative
            should be computed.  The length of *x* must be the
            same as the dimension of *y* along *axis*.
            
        *axis:  the array axis over which to differentiate *y*.
            
        *order:  an integer specifying the order of the
            derivative to be taken in each component.
            DEFAULT: 1 (first derivative)
            
        *interpolation:  specify the interpolation to be used
            when calculating derivatives.  Must be one of
            True, False, 'linear', 'quartic', 'cubic' or 'wrap'.
            If the chosen option is 'wrap', values of *y* are 
            assumed periodic along *axis*.
            If interpolation is used, the returned axis
            coordinates are identical to the input axis
            coordinates.
            DEFAULT: True --> linear interpolation
    """
    
    ####Establish interpolation####
    if interpolation not in [False,None]:
        from scipy.interpolate import interp1d as interpolator_object
       
    #####Check inputs#####
    ####Check x and y####
    Logger.raiseException('*y* and *x* must both be iterables.',\
                     unless=(hasattr(y,'__len__') and (x is None or hasattr(x,'__len__'))),\
                     exception=TypeError)
    assert isinstance(axis,int) and isinstance(order,int)
    if not isinstance(y,numpy.ndarray): y=numpy.array(y)
    Logger.raiseException('*y* cannot be an empty array.', unless=(0 not in y.shape), exception=ValueError)
    axis=axis%y.ndim #make axis positive    
    
    ####Verify that x and y are broadcast-able####
    if x is None:
        if isinstance(y,baseclasses.ArrayWithAxes): x=y.axes[axis]
        else: x=numpy.arange(y.shape[axis])
    else:
        x=numpy.array(x)
        Logger.raiseException('*x* must be 1-dimensional and of the same length as the *axis* dimension of *y*.',\
                         unless=(len(x)==y.shape[axis] and x.ndim==1),\
                         exception=IndexError)
    
    ##If interpolation is not None, then we are asking for interpolation##
    if interpolation is True: interpolation='linear'
    if not interpolation or interpolation=='wrap':
        wrap=True; interpolation_chosen=False
    else: wrap=False; interpolation_chosen=True
    
    #####Differentiate#####
    global diff_x_1d
    for i in range(order):
        
        ###Step 1. ###
        #Differentiate x (Only needed once if we're interpolating back each time)
        if interpolation_chosen==False or (interpolation_chosen==True and i==0):
            
            #Wrap differentiation maintains same number of x points
            if wrap:
                diff_x_1d=numpy.roll(x,1,axis=0)-x
                x_reduced=x+diff_x_1d/2.
            #Differentiating reduces by 1 data point#
            else:
                diff_x_1d=numpy.diff(x)
                x_reduced=x[:-1]+diff_x_1d/2.
            
            ##Get *x* array ready for broadcasting correctly when doing *diff_y/diff_x*##
            new_shape=[1 for dim in range(y.ndim)]
            new_shape[axis]=len(diff_x_1d)
            diff_x=numpy.resize(diff_x_1d,new_shape)
            
            if interpolation_chosen==True:
                #First make sure x is in uniformly increasing order#
                Logger.raiseException('If interpolating to original values, *x* must be in uniformly increasing order.',\
                                 unless=(diff_x_1d>0).all(), exception=ValueError)
                
                #linearly expand x by 1 point in both directions to permit later interpolation at edges#
                #this value only used in interpolation function
                #Expand analytically if x_reduced has length
                x_enlarged=numpy.concatenate( ([x[0]-diff_x_1d[0]/2.],\
                                                    x_reduced,\
                                                    [x[-1]+diff_x_1d[-1]/2.]) )
        
        ###Step 2. ###
        #Differentiate y
        #Differentiating reduces by 1 data point
        if wrap: diff_y=numpy.roll(y,1,axis=axis)-y
        else: diff_y=numpy.diff(y,axis=axis)
        dydx_reduced=diff_y/diff_x #broadcasting on *diff_x* works as needed
        
        if interpolation_chosen==True:
            
            ###Step 3: ###
            #linearly expand derivative *dydx_reduced* by 1 point in both directions along *axis* to permit later interpolation at edges#
            #If there literally is no derivative, make one up (zeros)
            if len(dydx_reduced)==0:
                shape=list(y.shape)
                shape[axis]=2
                dydx_enlarged=numpy.zeros(shape)
            
            else:
                edge_slicer_bottom=[None for dim in range(y.ndim)]
                edge_slicer_top=copy.copy(edge_slicer_bottom)
                edge_slicer_bottom[axis]=0; edge_slicer_top[axis]=-1
                #this value only used in interpolation function
                dydx_enlarged=numpy.concatenate((get_array_slice(dydx_reduced,edge_slicer_bottom,squeeze=False),\
                                                 dydx_reduced,\
                                                 get_array_slice(dydx_reduced,edge_slicer_top,squeeze=False)),\
                                                 axis=axis)
            
            ###Step 4. ###
            #interpolate back to original x values#
            try:
                dydx_enlarged=numpy.swapaxes(dydx_enlarged,axis,-1) #We swap axes because interp1d has a bug for axis!=-1
                interpolator=interpolator_object(x_enlarged,dydx_enlarged,kind=interpolation,axis=-1,\
                                                 copy=False,bounds_error=True)
                dydx=interpolator(x)
                dydx=numpy.swapaxes(dydx,axis,-1)
            except: 
                Logger.logNamespace(locals())
                print('X:',x_enlarged)
                print('dY/dX:',dydx_enlarged)
                raise
            #*x* unchanged, likewise *diff_x*
            
        ###If not interpolating, keep x, y reduced###
        else: 
            x=x_reduced
            dydx=dydx_reduced
            
    if isinstance(y,baseclasses.AWA): axes=y.axes; axis_names=y.axis_names
    else: axes=[None]*y.ndim; axis_names=None
    axes[axis]=x

    return baseclasses.AWA(dydx,axes=axes,axis_names=axis_names)

def gradient(y,axes=None,order=1,interpolation='linear'):
    """
    Compute the gradient of an array *y* with respect
    to each of its axes {*x1,x2,...*}, and provide the
    axes values at which the elements of each component
    of the gradient is computed.
    
    OUTPUTS:
        (g1(x'),g2(x'),...)
    Here *g1*, etc. indicates the first, second, and
    each component of the gradient.  Primed *x'* return values
    correspond to axis coordinates at which each component
    is computed.  If interpolation is used, the primed
    coordinates are identical to the input axis coordinates,
    otherwise central difference coordinates are used.
    
    INPUTS:
        *axes:  (xs1, xs2,...)
            A 1-D axis iterable should be provided for
            each dimension of the array *y*.  Alternatively,
            omit {*x1,x2,...*} and the index values at each
            point in *y* will be used as appropriate.
            
        *order:  an integer specifying the order of the
            derivative to be taken in each component.
            DEFAULT: 1 (first derivative)
            
        *interpolation:  specify the interpolation to be used
            when calculating derivatives.  Must be one of
            True, False, 'linear', 'quartic', 'cubic'.  If
            interpolation is used, the returned axis
            coordinates are identical to the input axis
            coordinates.
            DEFAULT: True (linear interpolation)
    """
    
    if not isinstance(y,numpy.ndarray): y=numpy.array(y)
    
    ##Determine axes
    if axes is None: 
        if isinstance(y,baseclasses.ArrayWithAxes): axes=y.axes
        else: axes=[None]*y.ndim
    ##If *y* has 1 dimension and *axes* is not a list, interpret as single axis array#
    elif y.ndim==1 and isinstance(axes,numpy.ndarray):
        if axes.ndim==1: axes=[axes]
    ##Check each axis##
    for dim in range(y.ndim):
        
        if dim>=len(axes): axes.append(numpy.arange(y.shape[dim])); continue
        elif axes[dim] is None: axes[dim]=numpy.arange(y.shape[dim]); continue
        
        try: axes[dim]=numpy.array(axes[dim])
        except TypeError:
            Logger.raiseException('The axis provided for dimension %s of *y* cannot '%dim+\
                             'be cast as an array.',exception=TypeError)
        
        Logger.raiseException('The axis provided for dimension %s of *y* must be '%dim+\
                         'one-dimensional with length %s.'%y.shape[dim],\
                         unless=(axes[dim].shape==(y.shape[dim],)),\
                         exception=ValueError)
        
    #####Get gradient#####
    grad=[]
    for i in range(y.ndim):
        derivative=differentiate(y,x=axes[i],axis=i,order=order,interpolation=interpolation)
        grad.append(derivative)
        
    return tuple(grad)

def zeros(y,axes=None):
    """
    Compute the position of the zeros of an array. Zeros
    are defined as coordinate positions where:
        1) The array value is identically zero, or
        2) array values cross the zero line (the position
           is inferred through linear interpolation)
    
    INPUTS:
        *Non-keywords:  non-keyword values should be
            passed in the following way:
                *zeros(x1,x2,...,y)*
            A 1-D axis iterable should be provided for
            each dimension of the array *y*.  Alternatively,
            omit {*x1,x2,...*} and the index values at each
            point in *y* will be used as appropriate.
    
    OUTPUTS:
        A list of point coordinates of the form:
            ((x1,y1,...), (x2,y2,...), ...)
        Each coordinate in the list has as many elements
        as the number of dimensions in the input array *y*.
    """

    #####Check all inputs#####
    if not isinstance(y,numpy.ndarray): y=numpy.array(y)
    
    ##Determine axes
    if axes is None: 
        if isinstance(y,baseclasses.ArrayWithAxes): axes=y.axes
        else: axes=[None]*y.ndim
    ##If *y* has 1 dimension and *axes* is not a list, interpret as single axis array#
    elif y.ndim==1 and isinstance(axes,numpy.ndarray):
        if axes.ndim==1: axes=[axes]
    ##Check each axis##
    for dim in range(y.ndim):
        
        if dim>=len(axes): axes.append(numpy.arange(y.shape[dim])); continue
        elif axes[dim] is None: axes[dim]=numpy.arange(y.shape[dim]); continue
        
        try: axes[dim]=numpy.array(axes[dim])
        except TypeError:
            Logger.raiseException('The axis provided for dimension %s of *y* cannot '%dim+\
                             'be cast as an array.',exception=TypeError)
        
        Logger.raiseException('The axis provided for dimension %s of *y* must be '%dim+\
                         'one-dimensional with length %s.'%y.shape[dim],\
                         unless=(axes[dim].shape==(y.shape[dim],)),\
                         exception=ValueError)
    x_values=axes
    
    ####The calculation that follows uses interpolation, which we can't do if
    #any dimension is trivially of length 1 - so we fudge it to length 2####
    resized_y=copy.copy(y)
    resized_x_values=copy.copy(x_values)
    
    for axis in range(y.ndim):
        if resized_y.shape[axis]==1:
            
            new_shape=list(copy.copy(resized_y.shape))
            new_shape[axis]=2
            resized_y=numpy.resize(resized_y,tuple(new_shape))
            resized_x_values[axis]=[x_values[axis][0]-.5,x_values[axis][0]+.5]
    #Now we're OK for interpolation#
    
    ######1) First, identify where zeros should reside based on values that cross the zero line######
    #####Calculate derivatives at interpolated grid points#####
    #This is necessary to compare "apples to apples", since the only
    #way to identify a zero line crossing is to evaluate derivatives
    #of *sign(y)* at "in-between" points and compare along each axis.
    #However each derivative must be computed at identical points.  
    #So, we compute a new *y* interpolated at every axis EXCEPT the 
    #derivative axis, so that in the end every derivative will end 
    #up being evaluated along an identical grid, good for comparison
    derivatives=[]
    interp_x_values=[]
    for axis in range(y.ndim):
        
        ###Compute interpolated grid###
        #Iterate over interpolation directions
        #and interpolate##
        interp_y=copy.copy(y)
        for axis2 in range(y.ndim):
            
            #Don't interpolate to "in-between" points along derivative axis#
            if axis2==axis: continue
            
            ###Compute interpolated y grid###
            reducer=[None]*interp_y.ndim
            reducer[axis2]=[0,-1] #this will lop off the end point along *axis2*
            dy=numpy.diff(interp_y,axis=axis2)
            interp_y=get_array_slice(interp_y,reducer) #lopped
            interp_y=interp_y+dy/2. #now we extend to midpoint towards the endpoint we lopped off
        
        ###Compute interpolated x axis###
        current_x=copy.copy(resized_x_values[axis])
        interp_x=current_x[0:-1]
        dx=numpy.diff(current_x,axis=0)
        interp_x=interp_x+dx/2.
        
        ###Compute derivative of *sign*###
        #Now we have interpolated y grid for this derivative#
        derivative=differentiate(numpy.sign(interp_y),order=1,interpolation=False,axis=axis)[1]
        derivatives.append(derivative)
        interp_x_values.append(interp_x)
    
    ###Find where sign of 1st deriv changes in *any* dimension###
    #Iterating through dimensions and adding boolean arrays is logical *or* operation#
    for i in range(len(derivatives)): #All derivatives should be of uniform shape, evaluated at the same grid
        if i==0: interp_zeros=(numpy.abs(derivatives[i])>1) #changing from a finite value to 0 is insignificant, we need a *delta* of 2
        else: interp_zeros+=(numpy.abs(derivatives[i])>1)
    
    ###Create array of linear indices###
    from numpy.lib.index_tricks import unravel_index
    all_lin_indices=numpy.arange(numpy.product(interp_zeros.shape)) #a single line of incrementing integers
    all_lin_indices.resize(interp_zeros.shape) #reshaped as an array of memory-indices
    
    ###Get indices###
    lin_indices=all_lin_indices[interp_zeros] #Index by an array of boolean values --> 1-D output
    indices=[]
    coordinates=[]
    
    ####A particular linear index corresponds to a particular point in the array adjacent to where a zero resides####
    for lin_index in lin_indices:
        ###This is the index coordinate for this linear index####
        index_set=unravel_index(lin_index,interp_zeros.shape)
        coordinate_set=[]
        for axis in range(len(index_set)):
            axis_index=index_set[axis]
            coordinate_set.append(interp_x_values[axis][axis_index])
            
        indices.append(tuple(index_set))
        coordinates.append(tuple(coordinate_set))
        
    ######2) Next, let's include the indices where the y-value actually is zero#####
    all_lin_indices=numpy.arange(numpy.product(y.shape)) #a single line of incrementing integers
    all_lin_indices.resize(y.shape) #reshaped as an array of memory-indices
    lin_indices=all_lin_indices[y==0]
    
    ####A particular linear index corresponds to a particular point in the array that is identically zero####
    for lin_index in lin_indices:
        ###This is the index coordinate for this linear index####
        index_set=unravel_index(lin_index,y.shape)
        coordinate_set=[]
        for axis in range(len(index_set)):
            axis_index=index_set[axis]
            coordinate_set.append(x_values[axis][axis_index])
            
        indices.append(tuple(index_set))
        coordinates.append(tuple(coordinate_set))
        
    ####At last, we have everything that could be called a zero point#####
    coordinates.sort(cmp=sequence_cmp) #sort values in some way (well, at least it's sorted by the first entry)
    indices.sort(cmp=sequence_cmp)
    
    return {'indices':tuple(indices),'coordinates':tuple(coordinates)}

def extrema(y,axes=None):
    
    global Dy_signs,DDy_signs,where_maxima,where_minima,where_saddle,inner_index_grids
    
    if axes is None and isinstance(y,baseclasses.ArrayWithAxes):
        y=y.sort_by_axes()
        axes=y.axes
    else:
        try: y=baseclasses.ArrayWithAxes(y,axes=axes)
        except IndexError:
            Logger.raiseException('If provided, `axes` must be an iterable of arrays sized to each dimension of `y`.',\
                                  exception=IndexError)
    
    ### The only significant dimensions are those where size of `y` is more than two ###
    sig_dims=[dim if y.shape[dim]>2 else 0 for dim in range(y.ndim)]
    Logger.raiseException('`y` must have length greater than 2 along at least one axis.',\
                          exception=ValueError,unless=sig_dims)
    
    index_grids=y.get_index_grids(broadcast=True) # Shape is 1 except along associated index
    inner_index_grids=[baseclasses.get_array_slice(index_grids[dim], [1,-1],\
                                                  squeeze=False, axis=dim) for dim in sig_dims]
    
    ### Get array values at interior points (where local extrema are defined) ###
    inner_y=y
    for dim in sig_dims: inner_y=baseclasses.get_array_slice(inner_y, [1,-1],\
                                                             squeeze=False, axis=dim)
    
    
    def collapse_to_interstices(arr,along_axes):
        
        for axis in along_axes:
            arr0=baseclasses.get_array_slice(arr, [0,-1], squeeze=False, axis=axis)
            arr=arr0+numpy.diff(arr,axis=axis)/2.
            
        return arr
    
    
    ### Evaluate first and second derivatives at interstices ###
    Dys=[collapse_to_interstices(numpy.diff(y,axis=i),\
                                along_axes=set(sig_dims)-set((i,))) \
        for i in sig_dims]
                
    Dy_signs=[numpy.where(Dys[dim]>0,1,Dys[dim]) for dim in sig_dims] #Reduce to 1 where positive
    Dy_signs=[numpy.where(Dy_signs[dim]<0,-1,Dy_signs[dim]) for dim in sig_dims] #Reduce to -1 where negative
    
    # We've gone from evaluation at interstices to interior points only #
    # Equals 1 where concavity along that axis is positive, 1 otherwise, zero when derivative has not changed signs
    DDy_signs=numpy.array([collapse_to_interstices(numpy.diff(Dy_signs[dim],axis=dim),\
                                       along_axes=set(sig_dims)-set((dim,))) \
                           for dim in sig_dims])
    
    ## Conditions for extrema ##
    # local maximum requires DDy simultaneously <0 along all axes
    # local minimum requires DDy simultaneously >0 along all axes
    # local saddle requires (otherwise) abs(DDy)>0 along all axes 
    where_maxima=numpy.prod(DDy_signs<0,axis=0).astype(bool)
    where_minima=numpy.prod(DDy_signs>0,axis=0).astype(bool)
    
    #Saddle point condition is based on indeterminacy of Hessian matrix,
    # but its eigenvalues are not an analytic function of the second
    # derivatives for general n-dimensional data.
    
    ## Convert conditions into corresponding index, axis, and array values ##
    d={}
    axes=y.axes
    for name,condition in zip(['maxima','minima'],\
                              [where_maxima,where_minima]):
        
        ## Suppose we don't have condition met anywhere. Empty lists. ##
        if not condition.any():
            d[name]={'values':[],\
                     'indices':[],\
                     'coordinates':[]}
            continue
        
        #Get partial lists of values, only with entries for significant dimensions
        array_values=inner_y[condition]
        partial_index_values=[inner_index_grid[condition] for inner_index_grid in inner_index_grids]
        N=len(partial_index_values[0])
        
        sig_dim=0
        axis_values=[]; index_values=[]
        for dim in range(y.ndim):
            
            #If we want the coordinates along insignificant dimension, use 0,
            # since the location of extrema is not well defined along such an axis
            if dim not in sig_dims:
                index_values.append(N*[0])
            else:
                index_values.append(partial_index_values[sig_dim])
                sig_dim+=1
                
            axis_values.append(axes[dim][index_values[-1]])
        
        # Get tuples and sort them by array value #
        index_tuples=list(zip(*index_values))
        axis_tuples=list(zip(*axis_values))
        sorted_values=sorted(zip(array_values,index_tuples,axis_tuples)) #Comparison performed on first entry
        if name!='minima': sorted_values.reverse() #Want maximum value first, unless we are sorting minima
        array_values,index_tuplesl,axis_tuples=list(zip(*sorted_values)) #Unpack the sorted list
        
        d[name]={'values':array_values,\
                 'indices':index_tuples,\
                 'coordinates':axis_tuples}
        
    return d

"""
def extrema(y,axes=None,**kws):

    from scipy.interpolate import LinearNDInterpolator

    ####Use the same prescription as in *gradient* to expand y and axes####
    if not isinstance(y,numpy.ndarray): y=numpy.array(y)
    
    ##Determine axes
    if axes is None: 
        if isinstance(y,baseclasses.ArrayWithAxes): axes=y.axes
        else: axes=[None]*y.ndim
    ##If *y* has 1 dimension and *axes* is not a list, interpret as single axis array#
    elif y.ndim==1 and isinstance(axes,numpy.ndarray):
        if axes.ndim==1: axes=[axes]
    ##Check each axis##
    for dim in range(y.ndim):
        
        if dim>=len(axes): axes.append(numpy.arange(y.shape[dim])); continue
        elif axes[dim] is None: axes[dim]=numpy.arange(y.shape[dim]); continue
        
        try: axes[dim]=numpy.array(axes[dim])
        except TypeError:
            Logger.raiseException('The axis provided for dimension %s of *y* cannot '%dim+\
                             'be cast as an array.',exception=TypeError)
        
        Logger.raiseException('The axis provided for dimension %s of *y* must be '%dim+\
                         'one-dimensional with length %s.'%y.shape[dim],\
                         unless=(axes[dim].shape==(y.shape[dim],)),\
                         exception=ValueError)
    
    #Turn y into an 
    y=baseclasses.ArrayWithAxes(y,axes=axes)
    x_values=axes

    #####Get gradient#####
    #Gradient, no interpolation REDUCES points by 1 on each axis
    g=gradient(y,axes,order=1,interpolation=None)
    ndim=len(g)
    
    ####Decide on dims to include in analysis####
    if kws.has_key('scan_dims'): 
        scan_dims=kws['scan_dims']
        ###Check input scan_dims###
        misc.check_vars(scan_dims,int)
        if not hasattr(scan_dims,'__len__'):
            try: scan_dims=list(scan_dims)
            except: scan_dims=[scan_dims]
        ###Make all input dims positive###
        for i in range(len(scan_dims)):
            scan_dims[i]=scan_dims[i]%ndim
    else:
        ###Pick default - all axes###
        scan_dims=range(ndim)
    
    dgdx=[]
    new_x_values=axes #x-values will be retained
    for axis in scan_dims:
        
        ###Collect component of gradient for this axis###
        s_reduced=numpy.sign(g[axis])
        x_reduced=new_x_values[axis]
        
        ####Interpolate outwards to two extra points along *axis*####
        ###If gradient for this component is non-existent:###
        if len(x_reduced)==0:
            ##Enlarge *s*
            #Extend outwrds to ENLARGE points by 2 on axis by making new array
            shape=list(s_reduced.shape)
            shape[axis]+=2
            s_enlarged=numpy.zeros(tuple(shape)) #all zeros anyway
            
            ##Enlarge *x*
            #If we have x-values to start with for this axis:
            try: x_enlarged=[x_values[axis][0],x_values[axis][0]+1] #it's meaningless what the second point is, it's interpolated from nothing
            #If we don't:
            except IndexError: x_enlarged=[0,1]
        
        ###Or gradient is full, so compute real values###
        else:
            ##Enlarge *s*##
            #Extend outwards to ENLARGE points by 2 on axis##
            edge_slicer_bottom,edge_slicer_top=[None]*s_reduced.ndim,[None]*s_reduced.ndim
            edge_slicer_bottom[axis]=0; edge_slicer_top[axis]=-1
            s_enlarged=numpy.concatenate((get_array_slice(s_reduced,edge_slicer_bottom,squeeze=False),\
                                          s_reduced,\
                                          get_array_slice(s_reduced,edge_slicer_top,squeeze=False)),\
                                          axis=axis)
            
            ##Enlarge *x*##
            #If x has 2 or more elements:
            try:
                diff_x_bottom=x_reduced[1]-x_reduced[0]
                diff_x_top=x_reduced[-1]-x_reduced[-2]
            #Otherwise, go here:
            except IndexError:
                #Bogus differential values#
                diff_x_bottom,diff_x_top=1,1
                
            x_enlarged=numpy.concatenate(([x_reduced[0]-diff_x_bottom],x_reduced,[x_reduced[-1]+diff_x_top]))
        
        #Compute second derivative discretely at dydx=0 points for each x
        #no interpolation REDUCES points by 1 on each axis
        deriv=differentiate(s_enlarged,x=x_enlarged,axis=axis,interpolation=None)
        dgdx.append(deriv) #must diff. with respect to correct axis, could result in sign flip
        #new dgdx has same shape as original array, same x values
        #extremal points must now ==+/-1 at their locations in this array
        
        #Interpolate y out to `x_enlarged`
        #y=y.interpolate_axis(x_enlarged,axis=axis,bounds_error=False,extrapolate=True)
    
    ###Find where sign of 2nd deriv changes in *all* dimensions###
    #Iteration through dimensions and multiplying boolean arrays is logical *and* operation#
    for i in range(len(dgdx)):
        if i==0:
            minima=(dgdx[i]>0)
            maxima=(dgdx[i]<0)
        else:
            minima*=(dgdx[i]>0)
            maxima*=(dgdx[i]<0)
            
    prod=numpy.product(numpy.array(dgdx),axis=0) #multiply components, collected in first "axis" of pseudo-array
    #*abs(bool-1)* equivalent to element-wise negation
    #multiplication equivalent to element-wise "and" operator
    saddle=(prod!=0)*(numpy.abs(minima-1))*(numpy.abs(maxima-1))
    #convert saddle points to boolean instead of 0,1 values
    saddle=(saddle!=0)
    
    ###Create array of linear indices###
    from numpy.lib.index_tricks import unravel_index
    all_lin_indices=numpy.arange(numpy.product(prod.shape)) #a single line of incrementing integers
    all_lin_indices.resize(prod.shape) #reshaped as an array of memory-indices
    
    ###Get indices/values at each minimum, maximum, saddle###
    all_indices=[]
    all_coordinates=[]
    all_values=[]
    for extrema_type in [minima,maxima,saddle]:
        #Index by an array of boolean values --> 1-D output
        this_type_lin_indices=all_lin_indices[extrema_type]
        this_type_indices=[]
        this_type_coordinates=[]
        this_type_values=[]
        
        ####A particular linear index corresponds to a particular point in the array for this extremum type####
        for this_type_lin_index in this_type_lin_indices:
            ###This is the index coordinate for this linear index####
            this_type_index_set=unravel_index(this_type_lin_index,prod.shape)
            ###Populate axis coordinates###
            this_coordinate_set=[]
            for axis in range(len(this_type_index_set)):
                axis_index=this_type_index_set[axis]-1 #subtract 1 since we enlarged the array by 1
                
                #perhaps we don't have a set of x_values to draw from in *x_values*
                try: this_coordinate_set.append(x_values[axis][axis_index])
                #indeed we don't, just use axis index
                except IndexError: this_coordinate_set.append(axis_index)
                
            try:
                this_type_indices.append(this_type_index_set)
                this_type_coordinates.append(tuple(this_coordinate_set))
                
                #Interpolate our original y-values back to the coordinate from the expanded, incommensurate grid
                yinterp=y
                for i in range(y.ndim): yinterp=yinterp.interpolate_axis(this_coordinate_set[-1-i],axis=-1-i,bounds_error=False)
                this_type_values.append(yinterp)
            except: continue
            
        all_indices.append(this_type_indices)
        all_coordinates.append(this_type_coordinates)
        all_values.append(this_type_values)
        
    ###Sort by value###
    min_indices,min_coordinates,min_values=all_indices[0],all_coordinates[0],all_values[0]
    max_indices,max_coordinates,max_values=all_indices[1],all_coordinates[1],all_values[1]
    saddle_indices,saddle_coordinates,saddle_values=all_indices[2],all_coordinates[2],all_values[2]
    
    #A comparator for ordering the maxima in decreasing order#
    def descending(a,b):
        if a>b: return -1
        elif a==b: return 0
        else: return 1
    
    print min_values,min_indices,min_coordinates
    max_values,max_indices,max_coordinates=misc.sort_by(max_values,max_indices,max_coordinates,cmp=descending)
    min_values,min_indices,min_coordinates=misc.sort_by(min_values,min_indices,min_coordinates,cmp=cmp)
        
    return {'minima':{'indices':min_indices,'coordinates':min_coordinates,'values':min_values},\
            'maxima':{'indices':max_indices,'coordinates':max_coordinates,'values':max_values},\
            'saddle':{'indices':saddle_indices,'coordinates':saddle_coordinates,'values':saddle_values}}
"""

def convolve_periodically(in1,in2):
    """Perform a periodic (wrapping) convolution of the two 2-D inputs.  They
    must both be rank-1 real arrays of the same shape."""
    
    from numpy.fft import fft,ifft
    
    s1=fft(fft(in1,axis=0),axis=1)
    s2=fft(fft(in2,axis=0),axis=1)
    
    out=ifft(ifft(s1*s2,axis=0),axis=1).real
    
    return out#/numpy.sqrt(numpy.prod(out.shape))

def convolve_periodically_new(in1,in2):
    """Perform a periodic (wrapping) convolution of the two 2-D inputs.  They
    must both be rank-1 real arrays of the same shape over the same manifold.
    
    Result is normalized to represent a convolution sum.  To represent an
    integral, result should be multiplied by `dx*dy`, where `dx` and `dy` are
    respective integral measures of `x` and `y` pixels."""
    
    from numpy.fft import fft,ifft,fftshift,ifftshift
    
    s1=in1; s2=in2
    for i in range(2):
        s1=fftshift(fft(s1,axis=i),axes=i)
        s2=fftshift(fft(s2,axis=i),axes=i)
    
    out=s1*s2
    for i in range(2):
       out=ifft(ifftshift(out,axes=i),axis=i)
    
    #s1=fftshift(fft(shift(fft(fft(in1,axis=0),axis=1)
    #s2=fft(fft(in2,axis=0),axis=1)
    
    #out=ifft(ifft(s1*s2,axis=0),axis=1).real #Directly from the convolution theorem
    
    return out.real#*numpy.prod(out.shape)

def _downcasting_spectrum_method_(old_method):
    
    ##We don't want to redefine *view*, since redefinitions call *view*##
    if old_method.__name__ is 'view': return old_method
    
    def new_method(*args,**kwargs):
        
        ##Obtain result of bound method##
        result=old_method(*args,**kwargs)
        
        ##If result is another *Spectrum*, check that we have frequencies##
        if isinstance(result,Spectrum):
            #No frequencies, so downcast - no longer a true *Spectrum*#
            if len(result.get_frequency_dimensions())==0:
                return result.view(baseclasses.ArrayWithAxes)
            
        ##If result type is appropriate##
        return result
    
    ##Make new method attributes mirror the input method##
    for attr_name in dir(old_method):
        try: setattr(new_method,attr_name,getattr(old_method,attr_name))
        except: pass
        
    setattr(new_method,'__name__',old_method.__name__)
    
    return new_method
    
class Spectrum(baseclasses.ArrayWithAxes):
    """Acquire the spectrum of an input N-dimensional `source` along an
    axis of your choosing.  The result is a subclass of
    <common.baseclasses.ArrayWithAxes>, with additional methods added for the
    manipulation and display of spectral characteristics.  Unlike the conventional
    FFT, the resultant object is consistent with Parseval's theorem for the
    Fourier transform, which like all unitary transformations satisfies:
    
        integrate(|source|^2*dt) = integrate(|spectrum|^2*df)
        
    In particular, the action of `Spectrum.get_inverse` returns to the original
    source (internally, via the IFFT) with appropriate normalization to ensure
    complete identity."""
    
    ####Overload all inherited methods with downcasting equivalent####
    attr_names=dir(baseclasses.ArrayWithAxes)
    for attr_name in attr_names:
        method=getattr(baseclasses.ArrayWithAxes,attr_name)
        ##Only decorate callable methods##
        if not isinstance(method,types.MethodType): continue
        locals()[attr_name]=_downcasting_spectrum_method_(method)
    
    def __new__(cls,source,axis=None,duration=None,\
                fband=None,n=None,\
                power=False,normalized=False,\
                verbose=True,axes=None,axis_names=None,\
                per_sample=False,\
                window=None):
        
        fourier_transform=True
        
        ####If new axes or axis names supplied, add to source####
        if axes!=None or axis_names!=None:
            if isinstance(source,baseclasses.ArrayWithAxes): source.set_axes(axes=axes,axis_names=axis_names)
            else: source=baseclasses.ArrayWithAxes(source,axes=axes,axis_names=axis_names)
        
        ####If input is already a spectrum class, no need to Fourier xform unless requested####
        if isinstance(source,Spectrum) and axis is None: 
            spectrum=source
            fourier_transform=False
        
        ####We also accept *ArrayWithAxes* with frequency axes####
        elif isinstance(source,baseclasses.ArrayWithAxes):
            ##If frequency dimensions can be identified##
            if (axis is None) and True in ['Frequency' in axis_name for axis_name in source.axis_names]:
                spectrum=source.view(Spectrum)
                fourier_transform=False
            elif 'Frequency' in source.axis_names[axis]:
                spectrum=source.view(Spectrum)
                fourier_transform=False
        
        ####Get spectrum and cast as new type####
        if fourier_transform:
            if axis is None: axis=0 #We need a default
            
            ###Require axis###
            Logger.raiseException('Frequency axes can not be discerned for *source* '+\
                                 'in its provided form.  Please provide either an '+\
                                 '*ArrayWithAxes* instance with one or more axes labeled '+\
                                 '"... Frequency", or specify an *axis* along which to Fourier '+\
                                 'transform.', unless=isinstance(axis,int),\
                                 exception=TypeError)
            
            ###Check inputs###
            assert isinstance(duration,number_types+(type(None),))
            
            ###See if we need to interpolate for homogeneous *axis*###
            axis=axis%source.ndim
            interpolate_axis=False
            if not isinstance(source,baseclasses.ArrayWithAxes):
                source=baseclasses.ArrayWithAxes(source,verbose=False)
            else:
                #*diff* must be homogeneously increasing/decreasing if we don't have to interpolate
                #We make the margin of error for inhomogeneities 1/10000.
                time_values=source.axes[axis]
                diff=numpy.diff(time_values)
                inhomogeneous=(numpy.abs(diff-diff[0])/diff[0]>=1e-5)
                if True in inhomogeneous: interpolate_axis=True
            axes=source.axes
            axis_names=source.axis_names
            
            ###Interpolate axis to equi-spaced points if homogeneity of axis is uncertain###
            if interpolate_axis:
                try: from scipy.interpolate import interp1d
                except ImportError: Logger.raiseException('The module <scipy> is required for interpolation to homogeneous '+\
                                                          'samples along *axis*.',exception=ImportError)
                Logger.write('Interpolating to evenly-spaced samples along *axis*.')
                interpolator=interp1d(axes[axis],source,axis=axis)
                
                ##Produce homogeneous axis, interpolate across it, and store it##
                homog_axis=numpy.linspace(axes[axis][0],axes[axis][-1],len(axes[axis]))
                source=interpolator(homog_axis)
                axes[axis]=homog_axis
            
            ###Define axis, duration, nsamples###
            nsamples=source.shape[axis]
            if n is None: n=nsamples
            if duration is None: duration=axes[axis][-1]-axes[axis][0]
            
            ###Apply window###
            if window:
                if hasattr(window,'__call__'):
                    window=window(source.shape[axis])
                else:
                    Logger.raiseException("Window must be one of 'hanning', 'hamming', 'bartlett', 'blackman'.",\
                                          unless=(window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']),\
                                          exception=ValueError)
                    window=getattr(numpy,window)(nsamples)
                window*=nsamples/numpy.sum(window)
                window_shape=[1,]*source.ndim
                window_shape[axis]=len(window)
                window.resize(window_shape)
                source=source*window
            
            ###Acquire spectrum###
            dt=duration/float(nsamples-1)
            spectrum=numpy.fft.fft(source,axis=axis,n=n)
            frequencies=numpy.fft.fftfreq(n=n,d=dt) #duration specified as a window per-sample
        
            ###Unpack "standard" fft packing###
            spectrum=numpy.fft.fftshift(spectrum,axes=[axis])
            frequencies=numpy.fft.fftshift(frequencies)
            
            ###Re-normalize to maintain original total power in power spectrum###
            #Before we do anything, default behavior of FFT is to ensure:
            # sum(abs(spectrum)**2)=sum(abs(source)**2)*n_FFT
            
            #The following normalization ensures Parseval's theorem applies
            # identically between the function and its spectrum inside their 
            # respective integration domains
            spectrum*=cls.parseval_consistent_prefactor(dt)
            
            #This additional normalization will make the spectrum
            # describe the amplitude of power spectral density per
            # sample of the input
            if per_sample: spectrum/=numpy.sqrt(nsamples)
            
            ###Cast result as spectrum class###
            spectrum=super(cls,cls).__new__(cls,spectrum)
    
            ##Set up new axes##
            axes[axis]=frequencies
            axis_names[axis]+=' Frequency'
            spectrum.set_axes(axes,axis_names,verbose=verbose)
            
            ##Since we took Fourier transform, it's obvious this is not a power or normalized spectrum##
            power=False
            normalized=False
        
        ###Impose frequency limits if provided###
        if fband!=None and axis!=None:
            spectrum=spectrum.impose_fband(fband,axis=axis,closest=True)
        
        ###Set status as power/normalized/per-sample spectrum##
        if power:
            spectrum[:]=numpy.real(spectrum)
            spectrum._power_=True
        if normalized:
            spectrum._normalized_=True
        if per_sample:
            spectrum._per_sample_=True
                
        return spectrum
        
    def __array_finalize__(self,obj):
        
        ##First use super class finalization##
        super(Spectrum,self).__array_finalize__(obj)
        
        ##Include some tags about the character of the spectrum##
        if not hasattr(self,'_normalized_'):
            ##Can inherit tag from parent##
            if isinstance(obj,Spectrum): self._normalized_=obj._normalized_
            else: self._normalized_=False
        if not hasattr(self,'_power_'): 
            ##Can inherit tag from parent##
            if isinstance(obj,Spectrum): self._power_=obj._power_
            else: self._power_=False
        if not hasattr(self,'_per_sample_'):
            ##Can inherit tag from parent##
            if isinstance(obj,Spectrum): self._per_sample_=obj._per_sample_
            else: self._per_sample_=False
        
    def __getattribute__(self,attr_name):
        
        method_mapping={'power':'get_power_spectrum',\
                        'folded':'fold_frequencies',\
                       'frequency_dims':'get_frequency_dimensions',\
                       'geometric_dims':'get_geometric_dimensions',\
                       'frequency_axes':'get_frequency_axes',\
                       'geometric_axes':'get_geometric_axes',\
                       'frequency_axis_names':'get_frequency_axis_names',\
                       'geometric_axis_names':'get_geometric_axis_names',\
                       'phase':'get_phase',\
                       'geometric_mean':'get_geometric_mean',\
                       'geometric_min':'get_geometric_min',\
                       'geometric_max':'get_geometric_max',\
                       'geometric_sum':'get_geometric_sum',\
                       'geometric_integral':'get_geometric_integral',\
                       'inverse':'get_inverse'}
        
        for key in list(method_mapping.keys()):
            if attr_name==key:
                method=getattr(self,method_mapping[key])
                return method()
        return super(Spectrum,self).__getattribute__(attr_name)
        
    ##Overload *set_axes* in order to make sure frequency axes are preserved##
    def set_axes(self,axes=None,axis_names=None,\
                 verbose=True,intermediate=False):
        
        ##Vet *axis_names* to preserve frequency axes##
        if not intermediate and hasattr(axis_names,'__len__'):
            
            axis_names=list(axis_names)
            while len(axis_names)<self.ndim: axis_names+=[None]
            
            ##Only pay attention to names that correspond with current frequency dim##
            frequency_dims=self.get_frequency_dimensions()
            for dim in frequency_dims:
                
                ##Check axis name##
                provided_name=axis_names[dim]
                if isinstance(provided_name,str):
                    if not 'Frequency' in provided_name:
                        axis_names[dim]+=' Frequency'
                        if verbose:
                            Logger.write('%s:\n'%type(self)+\
                                 '\tAxis name "%s" will be coerced to "%s" '%(provided_name,axis_names[dim])+\
                                 'to reflect that dimension %s is a frequency dimension.'%dim)
        
        return super(Spectrum,self).set_axes(axes=axes,axis_names=axis_names,\
                                             verbose=verbose,intermediate=intermediate)
    
    @staticmethod
    def parseval_consistent_prefactor(dt=None,fmax=None):
        
        if dt is not None: return dt
        elif fmax is not None: return 1/(fmax*2) #twice Nyquist frequency gives original sampling interval `dt`
        
    def is_power_spectrum(self):
        
        return self._power_
        
    def is_normalized(self):
        
        return self._normalized_
    
    def is_per_sample(self):
        
        return self._per_sample_
    
    def adopt_attributes(self,other):
        
        Logger.raiseException('`other` must be another `Spectrum` instance in order to adopt attributes.',\
                              exception=TypeError,unless=isinstance(other,Spectrum))
        self._power_=other.is_power_spectrum()
        self._normalized_=other.is_normalized()
        self._per_sample_=other.is_per_sample()
    
    def get_frequency_dimensions(self):
        
        axis_names=self.get_axis_names()
        frequency_dims=[]
        for i in range(self.ndim):
            if 'Frequency' in axis_names[i]: frequency_dims.append(i)
            
        return frequency_dims
    
    def get_geometric_dimensions(self):
        
        all_dims=set(range(self.ndim))
        frequency_dims=set(self.get_frequency_dimensions())
        geometric_dims=list(all_dims-frequency_dims)
        
        return geometric_dims
    
    def get_frequency_axes(self):
        
        frequency_dims=self.get_frequency_dimensions()
        axes=self.get_axes()
        
        return [axes[dim] for dim in frequency_dims]
    
    def get_geometric_axes(self):
        
        geometric_dims=self.get_geometric_dimensions()
        axes=self.get_axes()
        
        return [axes[dim] for dim in geometric_dims]
    
    def get_frequency_axis_names(self):
        
        frequency_dims=self.get_frequency_dimensions()
        axis_names=self.get_axis_names()
        
        return [axis_names[dim] for dim in frequency_dims]
    
    def get_geometric_axis_names(self):
        
        geometric_dims=self.get_geometric_dimensions()
        axis_names=self.get_axis_names()
        
        return [axis_names[dim] for dim in geometric_dims]
    
    def get_power_spectrum(self):
        
        if self.is_power_spectrum(): return self
        else: 
            ##Get magnitude squared for power
            power_spectrum=numpy.abs(self)**2
            power_spectrum._power_=True
                
        return power_spectrum
    
    def get_phase(self,unwrap=True,discont=numpy.pi):
        
        phase=numpy.angle(self) #return radians
        
        #Try to make continuous#
        if unwrap:
            try:
                for axis in self.frequency_dims: phase=numpy.unwrap(phase,axis=axis,discont=discont)
            except ImportError: pass
        
        #Something stupid happens here if the *imag(...)* function passes parent as
        #an ndarray object to *__array_finalize__*, attributes aren't inherited - so force it
        if not isinstance(phase,Spectrum): phase=0*self+phase #Inherit from parent dammit!!
        phase._power_=False
        
        phase[numpy.isnan(phase)]=0 #Define nan phase to be zero
        
        return phase.astype(float)
        
    def get_geometric_mean(self):
        
        ##Sum along each geometric dimension##
        geometric_dims=self.get_geometric_dimensions()
        geometric_dims.reverse() #Reverse ordering so affecting posterior dimensions won't affect positions of anterior axes
        
        geometric_mean=self
        for dim in geometric_dims: geometric_mean=geometric_mean.mean(axis=dim)
            
        return geometric_mean
        
    def get_geometric_max(self):
        
        ##Sum along each geometric dimension##
        geometric_dims=self.get_geometric_dimensions()
        geometric_dims.reverse() #Reverse ordering so affecting posterior dimensions won't affect positions of anterior axes
        
        geometric_max=self
        for dim in geometric_dims: geometric_max=geometric_max.max(axis=dim)
            
        return geometric_max
        
    def get_geometric_min(self):
        
        ##Sum along each geometric dimension##
        geometric_dims=self.get_geometric_dimensions()
        geometric_dims.reverse() #Reverse ordering so affecting posterior dimensions won't affect positions of anterior axes
        
        geometric_min=self
        for dim in geometric_dims: geometric_min=geometric_min.min(axis=dim)
            
        return geometric_min
        
    def get_geometric_sum(self):
        
        ##Sum along each geometric dimension##
        geometric_dims=self.get_geometric_dimensions()
        geometric_dims.reverse() #Reverse ordering so affecting posterior dimensions won't affect positions of anterior axes
        
        geometric_sum=self
        for dim in geometric_dims: geometric_sum=geometric_sum.sum(axis=dim)
        
        ##Reset axes##
        geometric_sum.set_axes(self.get_frequency_axes(),\
                                self.get_frequency_axis_names())
        geometric_sum=Spectrum(geometric_sum,power=self.is_power_spectrum(),\
                                normalized=self.is_normalized())
            
        return geometric_sum
    
    def get_geometric_integral(self,integration=None):
        
        ##Use default integration scheme##
        if integration is None:
            from scipy.integrate import trapz as integration
        
        ##Integrate along each geometric dimension##
        shape=self.shape
        geometric_dims=self.get_geometric_dimensions()
        geometric_dims.reverse() #Reverse ordering so affecting posterior dimensions won't affect positions of anterior axes
        
        geometric_integral=self
        for dim in geometric_dims: geometric_integral=geometric_integral.integrate(axis=dim,integration=integration)
        
        ##Reset axes##
        #result of integration is likely *ndarray*, so cast as *ArrayWithAxes*#
        geometric_integral=baseclasses.ArrayWithAxes(geometric_integral,\
                                                     axes=self.get_frequency_axes(),\
                                                     axis_names=self.get_frequency_axis_names())
        geometric_integral=Spectrum(geometric_integral,power=self.is_power_spectrum(),\
                                normalized=self.is_normalized())
            
        return geometric_integral
    
    def get_inverse(self,axis=None,n=None,origin=None,offset=None):
        """
        `offset` will be applied to the axis values along the inverted dimension.
        
        `origin` will be taken as the origin of the IFFT along the axis with "offset"
        axis values.  This means that if `origin!=offset`, the IFFT output will be
        rolled along the inverted axis to coincide with the desired `origin` position.
        """
        
        Logger.raiseException('This is a power spectrum.  Power spectra cannot be '+\
                              'deterministically inverse-Fourier-transformed!',\
                              unless=(not self.is_power_spectrum()),\
                              exception=ValueError)
        
        ##Interpret axis along which to inverse transform##
        frequency_dims=self.get_frequency_dimensions()
        frequency_names=self.get_frequency_axis_names()
        axes=self.get_axes()
        if isinstance(axis,int):
            axis=axis%self.ndim
            Logger.raiseException('If provided as an integer, *axis* must be one of the '+\
                                  'following frequency dimensions:\n'+\
                                  '\t%s'%repr(frequency_dims),\
                                  unless=(axis in frequency_dims),\
                                  exception=IndexError)
        elif isinstance(axis,str):
            Logger.raiseException('If provided as a string, *axis* must be one of the '+\
                                  'following frequency axis names:\n'+\
                                  '\t%s'%repr(frequency_names),\
                                  exception=IndexError)
            axis=self.get_axis_names().index(axis)
        elif axis is None and len(frequency_dims)==1: axis=frequency_dims[0]
        else: Logger.raiseException('This spectrum has more than a single frequency axis! '+\
                                    'Please provide a valid specific frequency *axis* integer dimension '+\
                                    'or axis name.', exception=ValueError)
        
        ###First interpolate back to FFT-consistent axis values###
        freq_window=axes[axis].max()-axes[axis].min()
        df=numpy.min(numpy.abs(numpy.diff(axes[axis])))
        nsamples=int(numpy.round(freq_window/df))+1
        
        if n is None: n=nsamples
        
        dt=1/float(freq_window)*nsamples/float(n) #duration of each sampling in putative inverse transform
        freqs=sorted(numpy.fft.fftfreq(n=n,d=dt)) #duration specified as a window per-sample
        fftconsistent_self=self.interpolate_axis(freqs,axis=axis,fill_value=0,bounds_error=False)
        
        ##Just to see how the FFT-consistent version looks
        #self.fftconsistent_self=fftconsistent_self
        
        ###Pack up into "standard" fft packing###
        shifted_self=numpy.fft.ifftshift(fftconsistent_self,axes=[axis])
        
        ###Re-normalize to maintain the original normalization condition of FFT: ###
        # sum(abs(spectrum)**2)=sum(abs(source)**2)*n_FFT
        shifted_self/=self.parseval_consistent_prefactor(dt) #We multiplied this earlier, now remove
        if self.is_per_sample(): shifted_self*=numpy.sqrt(nsamples)
        
        ###Acquire inverse spectrum###
        if offset is None: offset=0
        inverse=numpy.fft.ifft(shifted_self,axis=axis,n=n)
        positions=numpy.linspace(0,n*dt,n)+offset
        if origin is not None:
            offset=origin-positions[0] #if `offset` becomes positive, `mean_pos` is too small and positions need to be up-shifted
            nshift=int(offset/dt)# if desired desired origin is more positive than minimum position, offset is positive and roll forward
            inverse=numpy.roll(inverse,nshift,axis=axis)
            
        ###Re-assign axes and axis names###
        axes=self.get_axes()
        axis_names=self.get_axis_names()
        axis_name=axis_names[axis]
        frequency_exp=re.compile('\s+Frequency$')
        new_axis_name=re.sub(frequency_exp,'',axis_name)
        axis_names[axis]=new_axis_name
        axes[axis]=positions
        
        #Result of integration is likely *ndarray*, so cast as *ArrayWithAxes*#
        inverse=baseclasses.ArrayWithAxes(inverse,axes=axes,axis_names=axis_names)
        
        #If frequency axes remain, re-cast as spectrum#
        if len(frequency_dims)>=2: inverse=Spectrum(inverse,normalized=self.is_normalized())
        
        return inverse
    
    def bandpass(self,fband,axis,closest=False):
        
        ##Check fband##
        message='*fband* must be an iterable of length 2, both entries positive frequency values.'
        Logger.raiseException(message,unless=hasattr(fband,'__len__'), exception=TypeError)
        Logger.raiseException(message,unless=(len(fband)==2), exception=IndexError)
        Logger.raiseException(message,unless=(fband[0]>=0 and fband[1]>=0), exception=ValueError)
        
        ##Check axis, make sure it's a frequency axis##
        frequency_dims=self.frequency_dims
        if isinstance(axis,str):
            axis_names=self.axis_names
            Logger.raiseException('If a string, *axis* must be a frequency axis among %s'%axis_names,\
                                  unless=(axis in axis_names), exception=IndexError)
            axis=axis_names.index(axis)
        else:
            assert isinstance(axis,int)
            axis=axis%self.ndim
        Logger.raiseException('*axis* must be one of the following frequency dimensions for '+\
                             'this spectrum: %s'%frequency_dims, unless=(axis in frequency_dims),\
                             exception=IndexError)
            
        frequencies=self.get_axes()[axis]
        abs_frequencies=numpy.abs(frequencies)
        limiting_indices=(abs_frequencies>=numpy.min(fband))*(abs_frequencies<=numpy.max(fband))
        if not limiting_indices.any() and closest:
            diff_lower=numpy.abs(abs_frequencies-numpy.min(fband))
            diff_upper=numpy.abs(abs_frequencies-numpy.max(fband))
            closest_lower=abs_frequencies[diff_lower==diff_lower.min()].min()
            closest_upper=abs_frequencies[diff_upper==diff_upper.min()].max()
            Logger.warning('No frequency channels in the range of %s were '%repr(fband)+\
                           'found, so the closest available range %s will '%repr((closest_lower,closest_upper))+\
                           'instead be imposed.')
            limiting_indices=(abs_frequencies>=closest_lower)*(abs_frequencies<=closest_upper)
        
        ##Multiply spectrum by a mask which removes the unwanted frequency channels##
        mask_shape=[1 for dim in range(self.ndim)]; mask_shape[axis]=len(limiting_indices)
        mask=limiting_indices.reshape(mask_shape)
        spectrum=self.copy()
        spectrum*=mask
        
        return spectrum
    
    def reduce_to_fband(self,fband,axis,closest=False):
        
        ##Check fband##
        message='*fband* must be an iterable of length 2, both entries positive frequency values.'
        Logger.raiseException(message,unless=hasattr(fband,'__len__'), exception=TypeError)
        Logger.raiseException(message,unless=(len(fband)==2), exception=IndexError)
        Logger.raiseException(message,unless=(fband[0]>=0 and fband[1]>=0), exception=ValueError)
        
        ##Check axis, make sure it's a frequency axis##
        frequency_dims=self.frequency_dims
        if isinstance(axis,str):
            axis_names=self.axis_names
            Logger.raiseException('If a string, *axis* must be a frequency axis among %s'%axis_names,\
                                  unless=(axis in axis_names), exception=IndexError)
            axis=axis_names.index(axis)
        else:
            assert isinstance(axis,int)
            axis=axis%self.ndim
        Logger.raiseException('*axis* must be one of the following frequency dimensions for '+\
                             'this spectrum: %s'%frequency_dims, unless=(axis in frequency_dims),\
                             exception=IndexError)
            
        frequencies=self.get_axes()[axis]
        abs_frequencies=numpy.abs(frequencies)
        limiting_indices=(abs_frequencies>=numpy.min(fband))*(abs_frequencies<=numpy.max(fband))
        if not limiting_indices.any() and closest:
            diff_lower=numpy.abs(abs_frequencies-numpy.min(fband))
            diff_upper=numpy.abs(abs_frequencies-numpy.max(fband))
            closest_lower=abs_frequencies[diff_lower==diff_lower.min()].min()
            closest_upper=abs_frequencies[diff_upper==diff_upper.min()].max()
            Logger.warning('No frequency channels in the range of %s were '%repr(fband)+\
                           'found, so the closest available range %s will '%repr((closest_lower,closest_upper))+\
                           'instead be imposed.')
            limiting_indices=(abs_frequencies>=closest_lower)*(abs_frequencies<=closest_upper)
        
        ##See if frequency limits aren valid##
        Logger.raiseException('Frequency channels in the range of %s are '%repr(fband)+\
                              'not available for the obtained spectrum.  Instead, *fband* '+\
                              'must span some (wider) subset in the range of %s.  '%repr((numpy.min(abs_frequencies),\
                                                                                      numpy.max(abs_frequencies)))+\
                              'Alternatively, use *closest=True* to accept frequency channels '+\
                              'closest to the desired frequency band.',\
                              unless=limiting_indices.any(), exception=IndexError)
    
        ##Boolean slicing would inevitably destroy axes information##
        #We must be careful to record the axes so we can re-apply after slicing#
        #Prepare a new set of axes with the correct frequencies#
        axis_names=self.axis_names
        new_axes=self.axes
        limited_frequencies=frequencies[limiting_indices]
        new_axes[axis]=limited_frequencies
        
        ##Perform slicing on vanilla array##
        limiting_slice=[None for i in range(self.ndim)]
        limiting_slice[axis]=limiting_indices
        spectrum=get_array_slice(numpy.asarray(self), limiting_slice, squeeze=False)
        
        ##Re-set axes and transform back into spectrum##
        spectrum=baseclasses.ArrayWithAxes(spectrum,axes=new_axes,axis_names=axis_names) #Re-set axes
        spectrum=Spectrum(spectrum)
        
        return spectrum
    
    impose_fband=reduce_to_fband
    
    def fold_frequencies(self):
        
        ####We'll need interpolation####
        try: from scipy.interpolate import interp1d
        except ImportError: Logger.raiseException('The module <scipy.interpolate> is required for \
                                              this operation, but is not available.',\
                                              exception=ImportError)
        
        ####Prepare container for folded frequencies spectrum####
        if self.is_power_spectrum(): s=self.copy()
        else: s=self.get_power_spectrum()
        
        ####Store parameters describing spectrum####
        axes=s.axes
        axis_names=s.axis_names
        normalized=s.is_normalized()
        power=s.is_power_spectrum()
        
        ##Add negative frequency channels to positive ones, keep only positive frequencies##
        for dim in self.get_frequency_dimensions():
            
            fs=axes[dim]
            where_pos=fs>=0
            where_neg=fs<0
            pos_half=get_array_slice(s,where_pos,axis=dim)
            neg_half=get_array_slice(s,where_neg,axis=dim)
            pos_fs=pos_half.axes[dim]
            neg_fs=neg_half.axes[dim]
            
            ##If no negative frequencies to fold here, move on!##
            if not len(neg_fs): continue
            
            ##Determine overlap range##
            pos_fband=[pos_fs.min(),pos_fs.max()]
            where_neg_overlap=(neg_fs>=-pos_fband[1])*(neg_fs<=-pos_fband[0])
            neg_overlap_fs=neg_fs[where_neg_overlap]
            overlap_fband=[numpy.abs(neg_overlap_fs).min(),\
                           numpy.abs(neg_overlap_fs).max()]
            where_pos_overlap=(pos_fs>=overlap_fband[0])*(pos_fs<=overlap_fband[1])
            
            ##Get positive and negative halves in this range only##
            neg_half=get_array_slice(neg_half,where_neg_overlap,axis=dim)
            pos_half=get_array_slice(pos_half,where_pos_overlap,axis=dim)
            neg_half_mirror=neg_half.coordinate_slice([None,None,-1],axis=dim)
            
            ##Reverse the negative half by index and give it positive frequencies, then coincide it with positive half##
            neg_half_mirror_axes=neg_half_mirror.axes
            neg_half_mirror_axes[dim]*=-1
            neg_half_mirror.set_axes(neg_half_mirror_axes)
            neg_half_mirror_coinciding=pos_half.get_coinciding_spectrum(neg_half_mirror)
            
            ##Make s the positive part, and add negative contribution##
            new_s=pos_half
            new_s+=neg_half_mirror_coinciding
            
            #If original spectrum was normalized, our folded result only really makes sense as a
            #normalized spectrum if we had folded both s1 and s2, THEN normalized s1 to s2.
            #Equivalently, we can divide by 2 right now so that it's if we had treated
            #s1 and s2 on the same footing.
            if s.is_normalized(): new_s/=2.
            
            #Concatenate the DC channel, if present
            if 0 in fs:
                where_DC=[numpy.argmin(numpy.abs(fs))]
                DC_slice=get_array_slice(s,where_DC,axis=dim)
                
                #Concatenate will 
                these_axes=new_s.axes; these_axis_names=new_s.axis_names
                these_axes[dim]=[0]+list(these_axes[dim])
                
                new_s=numpy.concatenate((DC_slice,new_s),axis=dim)
                new_s=Spectrum(new_s,axes=these_axes,axis_names=these_axis_names,axis=dim)
                new_s.adopt_attributes(s)
                
            s=new_s
            
        return s
                
    def spectral_overlap(self,spectrum):
        
        assert isinstance(spectrum,Spectrum)
        try: overlap_spectrum=numpy.real(self*numpy.conj(spectrum))
        except:
            Logger.raiseException('To obtain an overlap spectrum, *spectrum*, '+\
                             'must be of shape %s.'%repr(self.shape),\
                             exception=IndexError)
            
        ##Define inner product result as a power spectrum##
        overlap_spectrum._power_=True
        
        return overlap_spectrum
    
    def get_coinciding_spectrum(self,spectrum,geometric_resize='interpolate',copy=True):
        
        try: from scipy.interpolate import interp1d
        except ImportError:
            Logger.raiseException('The module <scipy> is required for spectral interpolation, but it is not available! '+\
                                 'This function is unusable until <scipy> is added to the library of Python modules.',\
                                 exception=ImportError)
        
        ####Check input####
        assert isinstance(spectrum,Spectrum)
        Logger.raiseException('*spectrum* must have the same number of '+\
                              'frequency dimensions (%s) as the present spectrum.'%len(self.frequency_dims),\
                              unless=(len(spectrum.frequency_dims)==len(self.frequency_dims)),\
                              exception=IndexError)
        geometric_resize_options=['interpolate','expand']
        Logger.raiseException('*geometric_resize* must be one of %s.'%geometric_resize_options,\
                              unless=geometric_resize in geometric_resize_options,\
                              exception=ValueError)
        
        ####Copy normalizing spectrum, we will change some its characteristics####
        if copy: spectrum=spectrum.copy()
        else: spectrum=spectrum.view()
        
        ###Expand out normalizing spectrum to correct number of geometric dimensions###
        self_geometric_dims=self.geometric_dims
        self_axes=self.axes
        spectrum_axes=spectrum.axes
        dummy_shape=list(spectrum.shape)
        
        ##Start inserting fake geometric dimensions##
        geometric_dims=numpy.array(spectrum.geometric_dims)
        for dim in self_geometric_dims:
            if dim in geometric_dims: continue
            #Insert a new spectrum_axis#
            dummy_shape.insert(dim,1)
            spectrum_axes.insert(dim,None) #Insert spectrum_axis placeholder
            #This pushes all dimensions up by 1, so add 1#
            geometric_dims+=1
            
        ##Give spectrum right number of dimensions##
        spectrum=numpy.resize(spectrum,dummy_shape)
        
        ####Now align each frequency spectrum_axis####
        for dim in self.frequency_dims:
            
            ###Align positive and negative frequencies separately###
            spectrum_halves=[]
            for fsign in [-1,1]:
                if fsign==-1:
                    self_indices=self_axes[dim]<0
                    spectrum_indices=spectrum_axes[dim]<0
                elif fsign==1:
                    self_indices=self_axes[dim]>=0
                    spectrum_indices=spectrum_axes[dim]>=0
                
                self_axis=self_axes[dim][self_indices]
                spectrum_half=get_array_slice(spectrum,spectrum_indices,axis=dim)
                spectrum_axis=spectrum_axes[dim][spectrum_indices]
                
                ###Prepare lengths in overlapping frequency range for comparison###
                self_length=len(self_axis)
                spectrum_length=len(spectrum_axis)
                #If there's no data in this frequency half for original spectrum, bail#
                if not self_length: continue
                
                ###Prepare frequency limits for comparison of FFT densities###
                try:
                    self_fband=[self_axis.min(),self_axis.max()]
                    spectrum_fband=[spectrum_axis.min(),spectrum_axis.max()]
                    
                    #We will find the overlap is zero if there's only a single spectrum point#
                    if self_length==1 and spectrum_fband[0]<=self_fband[0] \
                                      and spectrum_fband[1]>=self_fband[1]:
                        self_overlap=self_axis; spectrum_overlap=self_axis
                        
                    #Otherwise we have a legitimate overlap section#
                    else:
                        self_overlap=self_axis[(self_axis<=spectrum_fband[1])*\
                                                (self_axis>=spectrum_fband[0])]
                        spectrum_overlap=spectrum_axis[(spectrum_axis<=self_fband[1])*\
                                                       (spectrum_axis>=self_fband[0])]
                        
                    if not len(self_overlap): raise ValueError
                
                except ValueError:
                    Logger.raiseException('*spectrum* does not have any data points within the '+\
                                          'frequency range %s spanned by dimension %s.'%(self_fband,dim),\
                                          exception=ValueError)
                
                ##Notification of spectral accuracy##
                self_density=len(self_overlap)/(self_overlap.max()-self_overlap.min())
                spectrum_density=len(spectrum_overlap)/(spectrum_overlap.max()-spectrum_overlap.min())
                if spectrum_density < self_density:
                    Logger.warning('The spectral resolution of the normalizing spectrum along dimension '+\
                                   '%s for frequencies of sign %s is more coarse than '%(dim,fsign)+\
                                   'that of the input spectrum.  Note that the resulting normalized '+\
                                   'spectrum may suffer from inaccuracy, especially if either spectrum '+\
                                   'under-samples the total power of its source data.')
    
                ###Determine correction factor for power density###
                ##By adding/removing frequency channels, we've increased/decreased the total power##
                ##Because the number of frequency channels has changed, re-normalize according to power distribution##
                factor=spectrum_density/float(self_density)
                if not self.is_power_spectrum(): factor=numpy.sqrt(factor)
                spectrum_half*=factor
                    
                ###Expand spectrum to span whole frequency range of *self*###
                spectrum_axis=list(spectrum_axis)
                ##Insert lower values into normalization spectrum as needed##
                if self_fband[0]<spectrum_fband[0]:
                    bottom_slice=[None for i in range(spectrum_half.ndim)]
                    bottom_slice[dim]=0
                    bottom_edge=get_array_slice(spectrum_half,bottom_slice)
                    spectrum_half=numpy.concatenate((bottom_edge,spectrum_half),axis=dim)
                    spectrum_axis.insert(0,self_fband[0])
                #Insert upper values into normalization as needed#
                if self_fband[1]>spectrum_fband[1]:
                    top_slice=[None for i in range(spectrum_half.ndim)]
                    top_slice[dim]=-1
                    top_edge=get_array_slice(spectrum_half,top_slice)
                    spectrum_half=numpy.concatenate((spectrum_half,top_edge),axis=dim)
                    spectrum_axis.append(self_fband[1])
                
                ##Interpolate to input frequency channels##
                #This interpolator should span all frequencies we can throw at it from input spectrum.#
                try: interpolator=interp1d(x=spectrum_axis,y=spectrum_half,axis=dim)
                except ValueError:
                    Logger.raiseException('The normalizing spectrum is unsuitable - ill-posed data.')
                spectrum_halves.append(interpolator(self_axis))
                
            ##Concatenate the two spectrum halves together##
            spectrum=numpy.concatenate(spectrum_halves,axis=dim)
            
        ####Align geometric dimensions by resizing *spectrum*####
        #Now we are ready to safely resize using our home-cooked methods (only geometric dimensions will get resized)#
        if geometric_resize=='interpolate': spectrum=interpolating_resize(spectrum,self.shape)
        elif geometric_resize=='expand': spectrum=expanding_resize(spectrum,self.shape)
        
        ##Now axes should coincide, so adopt axes##
        return Spectrum(spectrum,axes=self.get_axes(),axis_names=self.get_axis_names())
    
    def normalize(self,normalizing_spectrum,\
                  in_place=True,\
                  geometric_mean=False,\
                  geometric_resize='interpolate',\
                  verbose=True):
        
        if not in_place: self=self.copy()
        
        ####Check input####
        Logger.raiseException('`normalizing_spectrum` must be a `Spectrum` instance.',\
                              unless=isinstance(normalizing_spectrum,Spectrum),\
                              exception=TypeError)
        Logger.raiseException('*normalizing_spectrum* must have the same number of '+\
                             'frequency dimensions (%s) as the present spectrum.'%len(self.frequency_dims),\
                             unless=(len(normalizing_spectrum.frequency_dims)==len(self.frequency_dims)),\
                             exception=IndexError)
        
        ####Copy normalizing spectrum, we need to change some of its characteristics####
        normalizing_spectrum=copy.deepcopy(normalizing_spectrum)
        
        ####Characterize normalizing spectrum####
        ###We want the normalizing spectrum to conform to the type of the input spectrum###
        if normalizing_spectrum.is_power_spectrum() \
           and not self.is_power_spectrum():
            normalizing_spectrum=numpy.sqrt(normalizing_spectrum) #make normalizing spectrum a "regular" spectrum (though we lose phase information)
            if verbose:
                Logger.write('You are normalizing a regular spectrum to a power spectrum.  '+\
                              'The square root of the normalizing spectrum intensity will be used, '+\
                              'but the resulting phase is indeterminate.')
        elif not normalizing_spectrum.is_power_spectrum() \
             and self.is_power_spectrum():
            normalizing_spectrum=normalizing_spectrum.get_power_spectrum() #make normalizing spectrum a power spectrum
            
        ###Reduce to geometric mean if called for###
        if geometric_mean==True: normalizing_spectrum=normalizing_spectrum.geometric_mean
        
        ###Make normalizing spectrum coincide dimensionally before we can safely divide###
        normalizing_spectrum=self.get_coinciding_spectrum(normalizing_spectrum,\
                                                          geometric_resize=geometric_resize,\
                                                          copy=False) #Don't copy, we've already done that, view instead
        
        ####Finally, the shapes are uniform and ready for normalization####
        self/=normalizing_spectrum
        self._normalized_=True
        
    def versus_wavelength(self):
        
        ##Turn frequency axes to wavelength axes##
        #This will be an *ArrayWithAxes* instance#
        wl_version=self.view(baseclasses.ArrayWithAxes)
        axes=wl_version.get_axes()
        axis_names=wl_version.get_axis_names()
        for dim in self.get_frequency_dimensions():
            axes[dim]=1/axes[dim]
            axis_names[dim]=axis_names[dim].rstrip('Frequency')+'Wavelength'
        wl_version.set_axes(axes,axis_names)
        
        return wl_version.sort_by_axes()
        
    def plot(self,*args,**kwargs):
        
        ##Default to plotting vs. frequency##
        if 'wavelength' in kwargs: wavelength=kwargs.pop('wavelength')
        else: wavelength=False
        
        ##Turn frequency axes to wavelength axes if called for##
        if wavelength: 
            to_plot=self.versus_wavelength()
            return to_plot.plot(*args,**kwargs)
        
        else: return super(Spectrum,self).plot(*args,**kwargs)

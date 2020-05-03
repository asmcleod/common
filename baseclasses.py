import copy
import os
import re
from functools import wraps
from .log import Logger,trace
from . import misc

__module_name__=__name__

class Serializable:
    """
    This class provides a transparent interface to
    automated instance method decorations.  The
    decorations are performed with the *__setstate__*
    method, which is called on instantiation of this
    base class.  The associated *__getstate__* method
    ensures that the object can be serialized.  This 
    is crucial if subclasses are to be used with the 
    "copy", "Pickle", "cPickle", or other serialization
    modules.
    
    The only explicit use of this base object by its
    subclass should be in calling the *__init__* method
    when and wherever decorations should be performed.
    The decorations argument to *__init__* is stored,
    but it does not overwrite on subsequent calls.
    Instead, it appends decorations, so it may be 
    called an arbitrary number of times by nested
    subclasses.
    
    INPUTS:
        decorations:  a dictionary of the form:
        
            *{decorator: attributes to be decorated, ...}*
            
            Here "decorator" is the name of a decorator
            method, and "attributes..." is a list of
            the names of attribute methods to be
            decorated by that decorator.  Decorations
            are performed in the order as listed by:
            *decorations.keys()*
            Note that all "decorator" keys and the
            associated entries of "attributes..." must
            be valid names (string types) of methods
            described by the subclass.
    """
    
    def __init__(self,decorations={},translate_methods_from=[],optimize=False):
        
        ####Check inputs####
        if not optimize:
            misc.check_vars(decorations,dict)
            for key in list(decorations.keys()): misc.check_vars(key,str)
            for value in list(decorations.values()):
                Logger.raiseException('Each entry in the dictionary *decorations* must be an iterable of string attribute names.',\
                                 unless=(hasattr(value,'__len__') and not type(value)==str),
                                 exception=TypeError)
                misc.check_vars(value,(str,)*len(value))
                
            misc.check_vars(translate_methods_from,list)
            for key in translate_methods_from:
                Logger.raiseException('Each entry in the list *translate_methods_from* must be an attribute name '+\
                                 'in the dictionary of the current class instance.',\
                                 unless=hasattr(self,key),exception=AttributeError)
                
        ###Store decorations###
        ##Append if storage already exists##
        if hasattr(self,'__decorations__'):
            for key in list(decorations.keys()): self.__decorations__[key]=decorations[key]
        ##Generate if it doesn't##
        else: self.__decorations__=decorations
        
        ###Store inheritance###
        ##Append if storage already exists##
        if hasattr(self,'__translate_methods_from__'):
            self.__translate_methods_from__+=translate_methods_from
        ##Generate it if it doesn't##
        else: self.__translate_methods_from__=translate_methods_from
        
        ###Perform decorations###
        self.__setstate__()
    
    def __setstate__(self,state_dict=None):
        """
        In order to enable coherent serialization, decorations and
        method translations are encapsulated in this method.  This
        method applies to both instances and their dictionaries.
        """
        
        ##If provided with a state dictionary, we are probably loading from a serialization##
        if state_dict!=None:
            #Perform saved method decorations on dictionary#
            self._decorate_dict_methods_(state_dict)
            #Perform saved method translations on dictionary#
            self._translate_dict_methods_(state_dict)
            #Bind dictionary#
            self.__dict__=state_dict
        ##We are instantiating for the first time on an actual object##
        else:
            #Perform saved method decorations#
            self._decorate_instance_methods_()
            #Perform saved method translations#
            self._translate_instance_methods_()
        
    def __getstate__(self):
        
        ###Prepare state dictionary as snapshot, 1 level copy###
        state_dict=copy.copy(self.__dict__)
        
        ##Remove decorated methods##
        self._remove_decorated_methods_(state_dict)
        ##Remove translated methods##
        self._remove_translated_methods_(state_dict)
            
        return state_dict
    
    def _decorate_dict_methods_(self,state_dict):
        
        ####Get class type####
        superclass=super(self.__class__,self.__class__)
        
        ####Decorations###
        if '__decorations__' not in state_dict: state_dict['__decorations__']={}
        decorations=state_dict['__decorations__']
            
        ####Loop over decorators####
        decorators=list(decorations.keys())
        for decorator in decorators:
            
            ###Identify decorator as a method###
            decorator_method=getattr(self,decorator)
            
            ###Loop over the attributes (methods) to decorate###
            to_decorate=decorations[decorator]
            for attr in to_decorate:
                
                try:
                    ##Use superclass method as template##
                    old_method=getattr(superclass,attr) #Use superclass method as template
                    new_method=decorator_method(old_method)
                    
                    ##Try to wrap with same referential attributes##
                    try: new_method=wraps(old_method,assigned=('__name__','__doc__'))(new_method)
                    except: pass
                
                    ##Bind new method##
                    state_dict[attr]=new_method
                    
                except:
                    Logger.write('<%s.Serializable._decorate_dict_methods_>:\n'%__module_name__+\
                                                '   Decorating method "%s" failed:'%attr)
                    Logger.exception()
        
    def _decorate_instance_methods_(self):
        
        ####Get superclass that preserves MRO####
        superclass=super(type(self),type(self))
        
        ####Decorations###
        decorations=self.__decorations__
            
        ####Loop over decorators####
        decorators=list(decorations.keys())
        for decorator in decorators:
            
            ###Identify decorator as a method###
            decorator_method=getattr(self,decorator)
            
            ###Loop over the attributes (methods) to decorate###
            to_decorate=decorations[decorator]
            for attr in to_decorate:
                
                try:
                    ##Use superclass method as template##
                    old_method=getattr(superclass,attr) #Use superclass method as template
                    new_method=decorator_method(old_method)
                    
                    ##Try to wrap with same referential attributes##
                    try: new_method=wraps(old_method,assigned=('__name__','__doc__'))(new_method)
                    except: pass
                
                    ##Bind new method##
                    setattr(self,attr,new_method)
                    
                except:
                    Logger.write('<%s.Serializable._decorate_instance_methods_>:\n'%__module_name__+\
                                                '   Decorating method "%s" failed:'%attr)
                    Logger.exception()
                
    def _translate_dict_methods_(self,state_dict):
            
        ####Method translations####
        if '__translate_methods_from__' not in state_dict:
            state_dict['__translate_methods_from__']=[]
        translate_methods=state_dict['__translate_methods_from__']
            
        ####Loop over objects whose methods we will translate####
        for object_name in translate_methods:
            ##If there's nothing to translate from, skip
            if object_name not in state_dict: continue
            instance=state_dict[object_name]
            current_attribs=list(state_dict.keys())
                
            ##Translate each method##
            for attrib_name in dir(instance):
                
                try:
                    ##Don't overwrite attributes##
                    if attrib_name in current_attribs: continue
                    new_method=getattr(instance,attrib_name)
                    
                    ##If attribute is a non-private method, make it callable from the scope of *self*##
                    if not hasattr(new_method,'__call__') or attrib_name.startswith('_'): continue
                    
                    #Bind in local scope#
                    state_dict[attrib_name]=new_method
                    
                except:
                    Logger.write('<%s.Serializable._translate_dict_methods_>:\n'%__module_name__+\
                                                '   Translating method "%s" from object "%s", type %s, failed:'%\
                                                (attrib_name,object_name,type(instance)))
                    Logger.exception()
                
    def _translate_instance_methods_(self):
            
        ####Method translations####
        translate_methods=self.__translate_methods_from__
            
        ####Loop over objects whose methods we will translate####
        for object_name in translate_methods:
            ##If there's nothing to translate from, skip
            if not hasattr(self,object_name): continue
            instance=getattr(self,object_name)
            current_attribs=dir(self)
                
            ##Translate each method 
            for attrib_name in dir(instance):
                
                try:
                    ##Don't overwrite attributes##
                    if attrib_name in current_attribs: continue
                    new_method=getattr(instance,attrib_name)
                    ##If attribute is a non-private method, make it callable from the scope of *self*##
                    if not hasattr(new_method,'__call__') or attrib_name.startswith('_'): continue
                    
                    #Bind in local scope#
                    try: setattr(self,attrib_name,new_method)
                    except TypeError: pass
                    
                except:
                    Logger.write('<%s.Serializable._translate_instance_methods_>:\n'%__module_name__+\
                                                '   Translating method "%s" from object "%s", type %s, failed:'%\
                                                (attrib_name,object_name,type(instance)))
                    Logger.exception()
                
    def _remove_decorated_methods_(self,state_dict):
        
        ###Decorations##
        #Don't store decorated methods, they cannot be serialized#
        decorations=state_dict['__decorations__']
        attribs_to_remove=[]
        for decorator in list(decorations.keys()): attribs_to_remove+=list(decorations[decorator])
                
        ###Remove attributes from state dictionary###
        for attrib in attribs_to_remove:
            try: del state_dict[attrib]
            except KeyError: pass
        
    def _remove_translated_methods_(self,state_dict):
            
        ###Method translations###
        #Don't store translated methods, they cannot be serialized#
        translated_object_names=state_dict['__translate_methods_from__']
        attribs_to_remove=[]
        for object_name in translated_object_names:
            if not hasattr(self,object_name): continue
            else: object=getattr(self,object_name)
            for attrib_name in dir(object):
                attrib=getattr(object,attrib_name)
                if hasattr(attrib,'__call__'): attribs_to_remove.append(attrib_name)
                
        ###Remove attributes from state dictionary###
        for attrib in attribs_to_remove:
            try: del state_dict[attrib]
            except KeyError: pass

###Try to set up some base classes that rely on numpy###
try:
    import numpy
    
    ###Define valid number types###
    number_types=(int,int,float,numpy.int32,numpy.int64,numpy.float,numpy.float32,numpy.float64)
    if 'float128' in dir(numpy): number_types+=(numpy.float128,)

    class LabeledBaseClass: pass

    def MakeLabeledClass(inherited_type):
    
        #This decorator is used only for the class definition,#
        #Not for an instance creation - this saves time and makes#
        #the result serializable.#
        #The decorator is defined outside the class definition#
        #because to use it in the class definition, it would have#
        #to be a static method, but static methods are not callable#
        #unless by an instance or by the class itself (frustrating).#
        
        def labeling_decorator(method):
            """Coerces the result of a method to be of type *LabeledObject*.
            The difference should be completely transparent as long as the
            class definition does its job."""
            
            def labeling_method(self,*args,**kwargs):
                
                result=method(self,*args,**kwargs)
                
                ##In place method, no labels to update##
                if result is self: pass
                ##Update labels on result##
                else:
                    ##Coercion to *type(self)* was not successful, update type##
                    if not isinstance(result,type(self)): 
                        try: result=type(result).__new__(type(self),result)
                        except: result=type(result).__new__(type(self))
                    
                    result._labels_=self._labels_
                
                return result
            
            ##Make its attributes mirror the input function##
            for attr_name in dir(method)+['__name__']: #__name__ attribute is not included in *dir*, must add it explicitly
                try: setattr(labeling_method,attr_name,getattr(method,attr_name))
                except: pass
            
            return labeling_method
    
        class LabeledClass(inherited_type,LabeledBaseClass):
                    
            ##Perform decorations##
            #This wraps method outputs into *TaggedObject*
            #instances which preserve the underlying markers.
            dont_decorate=['__new__','__init__','__getattr__','__setattr__','__getattribute__','__repr__']
            for attr_name in dir(inherited_type):
                if attr_name in dont_decorate: continue
                attr=getattr(inherited_type,attr_name)
                if hasattr(attr,'__call__'): locals()[attr_name]=labeling_decorator(attr)
                
            def __new__(cls,obj,labels={}):
                
                if not isinstance(labels,dict):
                    raise TypeError('*LabeledClass* label arguments *names* and *values* must support iteration.')
                    
                ##In case we're inheriting from another *LabeledClass* subclass, we can retrieve/update label dictionary##
                if isinstance(obj,LabeledBaseClass):
                    old_labels=getattr(obj,'_labels_',{})
                    old_labels.update(labels)
                    labels=old_labels
                
                try: obj=inherited_type.__new__(cls,obj)
                except TypeError: obj=inherited_type.__new__(cls) #Sometimes *__new__* doesn't take parameters at all
                
                obj._labels_=labels
                
                return obj
            
            def get_labels(self):
                
                return self._labels_
            
            def get_label(self,name):
                
                return self.get_labels()[name]
            
            def set_label(self,name,value):
                
                self.get_labels()[name]=value
            
        return LabeledClass
    
    ##Overload some rightwards-acting operators to take control and coerce results 
    #just like leftwards-acting ones do by default##
    
    def __inheriting_binary_op__(operator_name):
        
        ndarray_operator=getattr(numpy.ndarray,operator_name)
        
        @wraps(ndarray_operator,assigned=('__name__','__doc__'))
        def inheriting_operator(self,*args,**kwargs):
            
            ##Obtain result of bound method##
            try: result=ndarray_operator(self,*args**kwargs)
            #Perhaps kwargs are not accepted#
            except TypeError: result=ndarray_operator(self,*args)
            
            ##If not an array subclass, we're done##
            if not isinstance(result,numpy.ndarray): return result
            
            ##Otherwise, cast as object type##
            else: 
                ##Define what class we need to recast results as##
                object_type=type(self)
                result=result.view(object_type)
                
                ##finalize using attributes of *self*, not just those of *obj*##
                result.__array_finalize__(self)
                
                return result
        
        return inheriting_operator
    
    def __inheriting_unary_op__(operator_name):
    
        ndarray_operator=getattr(numpy.ndarray,operator_name)
        
        @wraps(ndarray_operator,assigned=('__name__','__doc__'))
        def inheriting_operator(self,*args,**kwargs):
            
            ##Obtain result of bound method##
            try: result=ndarray_operator(self,*args,**kwargs)
            #Perhaps kwargs are not accepted#
            except TypeError: result=ndarray_operator(self,*args)
            
            ##Intercept axis argument if included##
            axis=None #default
            #Find in keywords
            if 'axis' in kwargs: axis=kwargs['axis']
            #Or find as integer argument in first position
            elif len(args)>0:
                if isinstance(args[0],int): axis=args[0]
            
            ##If a cumulative operation, return result##
            if axis is None: return result
            
            ##Otherwise,  *axis*##
            #Object should already be cast of correct type since allocated from *self*#
            elif isinstance(result,numpy.ndarray): 
                ##Define what class we need to recast results as##
                object_type=type(self)
                result=result.view(object_type)
                
                ##Explicitly inherit axes in the correct way##
                axes=self.get_axes()
                axis_names=self.get_axis_names()
                axes.pop(axis)
                axis_names.pop(axis)
                result.set_axes(axes=axes,axis_names=axis_names,intermediate=True)
                
                return result
        
        return inheriting_operator
    
    def __inheriting_resizing_op__(operator_name):
    
        ndarray_operator=getattr(numpy.ndarray,operator_name)
        
        @wraps(ndarray_operator,assigned=('__name__','__doc__'))
        def inheriting_operator(self,*args,**kwargs):
        
            if self.debug:
                Logger.write('In resizing op:\n'+\
                             'args: %s\n'%repr(args)+\
                             'kwargs: %s\n'%repr(kwargs)+\
                             'shape before: %s\n'%repr(self.shape)+\
                             trace())
            
            #Retrieve axis grids before anything happens to *self*
            #(suppose *result* is *self* and axes get improperly fudged in *__array_finalize__*)
            axis_grids=self.get_axis_grids(broadcast=True) #changed
            old_axis_names=self.get_axis_names()
            old_ndim=self.ndim
            
            ###Apply resize to self###
            use_kwargs=True
            try: result=ndarray_operator(self,*args,**kwargs)
            #Perhaps kwargs are not accepted#
            except TypeError: result=ndarray_operator(self,*args); use_kwargs=False
            
            ###Bail if we don't have an array anymore###
            if not isinstance(result,type(self)): return result
            
            ###Apply resizing to axis grids###
            if use_kwargs: axis_grids=[ndarray_operator(axis_grid,*args,**kwargs) \
                                       for axis_grid in axis_grids]
            else: axis_grids=[ndarray_operator(axis_grid,*args) \
                              for axis_grid in axis_grids]
            
            ###If the axis grid is not actually an array, but a number, then we have
            #resized to return a single element of the array, and we may return it.
            if not isinstance(axis_grids[0],numpy.ndarray): return result
                
            ###Now discern new axes from sliced grids###
            axes=result.get_axes()
            axis_names=result.get_axis_names()
            new_ndim=result.ndim
            axes_set=[False for i in range(new_ndim)]
            
            ##Iterate over grids and see if they correspond to new axis##
            for i in range(old_ndim):
                
                axis_grid=axis_grids[i]
                axis_name=old_axis_names[i]
                
                ##Check if label can describe an axis along dimension *j* of new array##
                for j in range(new_ndim):
                    
                    ##Continue if this axis has already been set##
                    #FIFO takes precedence, there may be some degeneracy for reduced axes#
                    if axes_set[j]: 
                        if self.debug: Logger.write('\tAxis %i already set, continuing.'%j,\
                                                    format=False)
                    
                    #The conditions to correlate a label with an axis in the present array are:
                    #1. label grid is uniform in all dimensions besides *j*
                    #2. label values vary along dimension *j* OR dimension *j* is 1
                    #These conditions will only confuse multiple axes if they are both length 1, but
                    #distinguishing them in this case is not physically significant anyway.
                    slicing=[slice(0,1)]*j+[slice(None)]+[slice(0,1)]*(new_ndim-(j+1))
                    grid_slice=axis_grid.__getitem__(tuple(slicing))
                    if 0 in grid_slice.shape or len(grid_slice.shape)==0: continue
                    
                    #First condition
                    condition1=(grid_slice==axis_grid)
                    if issubclass(type(condition1),numpy.ndarray): condition1=condition1.all()
                    if not condition1: continue
                    if self.debug: Logger.write('\tCheck! Axis grid %i aligns along new dimension %i.'%\
                                                (i,j),format=False)
                    
                    #Second condition
                    condition2=(grid_slice.flatten()[0]!=grid_slice) #If new array has shape 1, how can axis vary?
                    if issubclass(type(condition2),numpy.ndarray): condition2=condition2.any()
                    #If condition1 is satisfied but not condition2, we have a degenerate axis along axis j#
                    #If the dimension size here is 1, that's ok, the axis fits, so proceed to use axis#
                    #If not, bail out.
                    if not condition2 and result.shape[j]!=1: continue
                    if self.debug: Logger.write('\tCheck! Axis grid %i varies along new dimension %i.'%\
                                                (i,j),format=False)
                    
                    ##Assign new axis based on grid slice along dimension *j*##
                    axes[j]=grid_slice.flatten()
                    axis_names[j]=axis_name
                    axes_set[j]=True
                    if self.debug: Logger.write('\tIdentified axis in position %i.'%j,\
                                                format=False)
                    break
            
            ##Set new axis array with found axes and axis names##
            result.set_axes(axes,axis_names,verbose=False,intermediate=True)
            
            return result
        
        return inheriting_operator

    def expand_limits(index_limits,shape):
        """Coerce a list of arguments to *slice* objects
        to equivalent positive index arguments applicable
        to an array with shape *shape*."""
        
        ##Make sure index_limits takes the proper form##
        Logger.raiseException('*index_limits* should be an iterable of numerical index_limits (with *None* representing axis auto-fill).',\
                         unless=(hasattr(index_limits,'__len__') or index_limits is None), exception=TypeError)
        if index_limits is None or len(index_limits)==0: index_limits=[None]*len(shape)
        else:
            #Assure that index_limits has the proper length#
            if len(shape)==1 and len(index_limits)==2: index_limits=[index_limits]
            elif len(index_limits)>len(shape): index_limits=index_limits[:len(shape)]
            elif len(index_limits)<len(shape): index_limits=list(index_limits)+[None]*(len(shape)-len(index_limits))
        
        slice_args=[]
        for i in range(len(index_limits)):
            limit_set=index_limits[i]
            
            ##If we've got a slice, turn it into indices which we can inspect and coerce##
            if isinstance(limit_set,slice): limit_set=[limit_set.start,\
                                                       limit_set.stop,\
                                                       limit_set.step]
            
            ###We turn limit sets into arguments for a slice operation over dimension *i*###
            if limit_set is None: slice_args.append([0,shape[i],1])
            
            ##Assume we are using advanced array slicing##
            elif isinstance(limit_set,numpy.ndarray): 
                Logger.raiseException('If an element of *index_limits* is provided as an index array, it must be of dimension 1.',\
                                 unless=(limit_set.ndim==1), exception=TypeError)
                slice_args.append(limit_set)
            
            ##Assume we are using range slicing##
            elif hasattr(limit_set,'__len__'):
                
                #An empty set must be filled#
                if len(limit_set)==0: slice_args.append([0,shape[i],1]) #Default to full list
                
                #Otherwise specify a slice#
                else:
                    Logger.raiseException('If an element of *index_limits* is provided as a limit list, it must be of length 3 or less.',\
                                     unless=(len(limit_set)<=3), exception=IndexError)
                    
                    #A single filled set denotes a single coordinate#
                    if len(limit_set)==1: limit_set=[limit_set[0],limit_set[0]+1]
                    
                    #More than one values signifies [start,stop(,step)]
                    #Coerce to valid index_limits#
                    limit_set=list(limit_set)
                    for j in range(2): 
                        limit=limit_set[j]
                        
                        #Coerce limit to extreme value if *None*#
                        if limit is None:
                            if j==0: limit=0
                            else: limit=shape[i]
                        
                        #Coerce limit to extreme value if outside range#
                        else:
                            if limit<0: 
                                if limit<-shape[i]: limit=0
                                limit=limit%shape[i]
                            elif limit>shape[i]: limit=shape[i]
                        
                        limit_set[j]=int(limit)
                    
                    ##Check out step size##
                    if len(limit_set)>2:
                        if limit_set[2] is None: limit_set[2]=1
                        else: limit_set[2]=numpy.round(limit_set[2])
                    #Add step size if we don't have it#
                    else: limit_set.append(1)
                    
                    slice_args.append(limit_set[:3])
            
            ##Assume we are using single-coordinate slicing##
            else:
                limit_set=int(limit_set)%shape[i]
                slice_args.append([limit_set,limit_set+1,1])
                
        return slice_args

    def get_array_slice(arr,index_limits,squeeze=False,axis=None):
        
        if axis!=None:
            index_set=index_limits
            index_limits=[None for dim in range(arr.ndim)]
            index_limits[axis]=index_set
        slice_args=expand_limits(index_limits,arr.shape)
        
        ##Prepare slices##
        slices=[]
        for args in slice_args:
            if type(args)==numpy.ndarray: slices.append(args)
            else: slices.append(slice(*args))
        
        ##Only works with h5py dataset objects when a tuple, don't know why
        output=arr[tuple(slices)]
        
        if squeeze==True: return output.squeeze()
        else: return output

    class ArrayWithAxes(numpy.ndarray):
        
        ##Set this class-wide attribute to True to provide debug messages about axis-setting mechanics##
        debug=False
        
        global __inheriting_binary_op__
        __binary_ops__=['__radd__',\
                        '__rand__',\
                        #'__rdiv__',\
                        '__rdivmod__',\
                        '__rfloordiv__',\
                        '__rshift__',\
                        '__rmod__',\
                        '__rmul__',\
                        '__ror__',\
                        '__rpow__',\
                        '__rrshift__',\
                        '__rsub__',\
                        '__rtruediv__',\
                        '__rxor__']
        for binary_op in __binary_ops__: locals()[binary_op]=__inheriting_binary_op__(binary_op)
        del binary_op
        
        global __inheriting_unary_op__
        __unary_ops__=['max','mean','min','prod','sum','std']
        for unary_op in __unary_ops__: locals()[unary_op]=__inheriting_unary_op__(unary_op)
        del unary_op
        
        global __inheriting_resizing_op__
        __resizing_ops__=['__getitem__',\
                          #'__getslice__',\
                          'compress',\
                          'transpose',\
                          'swapaxes',\
                          'resize',\
                          'reshape',\
                          'repeat']
        for resizing_op in __resizing_ops__: locals()[resizing_op]=__inheriting_resizing_op__(resizing_op)
        del resizing_op
         
        def __new__(cls, obj, axes=None,axis_names=None,verbose=False,dtype=None,adopt_axes_from=None):
            
            if cls.debug:
                Logger.write('cls: %s'%cls)
                if isinstance(obj,numpy.ndarray): Logger.write('obj shape: %s'%repr(obj.shape))
            
            ##Cast object to correct class type##
            if not isinstance(obj,cls): obj=numpy.asarray(obj,dtype=dtype).view(cls)
            #Otherwise just try to cast element type if we already have *ArrayWithAxes*
            elif dtype!=None: obj=obj.astype(dtype)
            
            ##If we have an ArrayWithAxes from which to adopt axes##
            if adopt_axes_from is not None:
                axes=adopt_axes_from.get_axes()
                axis_names=adopt_axes_from.get_axis_names()
            
            ##If we have explicit axes to set##
            if axes!=None or axis_names!=None:
                if cls.debug: Logger.write('\tSetting axes for the first time, id: %s'%id(obj),\
                                           trace(),format=False)
                obj.set_axes(axes,axis_names,\
                             verbose=verbose)
            
            return obj
            
        def __array_finalize__(self,parent):
            
            if self.debug:
                try: Logger.write('self: %s, shape: %s, id: %s\n'%(type(self),self.shape,id(self))+\
                                  'parent: %s, shape: %s, id: %s'%(type(parent),parent.shape,id(parent)))
                except: Logger.write('self: %s, shape: ?, id: %s\n'%(type(self),id(self))+\
                                  'parent: %s, shape: ?, id: %s'%(type(parent),id(parent)))
            
            #We are using explicit constructor - *__new__* will take care of default attributes
            if parent is None: return
            
            ##Or we are updating in place##
            elif parent is self: return
            
            ##Here's where we try to inherit parent's axes names etc. if we need to##
            #These axes may not be valid if the new allocated array is of a different shape#
            #But if there's a mismatch, we ask the constructor to fail silently and defaults#
            #will be assigned#
            if not self.has_set_axes() and isinstance(parent,ArrayWithAxes):
                if parent.has_set_axes() and parent.has_set_axis_names():
                    if self.debug:
                        Logger.write('\tSelf to inherit axes from parent...\n'+\
                                     trace(),format=False)
                    self.set_axes(parent._axes_,\
                                  parent._axis_names_,\
                                  verbose=False,\
                                  intermediate=True)
            
        ###Expose easy coordinate slicing via *__getitem__* in a small class###
        class __cslice__(object):
            
            def __init__(self,instance): self.toslice=instance
                
            def __getitem__(self,*args): 
                
                ##We know that *__getitem__* strangely bundles multiple slices into tuple as *arg[0]*, so unpack if necessary##
                if hasattr(args[0],'__len__'): args=args[0]
                
                return self.toslice.coordinate_slice(*args)
            
        def __getattribute__(self,attr_name):
        
            ##Provide easy access to *cslice* instance for accessing coordinate slices##
            if attr_name=='cslice': return self.__cslice__(self)
            elif attr_name=='T': return numpy.transpose(self)
        
            method_mapping={'axes':'get_axes',\
                            'axis_names':'get_axis_names',\
                            'index_grids':'get_index_grids',\
                            'axis_grids':'get_axis_grids',\
                            'axis_limits':'get_axis_limits'}
            
            for key in list(method_mapping.keys()):
                if attr_name==key:
                    method=getattr(self,method_mapping[key])
                    return method()
            
            return super(ArrayWithAxes,self).__getattribute__(attr_name)
            
        ##For pickling and unpickling, respectively##
        ##Both these functions were lifted from the famous ndarray subclass, *InfoArray*##
        def __reduce__(self):
            
            ndarray_state = list(super(ArrayWithAxes,self).__reduce__())
            subclass_state = self.__dict__
            ndarray_state[2] = (ndarray_state[2],subclass_state)
            
            return tuple(ndarray_state)
        
        def __setstate__(self,state):
            
            ndarray_state, subclass_state = state
            super(ArrayWithAxes,self).__setstate__(ndarray_state)
            self.__dict__.update(subclass_state)

        def set_axes(self,axes=None,axis_names=None,\
                     verbose=False,intermediate=False):
                
            if self.debug: Logger.write('Setting axes on self: id %i'%id(self))
                
            ##Check validity of axes##
            if axes is None: axes=self.get_axes() #Default axes list
            if axis_names is None: axis_names=self.get_axis_names() #Default names list
            Logger.raiseException('*axes* and *axis_names* must both be iterables containing '+\
                             'information for each axis of the data array.',\
                             unless=(hasattr(axes,'__len__') and hasattr(axis_names,'__len__')),\
                             exception=TypeError)
            try_axes,try_axis_names=list(axes),list(axis_names)
            if self.debug:
                Logger.write('self: %s, shape: %s, id: %s\n'%(type(self),self.shape,id(self))+\
                             'Axes shape provided: %s'%[len(axis) for axis in axes])

            ##Keep track of which provided axes/names have been assigned##
            axis_available=[True for axis in axes]
            axis_name_available=[True for axis_name in axis_names]

            ##Pre-populate with default axes##
            use_axes=self.get_axes()
            use_axis_names=self.get_axis_names()
            axis_set=[False for axis in use_axes]
            axis_name_set=[False for axis_name in use_axis_names]
            
            def try_to_assign_axis(axis_index,try_axis_index):
                
                if axis_set[axis_index] or not axis_available[try_axis_index]: return
                try:
                    try_axis=try_axes[try_axis_index]
                    #Indicate that axis wasn't provided#
                    if try_axis is None: raise IndexError
                    #Try to compress into suitable shape#
                    try_axis=numpy.array(try_axis)
                    if try_axis.ndim>1: try_axis=try_axis.flatten()
                    #Make into array and check for length/1-D size
                    if len(try_axis)!=self.shape[axis_index]: raise TypeError
                    use_axes[axis_index]=try_axis
                    axis_set[axis_index]=True
                    axis_available[try_axis_index]=False
                
                ##Keep default in this axis and continue
                except TypeError: Logger.raiseException('The axis in position %s provided for '%try_axis_index+\
                                                        'dimension %s is of improper length '%axis_index+\
                                                        'or cannot be cast as *ndarray*:\n'+\
                                                        'Provided axes dimensions: %s\n'%[len(axis) if not axis is None else None\
                                                                                          for axis in try_axes]+\
                                                        'Required axes dimensions: %s'%list(self.shape),\
                                                        unless=intermediate, exception=ValueError); return False
                
                ##Keep default in this axis and continue
                except IndexError:
                    if verbose: Logger.write('An axis for dimension %s was not '%axis_index+\
                                             'explicitly provided.  A default axis will be used.')
                    
                ##Try to set axis names##
                try:
                    try_axis_name=try_axis_names[try_axis_index]
                    #Indicate that axis name wasn't provided
                    if try_axis_name is None: raise IndexError
                    #Check if string#
                    elif not isinstance(try_axis_name,str): raise TypeError
                    #Finally, use this axis#
                    use_axis_names[axis_index]=try_axis_name
                    axis_name_set[axis_index]=True
                    axis_name_available[try_axis_index]=False
                    
                ##Keep default name in this axis and continue
                except TypeError: Logger.raiseException('The axis name in position %s provided '%try_axis_index+\
                                                   'for dimension %s is not a valid '%axis_index+\
                                                   'string-representable type.',\
                                                   unless=intermediate, exception=TypeError); return False
                    
                ##Keep default name in this axis and continue
                except IndexError:
                    if verbose: Logger.write('An axis name for dimension %s was '%axis_index+\
                                             'not explicitly provided.  '+\
                                             'A default axis name will be used.')
                    
                return True
            
            ##If we expect a 1:1 match between provided and required axes, iterate i-->i
            if len(try_axes)==self.ndim:
                for i in range(self.ndim): try_to_assign_axis(i,i)
            else:
                ##If we have more dimensions than provided axes, perhaps we have an unexpectedly added dimension##
                #For each provided axis, try to match along each dimension in turn
                if len(try_axes)<self.ndim:
                    for j in range(len(try_axes)):
                        for i in range(self.ndim):
                            if i+j>=self.ndim: break
                            try_to_assign_axis(i+j,j)
                ##If we have fewer dimensions than provided axes, perhaps we have an unexpectedly removed dimension##
                #For each dimension, try to match to each provided axis in turn
                elif len(try_axes)>self.ndim:
                    for i in range(self.ndim):
                        for j in range(len(try_axes)):
                            if j+i>=len(try_axes): break
                            try_to_assign_axis(i,j+i)
            
            if self.debug:
                Logger.write('Axis names provided: %s\n'%try_axis_names+\
                             'Axis names set to use: %s\n'%use_axis_names+\
                             'Axis names set: %s\n'%axis_name_set+\
                             'Axis names available: %s\n'%axis_name_available+\
                             trace())
            
            #Save checked axes and names#
            self._axes_=use_axes
            self._axis_names_=use_axis_names
        
        def adopt_axes(self,obj,verbose=True,intermediate=False):
            
            Logger.raiseException('Can only adopt axes of an *ArrayWithAxes* instance.',\
                             unless=isinstance(obj,ArrayWithAxes), exception=TypeError)
            axes=obj.get_axes()
            axis_names=obj.get_axis_names()
            
            return self.set_axes(axes,axis_names,\
                                 verbose=verbose,intermediate=intermediate)
        
        def get_index_grids(self,broadcast=True):
            
            # 2017.11.17 - Commented out, because we noticed `numpy.ogrid` was very slow for large sizes of `self`
            #if broadcast: grid=numpy.mgrid
            #else: grid=numpy.ogrid
            
            #return list(grid.__getitem__([slice(0,dim) for dim in self.shape]))
            
            grid=numpy.ogrid
            grids=grid.__getitem__([slice(0,dim) for dim in self.shape])
            
            if broadcast: grids=numpy.broadcast_arrays(*grids)
            
            #if broadcast:
                
            #    old_grids=grids
            #    grids=[]
            #    for i in range(self.ndim):
            #        
            #       grids_to_add=[grid if j==i else 0*grid \
            #                      for j,grid in enumerate(old_grids)]
            #        grids.append(numpy.sum(grids_to_add,axis=0))
                        
            return grids
        
        def get_axis_grids(self,broadcast=True):
            
            #Ca we speed this up like we did `get_index_grids`
            axes=self.get_axes()
            
            axis_grids=[]
            for i in range(self.ndim):
                slicing=[None for j in range(self.ndim)]
                slicing[i]=slice(None)
                axis_grids.append(axes[i][tuple(slicing)])
            
            if broadcast: axis_grids=numpy.broadcast_arrays(*axis_grids)
            
            #index_grids=self.get_index_grids(broadcast=broadcast)
            #axis_grids=[axes[i][index_grids[i]] for i in range(len(axes))]
            
            return axis_grids
            
        def ravel_axes(self,*args,**kwargs):
            
            axis_grids=self.get_axis_grids()
            raveled_axes=[axis_grid.ravel(*args,**kwargs) for axis_grid in axis_grids]
            
            return list(zip(*raveled_axes))
            
        def locate(self,value,dicts=False,closest=False):
            
            ###Find where equal to value###
            if not closest:
                where_equal=(self==value)
                grids=self.get_axis_grids()
                grid_lists=[grid[where_equal] for grid in grids]
                coord_groups=list(zip(*grid_lists))
                
            ###Or find where closest to value###
            else:
                try: diff=numpy.abs(self-value)
                except TypeError:
                    Logger.raiseException('Not all the data in this array support numeric subtraction.  '+\
                                     'Try using the *closest=False* keyword to test equality '+\
                                     'rather than minimization.',\
                                     exception=False)
                    raise
                from numpy.lib.index_tricks import unravel_index
                indices=unravel_index(diff.argmin(),self.shape)
                coord_groups=[tuple([self.axes[i][indices[i]] \
                                     for i in range(len(indices))])]
            
            ###Wrap into dictionaries with axis names###
            if dicts:
                axis_names=self.get_axis_names()
                return [dict(list(zip(axis_names,coord_group))) for coord_group in coord_groups]
            else: return coord_groups
            
        def has_set_axes(self):
            
            return hasattr(self,'_axes_')
            
        def has_set_axis_names(self):
            
            return hasattr(self,'_axis_names_')
            
        def get_axes(self):
            
            
            if self.debug: Logger.write('Getting axes for self: id %i'%id(self))

            ##Pre-prepare default axes##
            axes=[numpy.arange(self.shape[i]) for i in range(self.ndim)] #Start with default entries
            
            ##Now recall set axes if available##
            if hasattr(self,'_axes_'):
                if self.debug: Logger.write('Pre-set axes were found, using them.\n'+\
                                            trace())
                set_axes=getattr(self,'_axes_')
                for i in range(len(set_axes)):
                    ##Break if we've exceeded dimensions##
                    if (i+1)>self.ndim: break
                    ##Axis is valid to use##
                    elif len(set_axes[i])==self.shape[i]: axes[i]=set_axes[i]
                    elif self.debug: Logger.write('Could not continue using axis in position '+\
                                                  '%i, did not match current shape: %i!=%i'%\
                                                  (i,self.shape[i],len(set_axes[i])))
                    
            elif self.debug: Logger.write('No axes found! Using defaults.\n'+\
                                          trace())
            
            ##If any changes have been made to axes, store them##
            self._axes_=axes
                    
            return copy.deepcopy(axes) #Don't enable direct access to original cache
        
        def get_axis_names(self):
            
            ##Pre-prepare default axis names##
            axis_names=['axis%s'%dim for dim in range(self.ndim)]
            
            ##Now recall set axis names if available##
            if hasattr(self,'_axis_names_'):
                set_axis_names=getattr(self,'_axis_names_')
                for i in range(len(set_axis_names)):
                    ##Axis is not valid, ignore it##
                    if not isinstance(set_axis_names[i],str): continue
                    ##Or axis is valid##
                    axis_names[i]=set_axis_names[i]
            
            ##If any changes have been made to axes, store them##
            self._axis_names_=axis_names
            
            return copy.deepcopy(axis_names) #Don't enable direct access to original cache
        
        def get_axis_limits(self):
            
            axes=self.get_axes()
            return [(min(axis),max(axis)) for axis in axes]
        
        def coordinates_to_slices(self,*coordinates,**kwargs):
            
            ##Process keyword arguments##
            if 'axes' in kwargs: axes=kwargs['axes']
            else: axes=self.get_axes()
            if 'squeeze' in kwargs: squeeze=kwargs['squeeze']
            else: squeeze=True
            if 'inclusive' in kwargs: inclusive=kwargs['inclusive']
            else: inclusive=True
            if 'axis' in kwargs:
                ##If axis is provided as name, convert to dimension##
                axis=kwargs['axis']
                if isinstance(axis,str):
                    axis_names=self.axis_names
                    try: axis=axis_names.index(axis)
                    except ValueError: Logger.raiseException('If provided as a string label, *axis* must be one of %s.'%repr(axis_names),\
                                                        exception=ValueError)
                else: axis=axis%self.ndim
                ##Check that we only have one coordinate entry##
                Logger.raiseException('Too many coordinates provided for slicing along a single axis.',\
                                 unless=len(coordinates)==1,exception=IndexError)
                
            else:
                ##Check that we don't have too many coordinate entries provided##
                Logger.raiseException('Too many coordinates provided for the number of dimensions.',\
                                 unless=len(coordinates)<=self.ndim,exception=IndexError)
                axis=None
            
            ##Take coordinate, or pair of coordinates, and convert to an index along an axis##
            def coord_to_index(coord,axis_array):
                
                if coord is None: return None
                Logger.raiseException('All coordinate entries must be either numeric or *None*.',\
                                      unless=(type(coord) in number_types), exception=TypeError)
                
                #####Find most appropriate index#####
                return numpy.abs(axis_array-coord).argmin()
            
            ##Iterate over dimensions##
            slices=[slice(None) for dim in range(self.ndim)] #Default to all-encompassing slice entries
            for i in range(len(coordinates)):
                
                coord_entry=coordinates[i]
                #If we want default, we've already captured it#
                if coord_entry is None: continue
                
                ##If an *axis* for *coordinates* is specified, apply only to that dimension##
                if axis!=None: dim=axis
                ##If *axis* is not supplied, assume each element of *coordinates* applies to next dimension##
                else: dim=i
                
                #Try to treat coord entry as arguments to a slice
                try:
                    ##If we actually got a slice object, break it down##
                    if isinstance(coord_entry,slice): coord_entry=(coord_entry.start,\
                                                                   coord_entry.stop,\
                                                                   coord_entry.step)
                    
                    #Bail if we don't meet conditions for specifying a slice#
                    if not hasattr(coord_entry,'__len__'): raise TypeError
                    elif len(coord_entry)==1:
                        coord_entry=coord_entry[0]
                        raise TypeError
                    elif len(coord_entry) not in [2,3]:
                        Logger.raiseException('If indicating a coordinate slice, provided coordinate sets must be of length 2 or 3 only.',\
                                              exception=IndexError)

                    #Convert first two entries to indices from coordinates
                    slice_indices=[coord_to_index(coord_entry[j],axes[dim]) for j in range(2)]
                    
                    #Add bonus index to upper limit to enforce inclusive upper limit when#
                    #an empty set would otherwise result##
                    #This differs from traditional index slicing, but is required for some applications
                    #of coordinate slicing#
                    #if slice_indices[0]==slice_indices[1]!=None:
                    #    ##But we require that provided lower limit is greater than upper
                    #    if None in coord_entry: slice_indices[1]+=1
                    #    elif coord_entry[1]>coord_entry[0]: slice_indices[1]+=1
                    if slice_indices[1]!=None and inclusive: slice_indices[1]+=1
                        
                    #If a third entry is present, interpret as sampling step#
                    if len(coord_entry)==3: slice_indices.append(coord_entry[2])
                    
                    #Add slice to list#
                    slices[dim]=slice(*slice_indices)
                    
                #Treat coord entry as a single coordinate
                except TypeError:
                    index=coord_to_index(coord_entry,axes[dim])
                    
                    ##If we are retrieving a squeezed slice, return single index##
                    if squeeze: slices[dim]=index
                    ##Or take a single-index slice, without reducing the number of dimensions##
                    else: slices[dim]=slice(index,index+1)
            
            ##It's important that sequence of slices be tuple##
            #it leads to interpretation of elements as pertaining to
            #different dimensions, whereas list signifies only first dimension,
            #for *numpy.ndarray.__getitem__*.
            return tuple(slices)
        
        def coordinate_slice(self,*coord_entries,**kwargs):
            
            slices=self.coordinates_to_slices(*coord_entries,**kwargs)
            
            return self.__getitem__(slices)
        
        def sort_by_axes(self):
            
            ##Acquire ordered axes and corresponding re-ordered indices##
            axes=self.get_axes()
            ordered_axes=[]
            ordered_index_sets=[]
            for axis in axes:
                axis_list=list(axis)
                ordered_axis,ordered_indices=misc.sort_by(axis_list,\
                                                          list(range(len(axis_list))))
                ordered_axes.append(numpy.array(ordered_axis))
                ordered_index_sets.append(numpy.array(ordered_indices))
                
            ##Now obtain a view of self that has corresponding re-ordering##
            ordered=self.view()
            for i in range(len(ordered_index_sets)):
                ordering_slice=[slice(None)]*i+[ordered_index_sets[i]]
                ordered=ordered.__getitem__(ordering_slice)
                
            ##Add ordered axes##
            ordered.set_axes(axes=ordered_axes,verbose=False)
            
            return ordered
        
        def integrate_axis(self,axis,integration=None):
            
             ##Make sure we have numeric axis##
            axis_names=self.get_axis_names()
            if isinstance(axis,str):
                Logger.raiseException('If a string, *axis* must be one of %s.'%axis_names,\
                                      unless=(axis in axis_names), exception=ValueError)
                axis=axis_names.index(axis) #Get a numeric axis
            else: 
                assert isinstance(axis,int), 'If not a string label, `axis` must be an integer axis number.'
                axis=axis%self.ndim
            
            ##Get default integration##
            if not integration:
                try: from scipy.integrate import trapz as integration
                except ImportError: Logger.exception('The module <scipy.integrate> is required for '+\
                                                     'this operation but could not be imported.')
            
            integral=self
            axes=integral.axes
            
            ##If there's nothing to integrate along axis, just remove dimension##
            if self.shape[axis]==1:
                limits=[None for i in range(integral.ndim)]; limits[axis]=0
                integral=get_array_slice(integral,limits)
            ##Else integrate
            integral=integration(integral,x=axes[axis],axis=axis)
            
            return integral
        
        def interpolate_axis(self,new_axis,axis,bounds_error=True,extrapolate=True,**kwargs):
            
            try: from scipy import interpolate
            except ImportError: Logger.exception('The module <scipy.interpolate> is required for '+\
                                                 'this operation but could not be imported.')
            
            ##Make sure we have numeric axis##
            axis_names=self.get_axis_names()
            if isinstance(axis,str):
                Logger.raiseException('If a string, *axis* must be one of %s.'%axis_names,\
                                      unless=(axis in axis_names), exception=ValueError)
                axis=axis_names.index(axis) #Get a numeric axis
            else: misc.check_vars(axis,int); axis=axis%self.ndim
            
            ##Sort self and identify current axis##
            sorted=self.sort_by_axes()
            old_axes=sorted.get_axes()
            current_axis=old_axes[axis]
            
            ##Interpolate##
            if not hasattr(new_axis,'__len__'):
                new_axis=[new_axis]; should_squeeze=True
            else: should_squeeze=False
            new_axis=numpy.array(new_axis)
            interpolator=interpolate.interp1d(x=current_axis,y=sorted,\
                                              axis=axis,bounds_error=bounds_error,\
                                              **kwargs)
            result=interpolator(new_axis)
            
            ##Fill in out-of-bounds values##
            if not bounds_error and extrapolate:
                bottom,top=sorted.axis_limits[axis]
                    
                where_below=(new_axis<bottom)
                where_above=(new_axis>top)
                
                if where_below.any():
                    limits=[slice(None) for i in range(sorted.ndim)]
                    limits[axis]=slice(0,1)
                    vec_below=sorted[limits]
                    limits=[slice(None) for i in range(sorted.ndim)]; limits[axis]=where_below
                    result[tuple(limits)]=vec_below
                
                if where_above.any():
                    limits=[slice(None) for i in range(sorted.ndim)]
                    limits[axis]=slice(sorted.shape[axis]-1,sorted.shape[axis])
                    vec_above=sorted[limits]
                    limits=[slice(None) for i in range(sorted.ndim)]; limits[axis]=where_above
                    result[tuple(limits)]=vec_above
            
            ##Apply new axes##
            new_axes=copy.copy(old_axes)
            #If `new_axis` is just a number coordinate, remove axis in anticipations#
            if hasattr(new_axis,'__len__'): new_axes[axis]=new_axis
            else : new_axes.pop(axis)
                
            output=ArrayWithAxes.__new__(type(self),result,axes=new_axes,axis_names=axis_names)
            
            if should_squeeze: output=output.squeeze()#.tolist()
            return output
        
        def interpolate_axes(self,coordinates,order=3,mode='nearest',**kwargs):
            
            from scipy.ndimage import map_coordinates
            
            #First take coordinates and map them to pixel values#
            sorted_self=self.sort_by_axes()
            index_coordinates=[]
            for i in range(self.ndim):
                
                axes_to_indices=AWA(range(sorted_self.shape[i]),\
                                      axes=[sorted_self.axes[i]])
                ind_coords=axes_to_indices.interpolate_axis(coordinates[i],\
                                                            axis=0,**kwargs)
                index_coordinates.append(ind_coords)
                
            return map_coordinates(sorted_self,index_coordinates,mode=mode,order=order)
        
        def plot(self,plotter=None,plot_axes=True,labels=True,**kwargs):
            
            ##Make sure we've only got finite values in axes##
            #We've got to obtain a view of *self* ordered by axes#
            #Otherwise we get artifacts when plotting with axes#
            to_plot=self.view()
            if plot_axes:
                to_plot=to_plot.sort_by_axes()
                axes=to_plot.get_axes()
                for i in range(len(axes)):
                    slice_entry=[slice(None)]*i+[numpy.isfinite(axes[i])]
                    to_plot=to_plot.__getitem__(slice_entry)
            
                ##Get Axes##
                axes=to_plot.get_axes()
                axis_names=to_plot.get_axis_names()
            
            if to_plot.ndim in [1,2]:
                try: from matplotlib import pyplot
                except ImportError:
                    Logger.raiseException('Plotting of 1- and 2-D coordinate slices is unavailable because '+\
                                          'the matplotlib module is unavailable!', exception=ImportError)
                        
                ##Look up plotter if provided##
                if isinstance(plotter,str): plotter=getattr(pyplot,plotter)
                    
                if to_plot.ndim==1:
                    #Prepare kwargs#
                    exkwargs=misc.extract_kwargs(kwargs,log_scale=False)
                    log_scale=exkwargs['log_scale']
                    
                    ##Default  plotter##
                    if plotter is None:
                        if log_scale: plotter=pyplot.semilogy
                        else: plotter=pyplot.plot
                    
                    if plot_axes:
                        result=plotter(axes[0],to_plot,**kwargs)
                        if labels:
                            try: result.axes.set_xlabel(axis_names[0])
                            except: pyplot.xlabel(axis_names[0])
                    else: result=plotter(to_plot,**kwargs)
                    
                elif to_plot.ndim==2:
                    
                    #Flip *to_plot* to reverse axes so that it plots contour with x,y axes appropriate
                    to_plot=numpy.asarray(to_plot); to_plot=to_plot.transpose()
                    
                    #Prepare kwargs#
                    exkwargs=misc.extract_kwargs(kwargs,lev=80,\
                                                        log_scale=False,\
                                                        labelOnlyBase=False,\
                                                        colorbar=True)
                    lev=exkwargs['lev']
                    log_scale=exkwargs['log_scale']
                    labelOnlyBase=exkwargs['labelOnlyBase']
                    colorbar=exkwargs['colorbar']
                    
                    ##Default  plotter##
                    if plotter is None: plotter=pyplot.imshow
                    
                    #Ensure proper aspect#
                    if plotter is pyplot.imshow and 'aspect' not in kwargs: kwargs['aspect']='auto'
                    
                    #Special level configuration for log scale#
                    if log_scale:
                        ##Determine explicit levels on basis of whether we're including only base 10 marks in colorbar##
                        log_vals=numpy.log(to_plot)/numpy.log(10.)
                        where_finite=numpy.isfinite(log_vals)
                        levmin,levmax=log_vals[where_finite].min(),log_vals[where_finite].max()
                        if labelOnlyBase: levmin,levmax=numpy.floor(levmin),numpy.ceil(levmax); lev=lev-lev%(levmax-levmin)
                        lev=10**numpy.linspace(levmin,levmax,int(lev+1))
                        
                        ##Set up log10 color scale and log10 colorbar
                        from matplotlib.colors import LogNorm
                        kwargs['norm']=LogNorm(10**levmin,10**levmax)
                            
                    ##Prepare colorbar if requested##
                    if colorbar:
                        cbarkwargs={}
                        if 'orientation' in kwargs: cbarkwargs['orientation']=kwargs.pop('orientation')
                        if 'ticks' in kwargs: cbarkwargs['ticks']=kwargs.pop('ticks')
                        
                        ##Special log scale options##
                        if log_scale:
                            if 'ticks' not in cbarkwargs:
                                cbarkwargs['ticks']=10**numpy.linspace(levmin,levmax,int(levmax-levmin+1))
                                
                            from matplotlib.ticker import LogFormatter
                            cbarkwargs['format']=LogFormatter(10,labelOnlyBase=labelOnlyBase)
                            
                    ##We will try to plot with axes if requested##
                    try:
                        if not plot_axes: raise ValueError
                        #We require special treatment for imshow#
                        if plotter.__name__=='imshow':
                            extent=[axes[0].min(),axes[0].max(),\
                                    axes[1].min(),axes[1].max()]
                            result=plotter(to_plot,extent=extent,\
                                            origin='lower',**kwargs)
                        #Also special treatment for 
                        elif plotter.__name__ in ('contour','contourf'):
                            result=plotter(*(axes+[to_plot,lev]),**kwargs)
                        #Guess at call signature
                        else:
                            result=plotter(*(axes+[to_plot]),**kwargs)
                            
                        #Axis labels#
                        if labels:
                            try:
                                result.axes.set_xlabel(axis_names[0])
                                result.axes.set_ylabel(axis_names[1])
                            except AttributeError:
                                pyplot.xlabel(axis_names[0])
                                pyplot.ylabel(axis_names[1])
                        if colorbar:
                            pyplot.colorbar(**cbarkwargs)
                    
                    ##We'll omit axes##
                    except:
                        raise
                        result=plotter(to_plot,**kwargs)
                        if colorbar: pyplot.colorbar(**cbarkwargs)
                
            elif to_plot.ndim==3:
                ##Default  plotter##
                if plotter is None: 
                    try: from enthought.mayavi import mlab
                    except ImportError:
                        Logger.raiseException('%s:\n'%type(self)+\
                                         '   Plotting of 3-D coordinate slices is unavailable because '+\
                                         'the <enthought.mayavi.mlab> module is unavailable.',\
                                         exception=ImportError)
                    plotter=mlab.contour3d
                    
                ##Set some plot defaults##
                if 'opacity' not in kwargs: kwargs['opacity']=.5
                if 'contours' not in kwargs: kwargs['contours']=10
                
                result=plotter(to_plot,**kwargs)
                    
            else: Logger.raiseException('The dimension of the resulting coordinate slice must be 1, 2, or 3.',\
                                        exception=IndexError)
            
            return result
        
        def to_txt(self,path,comment='',**kwargs):
            
            assert self.ndim<=2, "Can only write 1- or 2-dimensional datasets to `.txt`."
            
            newarr=numpy.zeros(self.shape)
            xaxis_addition=self.axes[0]
            header='First column: %s'%self.axis_names[0]
            
            if self.ndim==1: newarr=numpy.vstack((xaxis_addition,newarr))
            else:
                
                xaxis_addition.resize((len(xaxis_addition),1))
                newarr=numpy.hstack((xaxis_addition,newarr))
                
                yaxis_addition=numpy.array([None]+list(self.axes[1])).astype(numpy.float)
                yaxis_addition.resize((1,len(yaxis_addition)))
                newarr=numpy.vstack((yaxis_addition, newarr))
                
                header+='; First row: %s'%self.axis_names[1]
            
            if comment: header+='; '+comment
                
            numpy.savetxt(path,newarr,header=header,**kwargs)

    ########################
    #---AWA: An abbreviation
    ########################
    AWA=ArrayWithAxes

###If *numpy* classes could not be initialized, provide message###
except ImportError:
    
    Logger.raiseException('Trouble setting up classes in module <%s>:\n'%__module_name__+\
                     'The <numpy> module is required for base classes *ArrayWithAxes* '+\
                     'and *ArrayWithEmbeddedAxes*, but it was not found in the python path.',\
                     exception=False)
    
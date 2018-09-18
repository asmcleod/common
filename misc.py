###############################################
#ASM# module "misc" from package "Common" #ASM#
###############################################

"""
These are miscellaneous functions useful for a variety of python purposes.
"""

import sys
import os
import re
import time
import types
import copy
from .log import Logger
import operator

try: from numpy import ndarray,array
except ImportError:
    ##The absence of numpy is benign, we simply don't have an explicit *ndarray* type available##
    ##Set to the next best thing in case of later reference##
    ndarray=list
    array=list
__module_name__=__name__

def unpickle_legacy(filename):
    
    import pickle
    with open(filename,'rb') as file:
        #ASM 2018.09.18 - Changed encoding to `'latin1'` per the suggestion of
        # https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
        # This rectifies some issues with loading `ArrayWithAxes` objects with intact metadata
        d=pickle.load(file,encoding='latin1')
        
        #if isinstance(d,dict):
        #    d=dict([(key.decode(),d[key]) for key in d.keys()])
            
        return d

def is_interactive(): return sys.stdin.isatty() and sys.stdout.isatty()

def all_indices(a, b):
    """
    This function returns all indices in *a* where *b=a[indices]*.  *a* must be
    a list-like object with a .index() method.  This function does not use
    recursion, and is not for arbitrary iterables *a*, but instead provides
    speed.  For the former generalization, use *where_true* with an equality
    argument ("==").
    """

    indices = [a.index(b)]
    while True:
        try:
            from_index=indices[-1]+1
            newest_index=a.index(b,from_index)
            indices.append(a.index(b,from_index))
        except ValueError:
            return indices

def remove_item(index, lst):
    """
    This removes item *index* from list *lst* and returns the resulting list.
    This uses the list.remove() method but circumvents having to re-type the list
    instance.
    """
    check_vars([index, lst], [int, list])
    lst.remove(lst[index])

    return lst

def sort_by(*lists,**keykwarg):
    """
    This function will sort a group of lists based on the sorting of the first.
    This way, elements will coincide by index after sorting as well as before,
    and the specified iterables are returned as a list of the input iterables
    (in the same order).  Use the keyword *cmp* to use a specific comparison
    function in order to sort.

        *lists: input iterables (of dimension 1) to be sorted by indices according
            to *cmp* applied to the first iterable in *lists*.
        **key: a keyword, the comparator function to be used in sorting the first 
            iterable in *lists*
            DEFAULT: `operator.itemgetter(0)`, enforces usual sorting on items from
                      the first input iterable.
    """

    #Get comparison keyword#
    if 'key' in list(keykwarg.keys()):
        key=keykwarg['key']
    else: key=operator.itemgetter(0)

    #Check list of lists for correct form and element lengths
    for i in range(len(lists)):
        Logger.raiseException('Inputs must be iterable.',\
                    unless=(is_iterable(lists[i])),\
                    exception=TypeError)
        Logger.raiseException('Input iterables must be of the same length.',\
                    unless=(len(lists[i])==len(lists[i-1])),\
                    exception=IndexError)

    #zip into list of tuples
    zipped=list(zip(*lists))
    if zipped==[]: return [[]]*len(lists)
    zipped.sort(key=key)
    
    return list(zip(*zipped))

def randomize(*args):
    """Randomizes the elements of a set of lists in a coincident manner."""

    import random
    grouped_args=list(zip(*args))
    n_elems=len(grouped_args)
    indices=list(range(n_elems))
    #For each index, decide what its new randomized index will be
    randomized_indices=[indices.pop(random.randint(0,len(indices)-1)) \
                        for i in range(n_elems)]
    randomized_args=[grouped_args[index] for index in randomized_indices]
    #Ungroup the result
    return list(zip(*randomized_args))

def is_iterable(item,protect=[str]):
    """
    Returns a boolean value describing whether an input 
    *item* can be considered iterable. *protect* is a 
    list of types which should be explicitly considered
    non-iterable.  The default is *[str,unicode].
    Types *str* or *unicode* of length 1 are explicitly
    protected, as are items of type *dict*.  This is
    largely a helper function for larger functions
    *apply_to_array*, *where_true*, and other iterable
    functions.
    """
    
    #protect non iterables from iteration
    for type_to_protect in protect:
        if isinstance(item,type_to_protect): return False
    #specifically protect strings of length one, they will never trip TypeError just by indexing them over and over
    if isinstance(item,str) and len(item)==1: return False
    elif isinstance(item,type): return False
    elif isinstance(item,dict): return False
    try: iter(item); return True
    except TypeError: return False

def flatten(l, ltypes=(list, tuple)):
    
    ltype = type(l)
    try: l = list(l)
    except TypeError: l=[l]; ltype=list
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def apply_to_array(func,*inputs,**args):
    """
    This function applies a function *func* to a set of
    iterables (with the __len__ attribute) entry-wise,
    returning a result of the same shape as the inputs.
    Inputs can themselves be of the same shape, or else
    those lacking in the final dimension will be expanded
    by their inner-most entries into appropriately sized
    iterables.  For example, *inputs: [1,2,3], 5* is valid.
    Keywords may be supplied to the input function.
    Be advised, this function is not fast (especially for
    hugely nested iterables) due to the need to perform
    type-checking at each "layer".  It is, however,
    general and flexible.
    
        *func: function to apply to the input lists, entry-wise.
               additional arguments may be provided to the 
               function using keywords.
        *inputs: input iterables (of the same shape, exempting
                 the final-dimension scenario explained above) 
                 supplied as a sequence of inputs.  *func* must 
                 be tailored to handle the appropriate number of 
                 inputs.
        *args: keyword arguments to *func*.
        
    Reserved Keywords:
    
        The keyword *index* is reserved, and passing its value
        to *func* is attempted for each entry when calling this
        function.  It contains a dictionary of the current entry's
        index (e.g. {0:x,1:y,2:z,...}), and may be used freely by
        *func*.  This index is a dictionary to provide default
        protection from iteration should the output be further
        processed by the user with *apply_to_array*.
        
        The keyword *protect* is also reserved, and it
        can be passed as a list of types which will be considered
        non-iterable by application of the function.  This can
        be useful if the user seeks to protect a certain type,
        e.g. tuples, from application of the function on its
        entries.  Instead, the function will act on that type
        when it is found. Length 1 strings and type *type* are 
        always protected.
        DEFAULT: [str,dict,unicode]
        
    A Note on Function Evaluation:
        
        You may note some unexpected results from your supplied
        function when acting on nested iterables.  To get an
        idea of what to expect, note that this function tries to
        operate the function on all iterables it considers to be
        unprotected.  Sometimes, this results in a valid operation
        (raising no errors) of the sort you might not expect, so 
        take note when devising the appropriate function to use.
    """
    
    #####Set what types are protected from iteration#####
    if 'protect' in list(args.keys()):
        protect=args['protect']
    else:
        protect=[str,str]
    
    #####Set what unprotected iterable type should be used for forming output#####
    if tuple in protect: unprotected_type=list
    elif list in protect: unprotected_type=tuple
    else:
        unprotected_type=None
        ###Iterate through inputs to see what iterable is being passed
        for item in inputs:
            if type(item)==list: unprotected_type=list; break
            elif type(item)==tuple: unprotected_type=tuple; break
        if unprotected_type==None: unprotected_type=list ##default to list

    ####Want iterable list of inputs, not tuple#####
    #This transformation permits assignments to be made when expanding elements
    #doesn't impact protected-ness since only elements of *inputs* can themselves be protected
    inputs=list(inputs)

    ####Find maximum length of input lists####
    max_length=0
    for i in range(len(inputs)):
        if is_iterable(inputs[i],protect=protect):
            if len(inputs[i])>max_length:
                max_length=len(inputs[i])

    ####Any size mismatch between inputs should be expanded out####
    for i in range(len(inputs)): #inputs is a list of (possibly) iterables
        if not is_iterable(inputs[i],protect=protect):
            #Expand one of the arguments as a uniform unprotected iterable to match argument with length
            if max_length>=1:
                inputs[i]=unprotected_type([inputs[i]]*max_length)
    
    ####Check for uniform lengths after modifying lengths####
    for i in range(len(inputs)):
        if max_length>=1: Logger.raiseException('input iterables must be of the same dimension.',\
                                     unless=(len(inputs[i])==len(inputs[i-1])),\
                                     exception=IndexError)

    ####If *index* keyword is not initialized, make it as an empty protected type#####
    ##Prepare type for index##
    if 'index' not in list(args.keys()):
        args['index']=dict()
        
    ####If it was initialized but not dict type, the user naively applied it and it should be reset####
    elif type(args['index'])!=dict:
        args['index']=dict()

    ####Try to iterate into iterable sub-lists####
    output_list=[]
    try:
        ##Dig into input lists##
        ##Raise TypeError to apply *func* if length is zero - none of the following will successfully trigger TypeError##
        for j in range(len(inputs)):
            if len(inputs[j])==0 or not is_iterable(inputs[j],protect=protect): raise TypeError
            
        ##We need to get all the j'th elements from each of the input lists which are indexed by i##
        for i in range(len(inputs[0])): #inputs[0] is the first input, and we are going through its indices
            
            ##Update args to reflect new appended index (a layer deeper, at item i)
            ##Copy args, because we don't want to change original, need it to stay the same as we iterate over i
            new_args=copy.copy(args)
            #expand list of index entries
            new_index_list=list(new_args['index'].values())
            new_index_list.append(i)
            #pack back into dict type
            new_args['index']={}
            for j in range(len(new_index_list)):
                new_args['index'][j]=new_index_list[j]
            new_args['protect']=protect
            
            ##Prepare set of inputs from entries in list at index i##
            input_list=[]
            for j in range(len(inputs)):
                ##Don't want to be stuck in a loop with strings or some protected type, so only apply to unprotected type
                if not is_iterable(inputs[j],protect=protect): raise TypeError
                ##Now sub list j is ok to be indexed at point i
                input_list.append(inputs[j][i])
                
            output_list.append(apply_to_array(func,*input_list,**new_args))
            
        return unprotected_type(output_list) #Change result to the unprotected iterable type (in case list is protected in input)
    
    ####Cannot iterate further####
    except TypeError:
        ##Prepare argument set, protect is not used by function##
        if 'protect' in list(args.keys()): args.pop('protect') #remove protect, not used by func
        
        ##See if function can handle index as an argument##
        #Try to see if we can access expected input names - if not, AttributeError
        try: 
            if 'index' in func.__code__.co_varnames: 
                return func(*inputs,**args)
            ##If func can't handle *index* argument##
            else:
                raise AttributeError
        #Can't handle index as argument
        except AttributeError:
            args.pop('index')
            return func(*inputs,**args)

def remove_empties(inlist,protect=[str,str]):
    """
    Remove "empty" entries from an iterable (or nested
    iterables).  Returns a list or tuple of the same
    shape as the input containing no iterables of
    length 0 (like () or []) and no *None* values.
    *protect* is a list of types to protect from 
    iteration. See the function *iterable* for
    further description.
    """
    
        #####Set what unprotected iterable type should be used for forming output#####
    ##Parallels output formatting in *apply_to_array*, but only applies to single input list##
    if tuple in protect: unprotected_type=list
    elif list in protect: unprotected_type=tuple
    elif type(inlist)==tuple: unprotected_type=tuple
    else: unprotected_type=list
    
    ###Convert temporarily to list for ease of element deletion###
    inlist=list(inlist)
    
    ###Search for empties###
    ##Only do something if input is unprotected##
    if type(inlist) not in protect:
        i=0
        i_displacement=0
        max_i=len(inlist)
        while i<max_i:
            ###Use a displacement on the index to account for items we already removed###
            ##Places us backwards an index, i.e. the TRUE *next element* after deletion##
            index=i-i_displacement
            ##If we've found a *None* to remove##
            if inlist[index]==None: 
                del inlist[index]
                i_displacement+=1
            ##Only do something if element is iterable##
            elif is_iterable(inlist[index],protect=protect):
                if len(inlist[index])==0: #mark for removal
                    del inlist[index]
                    i_displacement+=1
                else: #outsource to forked process if we have another iterable
                    inlist[index]=remove_empties(inlist[index])
                    if len(inlist[index])==0:
                        del inlist[index]
                        i_displacement+=1
            i+=1
            
    return unprotected_type(inlist) #Save in what we hope was original iterable form

def where_true(condition, *inputs, **args):
    """
    This function returns a list of indices where the entries of two
    (or more) iterables of the same shape when compared element-wise yield
    in a conditional return True values.  Optionally, the entries
    corresponding to those indices can be returned instead if the keyword
    *values* is set to *True*.  The conditional must be a string whose syntax
    can be evaluated by the python interpreter using builtin functions.

    The conditional can be evaluated with a single iterable or with multiple.
    If using multiple iterables, enter a list of them: e.g. [list1,list2,...].
    The first unique alphanumeric in the conditional will be taken to stand for
    elements of the first list, the second for those of the second list, etc.
    Python reserved words are protected, as are lexical expressions.

    Note: strings should NOT be treated like iterables in the context of this
          function.  This is due to an unavoidable trade-off in the function's
          input interpretation.

        *condition: a string which evaluates to a true or false expression when
                    elements of iterables in *inputs* are passed consecutively for
                    successive alphanumeric variable names found in the conditional
                    string. Such an alphanumeric names can be omitted if no entry
                    references are required, e.g.:
                    condition='**2>5' --> condition='entries["input_1"]**2>5'.
                    NOTE: only builtin functions can be evaluated in the conditional.
        *inputs: input iterable, or a list of input iterables.  If a non-iterable
                is included, it is expanded into an iterable of appropriate length.
        *values: toggle to True to return the conditionally true values of the
                    first iterable in *inputs* rather than its indices.
                    DEFAULT: False
        *verbose: toggle to True to see what conditional statement will be used to
                  test truth outcomes for the entries of *inputs* based on user input.
                  DEFAULT: False
        *remove: toggle to True to remove returned *None* entries, corresponding to
                 False values in the conditional.
                 DEFAULT: True
        *booleans: toggle to True to return boolean entries indicating the result
                   of the conditional statement applied to each entry in turn.
        *protect: a list of types which are to be protected from iteration. Entries
                  of these types will be compared wholesale when found instead of
                  expanded.
                  DEFAULT: [dict,str,unicode]

    Examples:
        >>>a=[[5,2],[6,1]]
        >>>b=[[2,3],[4,1]]
        >>>where_true('a+1>3 or a==1',a)
        [[0,None],[None,1]]
        >>>where_true('/2.>=2.5',a,values=True)
        [[5,None],[6,None]]
        >>>where_true('a>b or b==a+1',a,b,values=True)
        [[5,2],[6,None]]
    """

    #####Verify input variables#####
    Logger.raiseException('The first input must be a string representing a conditional statement.',\
                unless=(type(condition)==str),exception=True)
    
    #####Transform list of input iterables into list, doesn't affect protection#####
    inputs=list(inputs)
    
    #####Identify keywords#####
    if 'values' in list(args.keys()): values=args['values']
    else: values=False
    if 'coordinates' in list(args.keys()): coordinates=args['coordinates']
    else: coordinates=False
    if 'verbose' in list(args.keys()): verbose=args['verbose']
    else: verbose=False
    if 'remove' in list(args.keys()): remove=args['remove']
    else: remove=True
    if 'booleans' in list(args.keys()): booleans=args['booleans']
    else: booleans=False
    if 'protect' in list(args.keys()): protect_kw={'protect':args['protect']}
    else: protect_kw={'protect':[str,str]}
    protect=copy.copy(protect_kw['protect'])
    protect_kw['protect']+=[dict]

    #####Determine what variables are employed in the condition#####
    ####First need a list of built-in keywords so as not to confuse string literal with them####
    keywords=['and','assert','break','class','continue','def','del',\
              'elif','else','except','exec','finally','for','from','global',\
              'if','import','in','is','lambda','not','or','pass','print',\
              'raise','return','try','while']

    ####Filter the condition for keywords####
    filtered_condition=condition.strip()
    for keyword in keywords:
        filtered_condition=filtered_condition.replace(' '+keyword+' ',' ') #removes all whitespace-surrounded words
        #remove leading words
        if filtered_condition[:len(keyword)+1]==keyword+' ':
            filtered_condition=filtered_condition[len(keyword)+1:]
        #remove trailing keywords
        if filtered_condition[-len(keyword)+1:]==' '+keyword:
            filtered_condition=filtered_condition[-(len(keyword)-1)]

    ####Search for possible variable names in filtered condition####
    var_names=[]
    for i in range(len(inputs)):
        try:
            #A match for a variable must not be preceded by ., ', or any \w character,
            #nor can it be followed by a ., ', (, or any \w character
            match_string='(?<!\.|\'|\w)[a-zA-Z_][a-zA-Z0-9_]*(?!\(|\'|\w)'
                        #Note: \w is any alpha-numeric character and the underscore
            new_var_name_exp = re.compile(match_string)
            new_var_name=new_var_name_exp.findall(filtered_condition)[0]
            var_names.append(new_var_name)
            #replace all true matches for this new variable name with empty space
            #this way we won't worry about matching it again
            exp_to_replace='(?<!\.|\'|\w)'+new_var_name+'(?!\(|\'|\w)'
            filtered_condition=re.sub(exp_to_replace,'',filtered_condition)
        ###No variable was found
        except IndexError:
            ##Since just one input, assume a reference to it precedes condition
            if len(inputs)==1:
                condition = 'placeholder' + condition
                var_names.append('placeholder')
            else:
                #Prepare a text for error message
                error_text=''
                if var_names!=[]: error_text+='Variables were interpreted in the following way:\n'
                for j in range(len(var_names)):
                    error_text+='entries[\'input_'+str(j+1)+'\'] --> '+var_names[j]+'\n'
                #Send error message
                Logger.raiseException(error_text+'\nNo reference to input '+str(i+1)+' found in conditional:\n'+'\''+condition+'\'',\
                            exception=True)

    #####Replacing used place-holders with fixed variable names from input dictionary#####
    new_condition=condition
    for i in range(len(var_names)):
        #This uses look-ahead and look-back assertions to prevent the variables from being part of a larger variable phrase#
        to_replace_exp = re.compile('(?<![a-zA-Z0-9_])' + var_names[i] + '(?![a-zA-Z0-9_])')
        #replace with input names that will be recognized by our boolean function
        new_condition = re.sub(to_replace_exp,'entries[\'input_'+str(i+1)+'\']', new_condition)

    #####Verbose Info#####
    if verbose==True:
        print('Variables were interpreted in the following way:')
        for i in range(len(var_names)):
            print('entries[\'input_'+str(i+1)+'\'] --> '+var_names[i])
        print()
        print('The following conditional was used in evaluating where True:')
        print(new_condition)
        print()

    #####Defining true/false function based on condition#####
    def boolean_func(entries): #must be passed a dictionary whose keys match variables in the new_conditional
        return eval(new_condition) #evaluation expects a dictionary with keys 'input_1','input_2',etc.

    ####Make a function to broadcast input iterable elements into structures of the above kind,
    #These can be evaluated by boolean_func#####
    def make_readable_structures(*input_elements):
        readable_struct={}
        for i in range(len(input_elements)):
            readable_struct['input_'+str(i+1)]=input_elements[i] 
        return readable_struct

    ####Make a synthesized array of readable structures#####
    synthesized_array=apply_to_array(make_readable_structures,*inputs,**protect_kw)
    
    ####Evaluate this synthesized array by element with conditional####
    boolean_array=apply_to_array(boolean_func,synthesized_array)
    if booleans==True: return boolean_array
    
    ####If we seek values where true####
    if values==True:
        def retrieve_value_where_true(bool_value,first_input):
            if bool_value: return first_input
            else: return None
            
        output=apply_to_array(retrieve_value_where_true,boolean_array,inputs[0],**protect_kw)
            
    ####If we seek indices where true####
    elif coordinates==True:
        def retrieve_index_where_true(bool_value,index):
            if bool_value: return index #index is a dictionary, get last element
            else: return None
            
        output=apply_to_array(retrieve_index_where_true,boolean_array,**protect_kw)
    else:
        def retrieve_index_where_true(bool_value,index):
            if bool_value: return index[list(index.keys())[-1]] #index is a dictionary, get last element
            else: return None
            
        output=apply_to_array(retrieve_index_where_true,boolean_array,**protect_kw)
        
    ####If we need to weed out None values####
    if remove==True:
        output=remove_empties(output,protect)
        
    return output

def unpack_list(lst,protect=[str,str]):
    """
    This will unpack all the entries of a list (of lists, etc.) into a single list of
    components which do not have the len() attribute.  Exceptions may be provided with
    the keyword *protect*, which may be set to a list of types which are to be
    protected from unpacking.  The default is *protect=[str,unicode]*.
    """

    #####Set what unprotected iterable type should be used for forming output#####
    ##Parallels output formatting in *apply_to_array*, but only applies to single input list##
    if tuple in protect: unprotected_type=list
    elif list in protect: unprotected_type=tuple
    elif type(lst)==tuple: unprotected_type=tuple
    else: unprotected_type=list

    unpacked_list = []

    #Try to unpack, if it is not exempt but it has no length, it will not be unpacked
    #but will terminate
    try:
        ##Only look at non-protected items to unpack
        if is_iterable(lst,protect=protect):
            for i in range(len(lst)):
                ##Issue again on entries if we have a list##
                to_add_on = unpack_list(lst[i],protect=protect) #another branch going down
                #From here, if lst[i] is not a list, we end the branch.

                #Otherwise results are passed up to this call by the following:
                for j in range(len(to_add_on)):
                    unpacked_list.append(to_add_on[j])

        #####Terminate at end of branch#####
        else: raise TypeError
    #####Terminate at end of branch due to object with no length#####
    except TypeError: 
        unpacked_list = [lst]

    return unprotected_type(unpacked_list) #return to original unprotected outer type

def check_vars(inputs,types,exception=TypeError,protect=None,verbose=True,subclass=False):
    """
    This function is a convenient checker to compare a set of variables
    with a set of types.  This could be desirable at the beginning of a
    new function, for example, to verify that inputs are properly formatted.
    
    Entries in the iterable *inputs* (a list of lists, array, etc.) are compared
    entry-wise with those in the corresponding iterable *types* to determine
    if the types match.  *inputs* and *types* must be of the same shape
    for a proper comparison to be performed.
    
    Alternatively, if for any entry in either *inputs* or *types*, one of these
    is missing a single dimension in comparison with the other, the former is
    expanded to match.  Therefore in the case that *types* is larger in one
    dimension than *inputs*, the entry in *inputs* need only match one of
    the type objects in that dimension of *types* to pass the comparison.
    This may be called a permissive comparison.
    The converse might be called a demanding comparison.
    The results of mixing permissive and demanding comparisons are undefined,
    so be wary.
    
        *inputs: a list of input objects, a list of lists, an array, or any
                iterable.
                
        *types: as above, except populated exclusively with types.
        
        *exception: set to True or an exception to class to throw that exception
                    in the event of a false outcome.
                    DEFAULT: None
                    
        *protect: a list of types to be protected from comparison expansion, whose
                  members will be added to the default list: all types listed in
                  *types*.
                  DEFAULT: all types listed in *types*
                  
        *subclass: allow an item in *inputs* to pass comparison as a subclass of
                   the corresponding type in *types*.
                   DEFAULT: False
    
    Examples:
        Simple comparison: 
            check_vars( [1,'a',[{},1.]], 
                        [int,str,[dict,float]] ) --> True
            check_vars( [1,'a',[{},1.]], 
                        [int,str,[dict,int]] ) --> False
        Permissive comparison:
            check_vars( [1,'a',[{},1.]], 
                        [int,str,[dict,[int,float]]] ) --> True
        Demanding comparison:
            check_vars( [[1,1.],'a',[{},1.]],
                        [int,str,[dict,float]] ) --> False
                        
    NOTE: Consider now that this scheme cannot compare list types if inputs
          are provided in a list (or list of lists).  This is because inputs
          of type *list* will be protected from expansion, preventing comparison.
          Instead, consider using either tuples OR lists to enclose *inputs*
          and *types*, while excluding the other from inclusion in these lists.
    """
    
    #####Find what to protect - we necessarily need to protect listed *types*#####
    if type(protect)==list: #if user specifies
        ###Check that we only have types###
        for item in protect:
            if type(item)!=type:
                Logger.raiseException('Provide only items of type *type* to protect.',exception=TypeError)
        users_protect=protect
    else: users_protect=None

    ###Check that we only have types in *protect* and no protected iterables###
    protect=list(unpack_list(types))
    for item in unpack_list(types,protect=protect):
        if type(item)!=type:
            Logger.raiseException('*types* can only be an iterable of objects of type *type* passed without nest protected iterables.\n'+\
                        'Protected type %s was found.'%type(item),\
                        exception=TypeError)

    ###Extend *protect* to include user's protected###
    if users_protect!=None:
        protect.extend(users_protect)
    ###Throw error if user is trying to check same type as they provide as iterable
    if type(types) in protect and hasattr(types,'__len__') and type(types)!=type:
        Logger.raiseException('Type %s cannot be protected or checked as a type when provided as an iterable.'%type(types),\
                    exception=TypeError)

    #####Get sequence of boolean comparison values#####
    try: #First time attempting to compare both inputs and types, could be index mismatch if inputs ill-formed
        if subclass==True: comparison='issubclass(type(input),input_type)'
        else: comparison='type(input)==input_type'
        match_list=where_true(comparison,inputs,types,booleans=True, protect=protect)
    except IndexError:
        Logger.raiseException('Inputs do not broadcast to expected sizes.  Check that input iterables are not ill-formed in their context.',\
                    exception=IndexError)
    match_list=unpack_list(match_list,protect=protect)
    
    #####Get full expanded list of indices#####
    full_indices_list=apply_to_array(lambda x,y,index: index, inputs, types,protect=protect)
    full_indices_list=unpack_list(full_indices_list)
    
    #####Get full list of inputs#####
    full_inputs_list=apply_to_array(lambda x,y: x, inputs, types, protect=protect)
    full_inputs_list=unpack_list(full_inputs_list,protect=protect) #protect is important here because we are dealing directly with input objects
    
    #####Get list of original input indices#####
    input_indices_list=apply_to_array(lambda x,index: index, inputs, protect=protect)
    input_indices_list=unpack_list(input_indices_list)
    
    #####Get list of original types indices#####
    types_indices_list=apply_to_array(lambda x,index: index, types, protect=protect)
    types_indices_list=unpack_list(types_indices_list)
    
    #####Identify indices of permissive and demanding comparisons#####
    permissive_indices=[]
    demanding_indices=[]
    for index in full_indices_list:
        ##Mark of a permissive comparison at index##
        if index not in input_indices_list: permissive_indices.append(index)
        ##Mark of a demanding comparison at index##
        if index not in types_indices_list: demanding_indices.append(index)
        
    #####Go through comparison results and identify match failures based on permissive/demanding/normal status at index#####
    result=True
    unmatched_item=None
    for i in range(len(match_list)):
        index=full_indices_list[i]
        match=match_list[i]
        ###If index is member of a demanding comparison, must be true###
        if index in demanding_indices:
            if match==False: result=False; unmatched_item=full_inputs_list[i]; break
        ###If index is member of permissive comparison, we have entered a permissive group###
        ##We need one member to be true##
        elif index in permissive_indices:
            ##Identify matches in permissive group##
            ##permissive group defined as all but last element of indices are identical##
            permissive_matches=where_true('match==match and index.values()[:-1]==indices.values()[:-1]',match_list,index,full_indices_list,values=True,protect=protect)
            if True not in permissive_matches: result=False; unmatched_item=full_inputs_list[i]; break
        ##Regular comparison
        else:
            if not match: result=False; unmatched_item=full_inputs_list[i]; break
    
    #####Send message if result is False and message is called for#####
    if verbose==True or exception!=False:
        Logger.raiseException('Input "%s" does not match prescribed types.'%repr(unmatched_item),\
                    unless=(result==True),\
                    exception=exception)

    return result

def extract_kwargs(kwargs,**default_kwargs):
    
    extracted={}
    for key in list(default_kwargs.keys()):
        if key in kwargs: extracted[key]=kwargs.pop(key)
        else: extracted[key]=default_kwargs[key]
        
    return extracted

def one_level_copy(item, verbose=False):
    """
    Completely copy an input dictionary *item* by its
    values using one-level object copying.  This function
    returns the created copy.
    """

    new_item = copy.copy(item)
    #update new item by updating its dictionary, which is MUTABLE
    try:
        #Get dictionary if the item itself is not a dictionary
        if type(new_item)==dict: new_item_dict=new_item
        else: new_item_dict=new_item.__dict__

    #Raised if item is not dictionary-like
    except AttributeError:
        Logger.raiseException('ALERT: Input must be a dictionary or have a __dict__ attribute. Try using the copy module instead.',\
                    exception=True)

    #can treat as dictionary, continue to copy
    else:
        #make copy of each element found in dictionary to assign each element a new location in memory
        for key in list(new_item_dict.keys()):
            try: new_item_dict[key] = copy.copy(new_item_dict[key])
            except:#Can't copy certain item for WHATEVER reason, they vary wildly
                if verbose==True:
                    print(sys.exc_info()[0],': ',sys.exc_info()[1],' --->')
                    print('\tALERT: Item ' + key + ' could not be copied')

    return new_item

def force_array(listlist, maxrows=None, maxcols=None, full=False, dtype=object):
    """
    This function will take a sequence (a list of lists) and force it into a
    2-D array form.  Missing row entries with respect to the longest row
    will be padded with None to indicate placeholders.

        *listlist: input list of lists
        *maxrows: maximum number of rows in resulting array
            Default: all rows are included
        *maxcols: maximum number of columns in resulting array
            Default: all columns are included
        *full: *True* or *False*, to set whether to force the resultant
                array to be completely full.  This excludes rows with
                entries of None.

    NOTE: Only returns an *ndarray* type object if the *numpy* module is
          available.  Otherwise, a list suitable for broadcasting into
          an *ndarray* type is returned.
    """

    #####Checking we have a list of lists#####
    Logger.raiseException('Item must have a "__len__" attribute.',\
                unless=('__len__' in dir(listlist)),\
                exception=True)
    newlist=[]
    for row in listlist:
        #protect strings and items without length attribute from attempted splitting
        if '__len__' not in dir(row) or type(row)==str:
            newlist.append([row])
        else: newlist.append(row)

    #####Get maximum row length#####
    num_columns = 0
    for row in newlist:
        if len(row) > num_columns: num_columns = len(row)

    #####Build suitable list#####
    new_newlist = []
    if maxrows is None: maxrows = len(newlist)
    if maxcols is None: maxcols = num_columns

    for i in range(maxrows): #iterate over desired number of rows
        row = newlist[i]
        new_row = []
        flag_None = 0

        for j in range(maxcols): #iterate over desired number of columns
            #Try getting a data entry at index j, if there is none return None
            try:
                #Try converting numbers to floats if data type is not str
                if dtype!=str:
                    try: row[j]=float(row[j])
                    except (TypeError,ValueError): pass

                new_row.append(row[j])
            except IndexError: new_row.append(None);flag_None = 1 #add None to list as a placeholder, flag that None has been entered

        #####Adding newest row depends on whether there are Nones and on force full#####
        if full==True:
            if flag_None == 0: new_newlist.append(new_row)
        else: new_newlist.append(new_row)

    ####Return an ndarray type if possible####
    if 'numpy' in list(sys.modules.keys()):
        array_2d = array(new_newlist,dtype=dtype) #any data type must be consistent with strings if string data is present
    else: array_2d=list(new_newlist)

    return array_2d

#####This expression will match any number I can think of, and will NOT match an invalid number#####
num_exp_text= '[\-\+]?'+\
              '(?:'+\
                  '(?:'+\
                      '(?:[0-9]{1,3}(?:,[0-9]{3})*,[0-9]{2})'+\
                      '|'+\
                      '[0-9]*'+\
                   ')'+\
                   '[0-9][.]?'+\
                   '|'+\
                   '[.](?=[0-9])'+\
               ')'+\
              '[0-9]*'+\
              '(?:[eE][\-\+]?[0-9]+)?'
num_exp=re.compile(num_exp_text)

def extract_listlist(fileobject,delimiter=None,literal_char='\"',comments=True,comment_char='#'):
    """
    This function will read through a file object and accumulate a sequence (list
    of lists) of all the numbers found in scientific notation.  The format is:
        [[numbers from 1st number row],
         [numbers from 2nd number row],
         [        ...etc....         ]]

        *fileobject: file object which to read, or an input string
        *delimiter: provide a string delimiter between fields.  If a field
                    contains any non-numeric data, the data is stored in the
                    final list as a string.  If the data is numeric, it is
                    stored in the final list as a float. If the value is None,
                    a global float search is performed and non-numeric data are
                    ignored.
                    DEFAULT: None
        *literal_char: provide a character that indicates a literal data value
                      protected from the presence of an internal delimiter. These
                      data will then be processed correctly.
                      DEFAULT: '"'
        *comments: True/False - if delimiter specified, indicates whether to
                     exclude comment lines from the final list of lists.  Comment
                     lines are identified by their lack of a delimiter character
                     (if a delimiter is specified) or the presence of '#' as the 
                     first non-white-space character.
                     DEFAULT: True
    """

    list_list = []

    if type(fileobject)==str: text=fileobject
    else: fileobject.seek(0); text=fileobject.read()
    line_list=re.split(re.compile('[\n\r]+'),text)

    for line in line_list:

        ##Not using delimiters, searching for floats##
        if delimiter==None:

            ##If trimming comments
            if comments==True:
                if line=='': continue
                if line.strip().startswith(comment_char): continue

            matches = num_exp.findall(line)
            float_matches=[]

            #Convert all matches to floats
            for number in matches:
                #commas are not interpreted by float(), remove them
                float_matches.append(float(number.replace(',','')))
            if float_matches != []: list_list.append(float_matches)

        ##Using delimiters, searching for floats and strings (and literals)
        else:

            ##If trimming comments, then require at least one delimiter character
            if comments==True:
                ##If no delimiter character in the line, it is clearly a comment and should be skipped
                if delimiter not in line: continue
                if line.strip().startswith(comment_char): continue

            matches=[]
            row_list=line.strip().split(delimiter)

            #Like a for loop, except we can push up the iteration
            i=0
            while i<len(row_list):
                field=row_list[i]

                #See if we have to complete a literal statement:
                if field.startswith(literal_char) and not field.endswith(literal_char):

                    start=i #Record the unterminated field index
                    i=i+1 #start search for literal termination with next element
                    while i<len(row_list):
                        field=row_list[i]
                        
                        #If field has a terminating literal character:
                        if field.endswith(literal_char) or (i+1)==len(row_list):
                            stop=i+1
                            #Join together all fields between start and stop indices
                            field=delimiter.join(row_list[start:stop])
                            break #break out of literal search
                        
                        #Otherwise, continue search for termination in while loop
                        else: i+=1

                #remove all completed literals#
                field=field.replace(literal_char,'')
                
                #Empty string is None#
                try: field=float(field)
                except ValueError:
                    if field=='': field=None
                matches.append(field)

                #Move ahead to next field
                i+=1

            list_list.append(matches)
    return list_list

def extract_array(fileobject, maxrows=None, maxcols=None, major='row', full=False,\
                  delimiter=None,literal_char='\"',comments=True,comment_char='#',dtype=object):
    """
    This function will read through a file object and accumulate a 2d array
    of (by default) all the numbers found in scientific notation.  By setting
    a definite delimiter character (and character for literal entries), this
    function will alteratively assemble the array with every delimited data
    element. The format is:
        array([[data from 1st number row],
               [data from 2nd number row],
               [       ...etc....       ]])
    Missing row entries with respect to the longest row will be padded with
    None as a placeholder.

        *fileobject: file object which to read, or an input string
        *maxrows: maximum number of rows in resulting array
            DEFAULT: all rows are included
        *maxcols: maximum number of columns in resulting array
            DEFAULT: all columns are included
        *major: 'row' or 'column', specifying row or column major array
        *full: 'yes' or 'no', to set whether to force the resultant
            array to be completely full.  This excludes rows with
            entries of None.
        *delimiter: provide a string delimiter between fields.  If a field
            contains any non-numeric data, the data is stored in the
            final array as a string.  If the value is None, normal
            float searching is performed and non-numeric data are
            ignored.
            DEFAULT: None
        *comments: True/False - if delimiter specified, indicates whether to
            exclude comment lines from the final list of lists.  Comment
            lines are identified by their lack of a delimiter character,
            or the presence of '#' as the first non-white-space character.
            DEFAULT: True
        *dtype: data type for the output array.  DEFAULTs to *object*,
            a useful form which converts numeric data into floats
            while preserving strings.
            DEFAULT: object
    """

    listlist = extract_listlist(fileobject,delimiter=delimiter,\
                                literal_char=literal_char,comments=comments,\
                                comment_char=comment_char)
    array_2d = force_array(listlist, maxrows=maxrows, maxcols=maxcols, full=full, dtype=dtype)
    if major == 'column': 
        ####Throw error if we don't have access to arrays####
        Logger.raiseException('Module *numpy* not available - column major form is unavailable at the current time.',\
                    unless=('numpy' in list(sys.modules.keys())),\
                    exception=True)
        array_2d = transpose(array_2d)

    return array_2d

def save_array(in_array,path='in_array.txt',delimiter='\t',literalchar='"',comment=None,append=False):
    """
    The inverse of the function *extract_array*, this function takes an array *in_array* 
    (may be an "MxN" list of lists) and saves it to file *path*.  The data is appended 
    to the file *path* if it exists.

        *in_array: array (or "MxN" list of lists) to be saved
        *path: a string path for the save file
                   DEFAULT: 'in_array.txt'
        *delimiter: a delimiting character used to separate data elements in the
                    save file
                    DEFAULT: ','
        *literalchar: a character to demarkate literal elements of data when their
                      string representation happens to include the delimiter character.
                      Beware that the elements' string representations do not contain
                      this literal character as well.
                      DEFAULT: '"'
        *comment: a comment string to be placed at the head of the save file
        *append: a boolean indicating whether to append data to file *path*
                 if it exists.
                 DEFAULT: True
    """

    ####Throw error if we don't have access to in_arrays####
    Logger.raiseException('Module numpy not available - this function is unusable at this time.',\
                unless=('numpy' in list(sys.modules.keys())),\
                exception=ImportError)

    #####Making sure we've got a contiguous in_array
    if type(in_array)==list:
        in_array=force_array(in_array,full='no')
    else: in_array=array(in_array)

    #####Getting data text#####
    text=''
    for i in range(len(in_array)):
        row=in_array[i]
        #Just in case our 'row' has no length, as for in_array is a row vector itself#
        if '__len__' not in dir(row): row=[row]
        for j in range(len(row)):
            item=row[j]
            ##Replace Nones with empty
            if item==None: text_item=''
            else: text_item=str(item)
            ##Set of literal data
            if delimiter in text_item:
                text_item=literalchar+text_item+literalchar
            ##Add data to text
            text+=text_item
            ##Add delimiter if we're not at the last element in the row
            if j<len(row)-1: text+=delimiter
        text+='\n'

    #####Add comment if applicable#####
    if comment!=None:
        comment_text=''
        for line in comment.splitlines():
            comment_text+='#'+line+'\n'
        text=comment_text+'\n'+text

    #####Check file existence to determine mode#####
    if path[0]=='/':
        directory='/'.join(path.split('/')[:-1])
        file_to_look_for=path.split('/')[-1]
    else:
        directory=os.getcwd()
        file_to_look_for=path
    if file_to_look_for in os.listdir(directory) and append==True: mode='a+'
    else: mode='w'
        
    #####Write to file#####
    output_file=open(path,mode)
    output_file.write(text+'\n')
    output_file.close()

    return
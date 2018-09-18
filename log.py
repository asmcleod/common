import os
import sys
import re
import types
import copy
import time
import tempfile
import logging
#Also expose everything from logging in this namespace,
#except those that are here overwritten
from logging import *

def extract_kwargs(kwargs,**default_kwargs):
    """This function pops a set of keywords off of a dictionary of
    keywords `kwargs`.  The keywords to pop are provided with 
    `**default_kwargs` defining the keyword names and their default
    values.  If a specified keyword is not found in `kwargs`, the
    default will be used.  Results in an extracted keywords dictionary."""
    
    extracted={}
    for key in list(default_kwargs.keys()):
        if key in kwargs: extracted[key]=kwargs.pop(key)
        else: extracted[key]=default_kwargs[key]
        
    return extracted

# _srcfile is used when walking the stack to check when we've got the first
# caller stack frame.
# This behavior is inherited from and complimentary to the logging module
if hasattr(sys, 'frozen'): #support for py2exe
    _srcfile = "log%s__init__%s" % (os.sep, __file__[-4:])
elif __file__[-4:].lower() in ['.pyc', '.pyo']:
    _srcfile = __file__[:-4] + '.py'
else:
    _srcfile = __file__
_srcfile = os.path.normcase(_srcfile)

##Redefine this function if sys._getframe is not defined##
if hasattr(sys,'_getframe'): _getframe=sys._getframe
else:
    def _getframe(depth=0):
        
        depth+=1 #We are already 1 frame deep
        try: raise RuntimeError
        except: tb = sys.exc_info()[-1]
        frame = tb.tb_frame
        for level in range(depth): frame=frame.f_back
        
        return frame
    
##This tool was largely lifted from a method of *Logger* in the *logging* module##
#We refer to their definition of "calling frame" as "external frame", as in, external to this module.#
def _get_external_frame():
    """
    Find the stack frame of the caller so that we can note the source
    file name, line number and function name. This function actually
    returns the frame itself for inspection.
    """
    f = _getframe()
    while hasattr(f, "f_code"):
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        if filename in  [logging._srcfile,_srcfile]: #This is the line that was changed, frames in this module will be disregarded
            f = f.f_back
            continue
        break
    return f

def _get_frame_depth(frame=None,external=True):
    """
    Get the absolute depth of the supplied frame, or
    the calling frame (or external frame, if specified).
    """
        
    ##Make sure we have a frame to deal with##
    if frame==None:
        if external: frame=_get_external_frame()
        else: frame=_getframe(1) #Up one to go outside this function
        
    at_end=False; depth=0
    while not at_end:
        try: frame=frame.f_back; depth+=1
        except AttributeError: at_end=True
        
    return depth

def _file_name_to_module_name(file_name):
    """Takes a file name/path (e.g. `"blah.py"`) and returns the module
    name to which it corresponds, if any.  If none, just returns the
    leaf file name with ".py" stripped off.
    
    This function is used to resolve file names reported in frame
    objects to their corresponding module names, as they appear in
    `sys.modules`."""
    
    if file_name.endswith('.pyc'): file_name=file_name[:-1] #Get to .py format
    
    try:
        import inspect
        return inspect.modulesbyfile[file_name]
    ##If error, we'll have to infer module name from file##
    except (ImportError,KeyError):
        file_name=os.path.split(file_name)[-1] #turn path into true file name
        if file_name.endswith('.py'): file_name=file_name[:-3] #strip off ".py"
        return file_name

def trace(frame=None,external=True,up=0,indent='  '):
    """
    Return a trace (in string form) of the function call.
    Useful to locate an event within an arbitrary scope
    heirarchy.
    
        *level: specify the frames "up" from this function
                call which to reveal in the output trace.
                DEFAULT: 1 (calling frame)
    """
    
    ##Make sure we have a frame to deal with##
    if frame==None:
        if external: frame=_get_external_frame()
        else: frame=_getframe(1) #Up one to go outside this function
        
    ##Move *up* as desired##
    for i in range(up):
        if hasattr(frame,'f_back'):
            frame=frame.f_back
    
    ##Get frames all the way up the stack##
    frames=[]
    while hasattr(frame,'f_back'):
        frames.append(frame); frame=frame.f_back
    frames.reverse()
        
    ##Use frame info to format a traceback##
    trace_str=''
    for i in range(len(frames)):
        ##Get relevant info from record##
        frame=frames[i]
        code=frame.f_code
        file_name,line_no,func_name=code.co_filename,frame.f_lineno,code.co_name
        #Try to resolve module name from filename#
        module_name=_file_name_to_module_name(file_name)
        trace_str+=indent*i+'<%s>: Line %i in "%s"\n'%(module_name,line_no,func_name)
    
    #Erase reference to code and frame#
    try: del frame; del code
    except NameError: pass
        
    if trace_str is '': trace_str='--No trace available.--'
    return trace_str

LogLevelNames=['debug',\
               'info',\
               'warning',\
               'error',\
               'critical',\
               'fatal']
LogLevelNumbers=[logging.DEBUG,\
                 logging.INFO,\
                 logging.WARNING,\
                 logging.ERROR,\
                 logging.CRITICAL,\
                 logging.FATAL]
LogLevelMap=dict(list(zip(LogLevelNames,LogLevelNumbers)))

def log_level_to_number(level):
    """Map a log level name to its number as defined in the
    `logging` module."""
    
    if isinstance(level,int): return level
    else: return LogLevelMap[level]

class FilterCondition(object):
    """This is an abstract base class for all filter condition objects."""
    pass

class LevelFilterCondition(object):
    
    def __init__(self,levels=None):
        
        if levels==None: levels=[]
        self.set_levels(levels)
        
    def set_levels(self,levels):
        
        self.levels=[]
        
        if levels=='all': levels=LogLevelNames
        
        for level in levels:
            if not isinstance(level,int):
                level=log_level_to_number(level)
            self.levels.append(level)
            
    def get_levels(self): return self.levels
    
    ##If record if of specified levels, it fails to go through##
    def __call__(self,record): return record.levelno not in self.get_levels()

class DepthFilterCondition(LevelFilterCondition):
    
    ##Will only apply for specified levels##
    def __init__(self,depth=0,levels='all'):
        
        self.set_depth(depth)
        return super(type(self),self).__init__(levels=levels)
    
    def set_depth(self,depth):
        
        frame=_get_external_frame()
        abs_depth=depth+_get_frame_depth(frame)
        self.depth=abs_depth
    
    def get_depth(self): return self.depth
    
    def __call__(self,record):
        
        if hasattr(record,'depth'): will_display=record.depth<self.get_depth()
        else: will_display=True
        
        return will_display or super(type(self),self).__call__(record)
        
class DisplayFilterCondition(LevelFilterCondition):
    
    def __init__(self,levels='all'): return super(type(self),self).__init__(levels=levels)
    
    def __call__(self,record):
    
        if hasattr(record,'display'): will_display=bool(record.display)
        else: will_display=True
        
        return will_display or super(type(self),self).__call__(record)
    
class DeluxeFilter(logging.Filter):
    
    def __init__(self,*args,**kwargs):
        
        ##Default attributes##
        self._latest_record=None
        self.clearFilterConditions()
        logging.Filter.__init__(self,*args,**kwargs)
        
        ##Debug option off by default##
        self.debugFilter(False)
        
    def filterConditions(self,*conditions):
        
        if not len(conditions): return self._filter_conditions
        else: self._filter_conditions=list(conditions)
        
    def addFilterCondition(self,condition): self._filter_conditions.append(condition)
    
    def removeFilterCondition(self,condition):
        
        if condition in self._filter_conditions:
            self._filter_conditions.remove(condition)
    
    def clearFilterConditions(self): self._filter_conditions=[]
        
    def debugFilter(self,boolean=None):
        
        if boolean!=None: self._debug=boolean
        else: return self._debug
        
    def getLatestRecord(self): return self._latest_record
        
    def filter(self,record):
        
        ##When deciding to filter, affix the calling frame and depth to record##
        external_frame=_get_external_frame()
        record.frame=external_frame
        depth=_get_frame_depth(external_frame)
        record.depth=depth
        
        ##Record this latest record##
        self._latest_record=record
        
        ##Evaluate filter conditions##
        conditions=self.filterConditions()
        evaluated=[]
        for condition in conditions:
            if hasattr(condition,'__call__'): evaluated.append(condition(record))
            else: evaluated.append(condition)
        
        ##We require all conditions to evaluate to True for display##
        will_display=True
        for item in evaluated: will_display*=bool(item)
        will_display=bool(will_display)
        
        if self.debugFilter() and not isinstance(sys.stdout,DeluxeLogger):
            print('Evaluated filter conditions:',evaluated)
            print('Calling depth:',depth)
            print('Will display:', will_display)
            
        return will_display

class FormatOperator(object):
    """This is an abstract base class for all format operator objects."""
    pass

class LevelFormatOperator(FormatOperator):
    
    def __init__(self,levels=None):
        
        if levels==None: levels=[]
        self.set_levels(levels)
        
    def set_levels(self,levels):
        
        self.levels=[]
        
        if levels=='all': levels=list(log_levels.values())
        
        for level in levels:
            if not isinstance(level,int):
                level=log_level_to_number(level)
            self.levels.append(level)
            
    def get_levels(self): return self.levels
    
    ##If record if of specified levels, it fails to go through##
    def __call__(self,record,message=None):
        
        if message is None: message=record.getMessage()
        
        if record.levelno in self.get_levels():
            return '%s: %s'%(record.levelname.upper(),message)
        else: return message
        
class LabelFormatOperator(FormatOperator):
    
    def __init__(self,label): self.set_label(label)
    
    def set_label(self,label): self.label=label
    
    def get_label(self): return self.label
    
    def __call__(self,record,message=None):
        
        if message is None: message=record.getMessage()
        
        label=self.get_label()
        if label: return '%s %s'%(str(label),message)
        else: return message

class TraceFormatOperator(FormatOperator):
    
    def __call__(self,record,message=None):
        
        if message is None: message=record.getMessage()
        
        ext_frame=_get_external_frame()
        trace_text=trace(frame=ext_frame)
        
        return '%s\n========TRACE FOR ABOVE MESSAGE:========\n%s\n\n'%(message,trace_text)
    
class TimeFormatOperator(FormatOperator):
    
    def __call__(self,record,message=None):
        
        if message is None: message=record.getMessage()
                
        return '========Time: %s========\n%s'%(time.ctime(),message)

class HeaderFormatOperator(FormatOperator):
    
    def __call__(self,record,message=None):
        
        if message is None: message=record.getMessage()
        
        ##By convention, add no header to "continued" (tabbed) messages##
        if message.startswith('\t'): return message
        
        ##Make a header that reads as "<module.class.method>:"##
        frame=_get_external_frame()
        code=frame.f_code
        file_name,func_name=code.co_filename,code.co_name
        
        ##Match filename to a module name##
        module_name=_file_name_to_module_name(file_name)
            
        ##Now we can check if function is likely a member of a class##
        #This will pick class names out from instancemethod
        #and classmethod objects by inspecting their arguments#
        locals=frame.f_locals; cls_name=None
        if 'self' in locals and hasattr(locals['self'],'__class__'):
            cls_name=locals['self'].__class__.__name__
        elif 'cls' in locals and hasattr(locals['cls'],'__name__'):
            cls_name=locals['cls'].__name__
            
        ##Supply text with a header##
        header_parts=[module_name]
        if cls_name!=None: header_parts.append(cls_name.lstrip('<').rstrip('>'))
        if func_name!='<module>': header_parts.append(func_name)
        message='<%s>:\n\t%s'%('.'.join(header_parts),message.replace('\n',\
                                                                      '\n\t'))
        
        #We didn't mean to add tabs to initial or ending newlines
        if message.startswith('\n\t'): message='\n'+message[2:]
        if message.endswith('\n\t'): message=message[:-2]+'\n'
        
        return message

class DeluxeFormatter(logging.Formatter):
    
    def __init__(self,*args,**kwargs):
        
        exkwargs=extract_kwargs(kwargs)
    
        self.clearFormatOperators()
        logging.Formatter.__init__(self,*args,**kwargs)
    
    def formatOperators(self,*operators):
        
        if len(operators): self._format_operators=list(operators)
        else: return self._format_operators
        
    def addFormatOperator(self,operator): self._format_operators.append(operator)
    
    def removeFormatOperator(self,operator):
        
        #If we get a particular operator, remove it#
        if operator in self._format_operators:
            self._format_operators.remove(operator)
        
        #If we get a type of operator, remove all such instances#
        elif isinstance(operator,type):
            operator_type=operator
            for operator in self._format_operators:
                if isinstance(operator,operator_type):
                    self._format_operators.remove(operator)
    
    def clearFormatOperators(self): self._format_operators=[]
        
    def format(self,record):
        
        message=logging.Formatter.format(self,record) #Format as superclass would
        
        ##Ensure trailing newline##
        if not message.endswith('\n'): message+='\n'
        
        ##Bail if record indicates not to format##
        if hasattr(record,'format') and not record.format: return message
                
        ##Call format operators##
        format_operators=self.formatOperators()
        for format_operator in format_operators:
            message=format_operator(record,message=message)
                
        return message

def _route_to_formatters_(method_name):
    
    def routed_method(self,*args,**kwargs):
        
        exkwargs=extract_kwargs(kwargs,strm=True,tee_strms=True)
        first_result=None
        
        ##Apply to stream formatter##
        if exkwargs['strm']:
            formatter_method=getattr(self.getFormatter(),method_name)
            result=formatter_method(*args,**kwargs)
            if first_result==None: first_result=result
            
        ##Apply to tee stream formatter##
        if exkwargs['tee_strms']:
            formatter_method=getattr(self.getTeeFormatter(),method_name)
            result=formatter_method(*args,**kwargs)
            if first_result==None: first_result=result
            
        return result
    
    return routed_method

def _route_to_filters_(method_name):
    
    def routed_method(self,*args,**kwargs):
        
        exkwargs=extract_kwargs(kwargs,strm=True,tee_strms=True)
        first_result=None
        
        ##Apply to stream##
        if exkwargs['strm']:
            filter_method=getattr(self.getFilter(),method_name)
            result=filter_method(*args,**kwargs)
            if first_result==None: first_result=result
        
        ##Apply to tee streams##
        if exkwargs['tee_strms']:
            tee_filter_method=getattr(self.getTeeFilter(),method_name)
            result=tee_filter_method(*args,**kwargs)
            if first_result==None: first_result=result
                
        return first_result
    
    return routed_method

class TeeStreamHandler(logging.StreamHandler):
        
    ###Here we dynamically redefine the class###
    #Wrap some methods so they direct to formatters#
    _apply_to_formatters_=['formatOperators','addFormatOperator',\
                           'removeFormatOperator','clearFormatOperators',\
                           'format']
    for method_name in _apply_to_formatters_:
        locals()[method_name]=_route_to_formatters_(method_name)
        
    #Wrap some methods so they direct to filters#
    _apply_to_filters_=['filterConditions','addFilterCondition',\
                        'removeFilterCondition','clearFilterConditions',\
                        'debugFilter','getLatestRecord','filter']
    for method_name in _apply_to_filters_:
        locals()[method_name]=_route_to_filters_(method_name)
        
    def __init__(self,strm,tee_strms=[sys.stdout],**kwargs):
        
        logging.StreamHandler.__init__(self,strm)
        self.tee_streams=[]
        for tee_strm in tee_strms: self.addTeeStream(tee_strm)
        
        ##Set formatters##
        #Some kwargs are interpreted as format kwargs#
        self.setFormatter(DeluxeFormatter())
        self.setTeeFormatter(DeluxeFormatter())
        
        ##Set filters##
        #Remaining kwargs are interpreted as filter kwargs#
        self.filters=[DeluxeFilter(**kwargs),DeluxeFilter(**kwargs)]
        
        ##Debug option off by default##
        self.debugFilter(False)
        
    def __str__(self):
        """Will print the handler's stream (NOT any tee streams)."""
        
        self.flush()
        stream=self.getStream()
        stream.seek(0)
        return stream.read()
        
    def handle(self,record):
        """
        Conditionally emit the specified logging record.

        Emission depends on filters which may have been added to the handler.
        Wrap the actual emission of the record with acquisition/release of
        the I/O thread lock. Returns whether the filter passed the record for
        emission.
        """
        
        ##Execute typical handling by passing to both normal and tee streams##
        first_result=None
        for strm_kwargs in [{'strm':True,'tee_strms':False},\
                            {'strm':False,'tee_strms':True}]:
            rv = self.filter(record,**strm_kwargs)
            if rv:
                self.acquire()
                try: self.emit(record,**strm_kwargs)
                finally:
                    self.release()
            if first_result==None: first_result=rv
            
        return first_result
        
    def emit(self,record,strm=True,tee_strms=False):
        
        ##Emit messages to whichever specified streams##
        if tee_strms:
            msg=self.format(record,strm=False,tee_strms=True)
            for tee_strm in self.getTeeStreams():
                if hasattr(tee_strm,'closed') \
                    and tee_strm.closed: continue
                tee_strm.write(msg)
        if strm: 
            msg=self.format(record,strm=True,tee_strms=False)
            stream=self.getStream()
            if not stream.closed: stream.write(msg)
        
    def seek(self,*args,**kwargs):
        
        for stream in self.getTeeStreams(): stream.seek(*args,**kwargs)
        self.getTeeStream().seek(*args,**kwargs)
        return self.getStream().seek(*args,**kwargs)
        
    def setStream(self,strm): self.stream=strm
    
    def getStream(self): return self.stream
    
    def addTeeStream(self,tee_strm): self.tee_streams.append(tee_strm)
    
    def getTeeStreams(self):  return self.tee_streams
    
    def getFormatter(self): return self.formatter
    
    def setTeeFormatter(self,formatter): self.tee_formatter=formatter
    
    def getTeeFormatter(self): return self.tee_formatter
    
    def getFilter(self): return self.filters[0]
    
    def getTeeFilter(self): return self.filters[1]

    ##Overload so that handler will flush tee streams as well##
    def flush(self):
        
        #for stream in self.getTeeStreams(): stream.flush()
        return logging.StreamHandler.flush(self)

def _route_to_handlers_(method_name):
    
    def routed_method(self,*args,**kwargs):
        
        exkwargs=extract_kwargs(kwargs,output=True,error=True)
        first_result=None
        
        ##Apply to output handler##
        if exkwargs['output']:
            handler_method=getattr(self.getOutputHandler(),method_name)
            result=handler_method(*args,**kwargs)
            if first_result==None: first_result=result
            
        ##Apply to error handler##
        if exkwargs['error']:
            handler_method=getattr(self.getErrorHandler(),method_name)
            result=handler_method(*args,**kwargs)
            if first_result==None: first_result=result
            
        return first_result
    
    return routed_method

class DeluxeLogger(logging.Logger):
    
    _extra_record_attributes={'display':True,'format':True}
    _max_namespace_records=1
    _max_error_records=1
    
    _apply_to_handlers_=['debugFilter',\
                         'seek','getLatestRecord',\
                         'filterConditions','addFilterCondition',\
                         'removeFilterCondition','clearFilterConditions',\
                         'formatOperators','addFormatOperator',\
                         'removeFormatOperator','clearFormatOperators']
    for method_name in _apply_to_handlers_:
        locals()[method_name]=_route_to_handlers_(method_name)
    
    def __init__(self,*args,**kwargs):
        
        ##Pick up default streams for default handler / other special kwargs##
        #2018.06.19 - Added `mode` kwarg to specify non-binary stream for python3
        exkwargs=extract_kwargs(kwargs,strm=tempfile.TemporaryFile(mode='w+'),\
                                       error_strm=tempfile.TemporaryFile(mode='w+'),\
                                       filter_levels='info')
        ##Initialize self##
        logging.Logger.__init__(self,*args,**kwargs)
        
        ##Set up tee handlers with their streams##
        output_handler=TeeStreamHandler(exkwargs['strm'],tee_strms=[sys.stdout])
        error_handler=TeeStreamHandler(exkwargs['error_strm'],tee_strms=[sys.stderr])
        error_handler.setLevel(ERROR)
        #Connect with handlers#
        self.addHandler(output_handler)
        self.addHandler(error_handler)
        
        #Naturally, error handler will take care of all error messages, output can filter them#
        self.addFilterCondition(LevelFilterCondition(levels=['error','critical']),\
                                output=True,error=False)
        
        ##Set up record for namespaces/exceptions##
        self.namespace_records=[]
        self.exceptions=[]
        
    def __str__(self):
        """Will print both the Logger's output and error handlers."""
        
        return 'Output Log Stream:\n'+\
                str(self.getOutputHandler())+\
                '\n\n'+'Error Log Stream:\n'+\
                str(self.getErrorHandler())
        
    ##We overload the base-level logging function to enable tapping of special keywords##
    #We insert special kwargs into *extra* container
    #The inherited *_log* machinery adds them to *LogRecord* instances
    def _log(self,level,msg,args,**kwargs):
        
        special_kwargs=extract_kwargs(kwargs,**self._extra_record_attributes)
        if 'extra' not in kwargs: kwargs['extra']={}
        kwargs['extra'].update(special_kwargs)
        
        #2018.06.19 - Added as appropriate use of `super` in python3.6
        return super()._log(level,msg,args,**kwargs)
        #return logging.Logger._log(self,level,msg,args,**kwargs)
    
    def handle(self,*args,**kwargs):
        
        logging.Logger.handle(self,*args,**kwargs)
    
    ##After all this toying, we have to overload *findCaller* so that 
    #it doesn't think the current module file qualifies as a caller##
    #2018.06.19 - Added keyword argument `stack_info` for compatibility with python3 version of `logging`
    def findCaller(self,stack_info=False):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = logging.currentframe()
        #On some versions of IronPython, currentframe() returns None if
        #IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        rv = "(unknown file)", 0, "(unknown function)"
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename in  [logging._srcfile,_srcfile]: #This is the line that was changed, frames in this module will be disregarded
                f = f.f_back
                continue
            rv = (filename, f.f_lineno, co.co_name)
            break
        return rv
    
    def write(self,*args,**kwargs):
        """An alias for *self.log(level,msg,**kwargs)*, exposing the ability
        of this logger to duck-type as a stream object."""
        
        #We basically emulate the code of *logging.Logger.log* in order
        #to maintain frame determination behavior
        #Throw away empty messages#
        args=list(args); msg=args.pop(0)
        if msg.isspace(): return
        level=extract_kwargs(kwargs,level='info')['level']
        level=log_level_to_number(level)
        if self.isEnabledFor(level): self._log(level, msg, args, **kwargs)
        
        self.flush()

    def writelines(self,lines,level='info', args=(), **kwargs):
        
        #We basically emulate the code of *logging.Logger.log* in order
        #to maintain frame determination behavior
        #Throw away empty messages#
        level=log_level_to_number(level)
        if self.isEnabledFor(level):
            for line in lines: self._log(level, msg, args, **kwargs)
            
        self.flush()
        
    def flush(self):
        """Flush the streams of all the log handlers that are instances
        of *logging.StreamHandler*."""
        
        for handler in self.handlers:
            if isinstance(handler,logging.StreamHandler): handler.flush()
    
    def setFilterDepth(self,depth=3,levels=None,output=True,error=True,strm=True,tee_strms=True):
        
        ##First gather all extant depth conditions from filters##
        condition_groups=[]
        if output:
            if strm: condition_groups.append(self.filterConditions(output=True,error=False,\
                                                                   strm=True,tee_strms=False))
            if tee_strms: condition_groups.append(self.filterConditions(output=True,error=False,\
                                                                        strm=False,tee_strms=True))
        if error:
            if strm: condition_groups.append(self.filterConditions(output=False,error=True,\
                                                                   strm=True,tee_strms=False))
            if tee_strms: condition_groups.append(self.filterConditions(output=False,error=True,\
                                                                        strm=False,tee_strms=True))
        
        ##Remove the extant depth filters##
        for condition_group in condition_groups:
            depth_conditions=[]
            for condition in condition_group:
                if isinstance(condition,DepthFilterCondition):
                    depth_conditions.append(condition)
            for depth_condition in depth_conditions:
                condition_group.remove(depth_condition)
                
        ##Finally add new depth filter where appropriate##
        self.addFilterCondition(DepthFilterCondition(depth,levels=levels),\
                                output=output,error=error,\
                                strm=strm,tee_strms=tee_strms)
    
    def getOutputHandler(self): return self.handlers[0]
    
    def getErrorHandler(self): return self.handlers[1]
    
    def getOutputRecord(self): return str(self.getOutputHandler())
    
    def getErrorRecord(self): return str(self.getErrorHandler())
    
    def setOutputStream(self,strm): return self.getOutputHandler().setStream(strm)
            
    def setErrorStream(self,error_strm): return self.getErrorHandler().setStream(error_strm)
    
    def getOutputStream(self): return self.getOutputHandler().getStream()
    
    def getErrorStream(self): return self.getErrorHandler().getStream()
    
    def logNamespace(self,namespace=None):
        
        ##If no explicit namespace, use locals in calling frame##
        frame=_getframe(1)
        if namespace==None: namespace=frame.f_locals
            
        ##Or check that namespace is OK##
        else:
            if not isinstance(namespace,dict):
                if hasattr(namespace,'__dict__'): namespace=getattr(namespace,'__dict__')
                else: raise ValueError('Namespace must be a dictionary instance \
                                        or have a "__dict__" attribute.')
                                        
        ##Log the namespace##
        self.namespace_records.append((time.ctime(),frame,namespace))
        #Trim to maximum length#
        for i in range(len(self.namespace_records)-\
                       self._max_namespace_records): self.namespace_records.pop(0)
                       
    def retrieveNamespace(self,index=-1): 
        
        record=self.namespace_records[index]
        return {'time':record[0],'frame':record[1],'namespace':record[2]}
        
    ##Overload this method to both report and store exception##
    def exception(self,*args,**kwargs):
        """Report and store a handled exception.  Gets exception information
        from provided *exc_info* and *frame* arguments.  If not provided,
        obtains this information from *sys.exc_info()* and from the calling
        frame.  Default log level is "error", but an explicit level can be 
        declare with keyword *level* (e.g. *level="critical"*)."""
        
        ##Get defaults values for namespace logging frame and log level##
        exkwargs=extract_kwargs(kwargs,frame=_getframe(1),#_get_external_frame(),\
                                       level='error',\
                                       exc_info=sys.exc_info())
        frame=exkwargs['frame']
        level=exkwargs['level']
        exc_info=exkwargs['exc_info']
        
        ##Log exception value if not null##
        exception=exc_info[1]
        if exception: 
            self.exceptions.append((time.ctime(),frame,exception))
            #Trim to maximum length#
            for i in range(len(self.exceptions)-\
                           self._max_error_records): self.exceptions.pop(0)
        
        ##Use inherited method - prints trace information##
        args=list(args)
        if len(args): msg=args.pop(0)
        else: msg='An exception of type %s was raised.'%type(exception)
        if exception:
            import traceback
            tb_lines=traceback.format_exception(*exc_info)
            msg+='\n\nOriginal Error:\n\n'+'\n'.join(tb_lines)
            
        self.write(msg,*args,**{'level':level})
        return msg
    
    def retrieveException(self,index=-1):
        
        record=self.exceptions[index]
        return {'time':record[0],'frame':record[1],'exception':record[2]}
        
    def raiseException(self,msg,unless=False,exception=None,level='error'):
        """
        Throw an error with *errormsg* if *unless* evaluates to False.
        This function can be used to assure *unless* evaluates to True
        before proceeding with an operation.
    
            *errormsg: string message to post on error
            *unless: must be True to avoid error
                        DEFAULT: False
            *exception: if True, raise RuntimeError when *unless* is False;
                        if False or None, only raise errors being handled
                        DEFAULT: None
        """
    
        ##If condition is unfulfilled, raise and log error##
        if not unless:
            
            ##Resolve exception type##
            if exception is True: exception=RuntimeError
            elif exception is None:
                #Try to get default exception from exception being handled#
                handled_type,handled_exception,handled_tb=sys.exc_info()
                #Use the exception type#
                if handled_exception: exception=handled_type
                #If there is nothing to handle, it would seem no exception was intended#
                else: exception=False
            
            ##If we actually are raising an exception, use *exception* method to log namespace and exception##
            if exception:
                msg=self.exception(msg,frame=_getframe(1),level=level)
                raise exception(msg)
            ##Otherwise, no need to write any misleading message message##
    
        return (not unless) #return opposite of unless -> True if error thrown, False if not
    
    def basicConfig(self):
        
        #Block no messages, just let handlers decide whether to display them#
        self.setLevel(logging.DEBUG)
        
        #-----Configure message formatting-----#
        ##Add format operators - the order matters!##
        ##In what order do we want to format the message?##
        #Add a level name indicator, excluding info level#
        self.addFormatOperator(LevelFormatOperator(set(LogLevelNames)\
                                                     -set(['info'])),\
                                  strm=True,tee_strms=True)
        #Show trace only for error messages#
        self.addFormatOperator(TraceFormatOperator(),output=False,error=True)
        #We want to display header everywhere
        self.addFormatOperator(HeaderFormatOperator(),strm=True,tee_strms=True)
        #Don't show time for tee streams
        self.addFormatOperator(TimeFormatOperator(),strm=True,tee_strms=False)
        
        #-----Configure message filtering-----#
        ##Some filter settings##
        self.setFilterDepth(6,levels=['info'],\
                              output=True,error=False,\
                              strm=False,tee_strms=True) #Info messages 4 or more frames deeper than this execution will not be tee'd
        #Enable display filtering only on tee streams#
        self.addFilterCondition(DisplayFilterCondition(),\
                                  output=True,error=True,\
                                  strm=False,tee_strms=True)
        ##Banish simple error messages from tee streams:##
        #If exceptions are handled, we wouldn't want error messages displayed.
        #If error is unhandled, it will get its own traceback and message anyway.
        self.addFilterCondition(LevelFilterCondition(levels=['error','critical']),\
                                  output=False,error=True,\
                                  strm=False,tee_strms=True)
        #Also, exclude debug messages from tee streams, just send to logs#
        self.addFilterCondition(LevelFilterCondition(levels=['debug']),\
                                  strm=False,tee_strms=True)


######################################
#-----Configure a default logger-----#
######################################
logging.setLoggerClass(DeluxeLogger)
Logger=logging.getLogger('common')
Logger.basicConfig()

##A function for testing depth and other log display options##
def test_log_at_depth(at_depth=1,depth=1,level='info',display=True):
    if depth<at_depth:
        depth+=1; return test_log_at_depth(at_depth,depth,level=level,display=display)
    else: Logger.write('a message!',level=level,display=display)
    
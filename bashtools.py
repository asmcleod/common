###################################################
#ASM# module "bashtools" for package "common" #ASM#
###################################################
"""
This module contains a number of functions written by Alex McLeod
which he considers to be useful.  They are all for shell-related
tasks, such as running a system process or finding a device.
Enjoy!
"""

import os
import copy
import locale
from .log import Logger
from .misc import check_vars,apply_to_array

class Process(object):
    """
    Instantiating this class executes the input argument string
    as an independent process whose output can be read.  The output
    is buffered in a temporary file which should not be visible in
    the file system on Unix platforms.
    """
    
    _processes_=[]
    max_procs=50
    
    @classmethod
    def add_process(cls,proc):
        
        check_vars(proc,cls,subclass=True)
        cls._processes_.append(proc)
        if len(cls._processes_)>cls.max_procs:
            top_process=cls._processes_[0]
            cls._processes_.remove(top_process)
    
    @classmethod
    def get_processes(cls,alive=True):
        
        if alive:
            processes=[]
            for proc in cls._processes_:
                if proc.is_alive(): processes.append(proc)
        else: processes=cls._processes_
        
        return processes
    
    @classmethod
    def kill_processes(cls,signal,\
                       children=True,\
                       recursive=True,\
                       verbose=False):
    
        for proc in cls.get_processes():
            proc.kill(signal,\
                      children=children,\
                      recursive=recursive,\
                      verbose=verbose)
    
    @classmethod
    def get_temp_file(cls):
        
        import tempfile
        
        try: file=tempfile.TemporaryFile()
        except OSError as exception:
            try:
                del cls._processes_[0]
                file=tempfile.TemporaryFile()
            except IndexError: raise exception
            
        return file
    
    def __init__(self,arg,**kwargs):
        
        import subprocess
        
        self.textoutput=''
        self.arg=arg
        #####We use our own output buffer because Pipes for output are limited in memory
        #and could cause a hang if the output is large.#####
        if 'output_buffer' in kwargs: self.output_buffer=kwargs.pop('output_buffer')
        else: self.output_buffer=self.get_temp_file()
        if 'error_buffer' in kwargs: self.error_buffer=kwargs.pop('error_buffer')
        else: self.error_buffer=self.get_temp_file()
        if 'input_buffer' in kwargs: self.input_buffer=kwargs.pop('input_buffer')
        else: self.input_buffer=self.get_temp_file()
        
        try: self.Popen=subprocess.Popen(arg,stdout=self.output_buffer.fileno(),
                                        stderr=self.error_buffer.fileno(),\
                                        stdin=self.input_buffer.fileno(),\
                                        shell=True,executable='/bin/bash',
                                        **kwargs); self.shell='/bin/bash'
                                        
        #/bin/bash may not be present, in this case use /bin/sh
        except OSError: self.Popen=subprocess.Popen(arg,stdout=self.output_buffer.fileno(),
                                                    stderr=self.output_buffer.fileno(),\
                                                    shell=True,executable='/bin/sh',\
                                                    **kwargs); self.shell='/bin/sh'
            
        ##Append process to global list of processes##
        type(self).add_process(self)
        
    def get_command(self): return self.arg
        
    def get_pid(self):
        
        return self.Popen.pid
    
    def code(self):
        """
        This returns the return code of the process, gives None if still is_alive.
        """
        return self.Popen.poll()
        
    def is_alive(self):
        """
        This returns a boolean indicating whether the process is still is_alive.
        """
        if self.code()==None: return True
        else: return False
        
    def wait(self):
        """
        This instructs the interpreter to wait for the process to complete.
        """
        if self.is_alive() is True: self.Popen.wait()
        else: pass
        
    def send_signal(self,signal,\
                    children=True,\
                    recursive=True,\
                    verbose=False):
        
        if not self.is_alive(): return True
        
        ##Kill children, then self##
        pids=[self.get_pid()]
        if children: pids+=get_child_processes(self.get_pid(),recursive=recursive)
        for pid in pids:
            if verbose: Logger.write('Sending signal %i to process "%s".'%\
                                     (signal,self.arg))
            try: os.kill(pid,signal)
            except OSError: pass #Perhaps it's already dead, or gone
        
    def kill(self,signal=2,\
             children=True,\
             recursive=True,\
             verbose=False):
        
        return self.send_signal(signal,\
                                children=children,\
                                recursive=recursive,\
                                verbose=verbose)
        
    def get_output_buffer(self): return self.output_buffer
        
    def get_output(self):
        """
        This returns STDOUT of the process.
        """
        self.output_buffer.seek(0)
        self.textoutput=self.output_buffer.read().strip() #remove extra white space
        
        return self.textoutput
        
    def get_error_buffer(self): return self.error_buffer
        
    def get_error(self):
        """
        This returns STDOUT of the process.
        """
        self.error_buffer.seek(0)
        return self.error_buffer.read().strip() #remove extra white space
        
    def get_input_buffer(self): return self.input_buffer
        
    def __str__(self):
        """
        Same as Process.output().
        """
        return self.get_output()
        
def bash_escape(filename,exclude=[]):
    """
    This function takes a string and escapes bash special characters.
    This is useful for e.g. specifying literal file names on the
    command line using *os.system* or other shell command calls.
    """
    
    check_vars(filename,[str,str])
    specialchars=[' ',']','[','{','}','(',')',\
                  '*','?','\'','`','&',';',',','"',\
                  '\n']
    
    for char in specialchars:
        if char not in exclude:
            filename=filename.replace(char,'\\'+char) #escape each special character (and escape for python)
        
    return filename

def get_child_processes(ppid,recursive=True):
    
    proc=Process("ps -o pid,ppid -ax | awk '{ if ($2==%s) print $1}'"%ppid)
    proc.wait()
    pids=set([int(pid) for pid in str(proc).splitlines()])
    
    ##Add recursive children as well if required##
    if recursive:
        top_pids=copy.copy(pids)
        for pid in top_pids: pids.update(get_child_processes(ppid=pid,recursive=True))
    
    return pids

###########################################################################################
def check_command(command):
    """
    Check to see whether *command*, in string representation,
    is a valid shell command in the present environment.
    """

    proc=Process('which '+str(command));proc.wait()
    if proc.code()!=0:
        Logger.raiseException('Command utility '+command+' not found in shell path.')
        print(str(proc))
        return False
    else: return True

def find_file(filename,source='.',all=False,sudo=False,pattern_chars=['*','?'],\
              exception=False,verbose=False):
    """
    Return a list of directory locations where file *filename*
    may be found below directory *source* (defaults to the
    present working directory).  This operation is facilitated
    with the Unix "find" command.
        *filename: string form of the filename for which to
                   search.
        *source: string form of the directory below which to
                 search.
                 DEFAULT: '.' (present working directory)
        *all: a boolean indicating whether or not to return
              all found locations.  If *False*, only the first
              result is returned as a single string.
              DEFAULT: False
        *exception: a boolean indicating whether to return an
                    exception on failure to find the specified
                    file.  If *True*, *RuntimeError* is raised.
                    Alternatively, an exception class can be
                    provided for a specific exception to be used.
                    DEFAULT: False
    """

    #####Expand '~' and relative paths####
    source=os.path.abspath(os.path.expanduser(source))
    filename=os.path.expanduser(filename)
                           
    find_command='find %s -iname "%s"'%(bash_escape(source),\
                                        bash_escape(filename,exclude=pattern_chars)) #exclude pattern 
    if sudo==True: find_command='sudo '+find_command
    
    proc=Process(find_command) #print found path to temp file
    proc.wait() #wait for process to complete
    if exception:
        Logger.raiseException('An error occurred finding file "%s":\n'%filename+\
                    'Command: "%s"\n'%proc.arg+\
                    'Result: "%s"'%proc.error(),\
                    unless=(proc.code()==0),exception=exception)
    
    ###Retreive output in lines###
    #We use *get_output* instead of *str*, since the latter may fail at decoding#
    lines=proc.get_output().splitlines() #piped process output
    findpaths=[]
    for line in lines:
        if 'Permission denied' not in line: findpaths.append(line.strip())
    
    if Logger.raiseException('File "%s" could not be found:\n'%filename+\
                   'Command: "%s"'%proc.arg,\
                    unless=(len(findpaths)!=0),exception=False): return None

    if all: return findpaths
    else: return findpaths[0]

def check_for_files(filenames,directories=None,verbose=True,samedirectory=False,exception=False):
    """
    This function checks to see if files in *filenames* are located in one
    of the directories supplied in *directories*.  Returns a dictionary of
    absolute file paths with provided filenames as keys.  If a file is not
    found in one of *directories*, its dictionary value is None.
    
    If an explicit file path is provided as a file name, the directories are
    not searched for its presence, but rather the provided path is tested
    for validity.  The corresponding dictionary value will be None if no such
    path exists.
    
        *filenames: a list of string filenames
        *directories: a list of absolute or relative directories through
            which to search.
            DEFAULT: current directory
        *samedirectory: set to True to require non-explicit path filenames
            in *filenames* to be found in the SAME directory among those in
            *directories.*  If all are not found in any one of these, None 
            is returned for all dictionary values.
            DEFAULT: False
        *verbose: set to False to suppress messages as to which files are
            missing, if any.
            DEFAULT: True
        *exception: set to True (or an exception class) to raise an exception
                    on any failure to find a file.
            DEFAULT: False
    """
    
    ####Check variable types#####
    if directories==None: directories=os.getcwd()
    if type(filenames)!=list: filenames=[filenames]
    check_vars(filenames,[[str,str]]*len(filenames))
    if type(directories)!=list: directories=[directories]
    check_vars(directories,[[str,str]]*len(directories))

    ####Expand directories#####
    directories=apply_to_array(os.path.expanduser,directories)
    directories=apply_to_array(os.path.abspath,directories)
    
    ####Prepare empty path list####
    filepaths=[]
    explicit=[]
    for file in filenames:
        ###We preserve explicit paths to be checked on their own###
        if file[0]=='/' or file[:2] in ['~/','./'] or file[:3]=='../':
            explicit.append(True)
            filepaths.append(os.path.abspath(os.path.expanduser(file)))
        else:
            explicit.append(False)
            filepaths.append(None)

    ####Check non-explicit path filenames by looping over directories####
    for directory in directories:
        ##Iterate over all files to see if they're found in directory
        for i in range(len(filenames)):
            if not explicit[i]:
                file=filenames[i]
                #We need to check for the file if its path is None (or if samedirectory has refreshed it to None)
                if filepaths[i]==None:
                    if os.access(os.path.join(directory,file), os.F_OK):
                        filepaths[i]=directory+'/'+file
                    
                #If we seek all in same directory, then break if we do not find one
                if filepaths[i]==None and samedirectory==True: break

        ##If they're required to be in the same directory, break if one lacks a match##
        if samedirectory==True:
            ##See if paths are unfilled still##
            unfilled=(None in filepaths)
            if unfilled:
                ####Prepare empty path list####
                for i in range(len(filepaths)):
                    if not explicit[i]: filepaths[i]=None

    ####Now check explicit paths#####
    for i in range(len(filepaths)):
        if explicit[i]:
            ##See if it's there##
            ##If not, set to None##
            if not os.access(filepaths[i],os.F_OK): filepaths[i]=None
        
    ####Verbose output for missing files####
    if verbose==True:
        error_text=''
        if samedirectory==False:
            for i in range(len(filepaths)):
                path=filepaths[i]
                filename=filenames[i]
                
                if path==None:
                    if explicit[i]:
                        error_text+='Explicit file path \"%s\" not found.'%filename+'\n'
                    else:
                        error_text+='File \"%s\" not found in directories:'%filenames[i]+'\n'
                        for directory in directories: error_text+=directory+'\n'
                    
        else:
            ##If files were not found in same directory, and not explicit, then unfilled=True##
            if unfilled:
                error_text+='These files not found in the same directory:\n'
                for i in range(len(filenames)): 
                    if not explicit[i]: error_text+=filenames[i]+'\n'
            ##Or if an explicit path wasn't found, we have a None anyway##
            elif None in filepaths:
                for i in range(len(filepaths)):
                    if explicit[i]:
                        if path==None:
                            path=filepaths[i]
                            filename=filenames[i]
                            error_text+='Explicit file path \"%s\" not found.'%filename+'\n'
                
        ###Throw error if we are using verbose and we have an error###
        Logger.raiseException(error_text,unless=(error_text==''),exception=exception)
   
    ####Prepare output dictionary####
    path_dict={}
    for i in range(len(filenames)):
        filepath=filepaths[i]
        if filepath[-1]=='/': filepath=filepath[:-1] #We don't want trailing "/" for reasons of convenience
        path_dict[filenames[i]]=filepath
        
    return path_dict

def recursive_copy_file(source,destination,verbose=True):
    """
    This function copies a file designated by the path *source* to the
    destination path *destination*.  If intermediate folders in the
    destination path do not exist, they are recursively created.  Returns
    the return code of the copy process.  If the code is None and no
    error message is thrown, then the file was already present at the
    destination.
    
        *source: string of the source path (must be to a file)
        *destination: string of the destination. Can be relative or
                      absolute.
        
        NOTE: actual copying is performed with the command "rsync"
    """
    
    argv=[source,destination]
    types=[[str,str],[str,str]]
    check_vars(argv,types)
    
    #####Get filename#####
    filename=source.split('/')[-1]
    
    #####Checking source#####
    proc=Process('ls '+bash_escape(source)); proc.wait()
    if proc.code()!=0: Logger.raiseException('Source file \"'+source+'\" does not exist.');return proc.code()
    
    #####Creating files if need be#####
    splitted=destination.split('/')
    
    for i in range(len(splitted)):
        pathcheck='/'.join(splitted[:i+1]) #builds up bits of the original path left to right
        if pathcheck=='': pass #possible if destination was /dir1/dir2 --> ['','dir1','dir2']
        else:
            proc=Process('ls '+bash_escape(pathcheck)); proc.wait()
            if proc.code()!=0:
                proc=Process('mkdir '+bash_escape(pathcheck))
                
    
    proc=Process('ls '+bash_escape(destination)+'/'+bash_escape(filename));proc.wait()
    returncode=proc.code()
    if returncode is 0:# and 'y' in verbose:
        if verbose==True: print('Destination file \"'+destination+'/'+filename+'\" already exists.')
        return None
    
    proc=Process('rsync '+bash_escape(source)+' '+bash_escape(destination)); proc.wait()
    if proc.code()!=0:
        print(str(proc))
    
    return proc.code()

def recurse_operation(function,base='.',verbose=False,*args,**kwargs):
    """
    Recurse an operation specified by execution of *function*
    under all sub-directories of *base*.  It is prudent and
    useful for *function* to be a file operation of some kind.
    
        *function: the function instance which to recurse
        *base: the base directory from which to begin recursion
               DEFAULT: '.' (present working directory)
        *verbose: a boolean setting whether to provide verbose
                  output.
                  DEFAULT: False
        *kwargs: any keywords to be provided to *function* during
                 operation.
    """
    
    if '__call__' not in dir(function): print('Please supply a valid function argument.');return
    
    #Expand base directory if special characters are used
    base=os.path.abspath(os.path.expanduser(base))
    try:
        os.chdir(base)
        if verbose==True: print('Operating in directory %s...'%os.getcwd())
    except OSError: Logger.raiseException('Not able to enter directory %s...'%base,showpath=False)
    
    #####Execute Function#####
    function(*args,**kwargs) #try keywords if possible
    
    #####Finding children directories#####
    directories=[]
    files=os.listdir(base)
    for file in files:
        if os.path.isdir(os.path.join(base,file)): directories.append(file)
    
    #####Recursing down or Exiting#####
    if directories==[]:
        if verbose==True: print('Branch stopped in directory '+os.getcwd()+'.')
    else:
        for directory in directories:
            new_base_path=os.path.join(base,directory)
            recurse_operation(function,new_base_path,verbose=verbose) #One child taken care of
            
    return

def recursively_list_files(rootdir):
    
    fileList=[]
    for root, subFolders, files in os.walk(rootdir):
        for file in files:
            fileList.append(os.path.join(root,file))
    return fileList
        
def find_device(label):
    """
    This function will return the path entry associated with the a device
    whose label contains the argument *label*.  Path retrieval is achieved
    using the 'df' and 'awk' commands via the shell.
    """
    
    check_vars(label,str)
    proc=Process('df | awk \'/'+label+'/ {print $6}\'')
    proc.wait()
    device_path=str(proc).splitlines()
    if len(device_path)==1: device_path=device_path[0]
    elif len(device_path)==0: device_path=None
    
    return device_path

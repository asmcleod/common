##################################################
#ASM# module "plotting" for package "common" #ASM#
##################################################
#TODO: Fix undo/redo comparison operations of PlotHistory
#TODO: enhance some matplotlib functions
"""
This module assists in many matplotlib related tasks, such as managing plot objects.
It automatically imports all from matplotlib.pylab as well as from numpy.

Note: If variable *plot_format* is set (in dict __main__._IP.user_ns) to
    a valid matplotlib.use() format, e.g. 'PS', this will be implemented.
    Otherwise, the user will be prompted for the plot backend.
    
    Setting plot_format=None will bypass this behavior and use the default
    renderer.
"""

#_________________________________________Imports_________________________________________

import os
import sys
import re
import copy
import types
from common.log import Logger
from common import misc
import numpy
np=numpy
__module_name__=__name__

from matplotlib import pyplot,axes,colors
from matplotlib import pyplot as plt

#---- Colormaps stored in files
cmap_data= {}
    
cdict = {'red':  ((0.0, 0.0, 0.0),
                   (0.35,0.0, 0.0),
                   (0.5, 1, 1),
                   (0.65, .9, .9),
                   (0.95, 0.5,  .5),
                   (1.0,   .4,  .4)),

         'green': ((0.0, 0.0, 0.0),
                   (0.35,0.0, 0.0),
                   (0.5, 1, 1),
                   (0.65,0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0,    .4,     .4),
                   (0.05, 0.5, 0.5),
                   (0.35, 0.9, 0.9),
                   (0.5, 1, 1),
                   (0.65,0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
name='BWR'
cmap_data[name]=cdict

cdict = {'red':  ((0.0, 0.0, 0.0),
                   (0.2,0.0, 0.0),
                   (0.5, 1, 1),
                   (0.8, .9, .9),
                   (0.95, 0.5,  .5),
                   (1.0,   .4,  .4)),

         'green': ((0.0, 0.0, 0.0),
                   (0.2,0.0, 0.0),
                   (0.5, 1, 1),
                   (0.8,0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0,    .4,     .4),
                   (0.05, 0.5, 0.5),
                   (0.2, 0.9, 0.9),
                   (0.5, 1, 1),
                   (0.8,0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
name='BWR2'
cmap_data[name]=cdict

cdict = {'blue':  ((0.0, 0.0, 0.0),
                   (0.2,0.0, 0.0),
                   (0.5, 1, 1),
                   (0.8, .9, .9),
                   (0.95, 0.5,  .5),
                   (1.0,   .4,  .4)),

         'green': ((0.0, 0.0, 0.0),
                   (0.2,0.0, 0.0),
                   (0.5, 1, 1),
                   (0.8,0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'red':  ((0,    .4,     .4),
                   (0.05, 0.5, 0.5),
                   (0.2, 0.9, 0.9),
                   (0.5, 1, 1),
                   (0.8,0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
name='BWR2_r'
cmap_data[name]=cdict

for name in cmap_data:
    cmap=colors.LinearSegmentedColormap(name,cmap_data[name])
    plt.register_cmap(name=name,cmap=cmap)

##Load all colormaps found in `common/colormaps` directory##
# The format of these files should be 4 columns:  x, r, g, b
# All columns should range from 0 to 1.
cmap_dir=os.path.join(os.path.dirname(__file__),'colormaps')
for file in os.listdir(cmap_dir):
    if file.endswith('.csv'):
        cmap_name=re.sub('\.csv$','',file)
        try:
            cmap_mat=misc.extract_array(open(os.path.join(cmap_dir,file)))
            x=cmap_mat[:,0]; r=cmap_mat[:,1]; g=cmap_mat[:,2]; b=cmap_mat[:,3]

            rtuples=numpy.vstack((x,r,r)).transpose().tolist()
            gtuples=numpy.vstack((x,g,g)).transpose().tolist()
            btuples=numpy.vstack((x,b,b)).transpose().tolist()
            cdict={'red':rtuples,'green':gtuples,'blue':btuples}
            cmap=colors.LinearSegmentedColormap(cmap_name,cdict)
            pyplot.register_cmap(name=cmap_name,cmap=cmap)

            r=r[::-1]; g=g[::-1]; b=b[::-1]
            rtuples_r=numpy.vstack((x,r,r)).transpose().tolist()
            gtuples_r=numpy.vstack((x,g,g)).transpose().tolist()
            btuples_r=numpy.vstack((x,b,b)).transpose().tolist()
            cdit_r={'red':rtuples_r,'green':gtuples_r,'blue':btuples_r}
            cmap_name_r=cmap_name+'_r'
            cmap=colors.LinearSegmentedColormap(cmap_name_r,cdict)
            pyplot.register_cmap(name=cmap_name_r,cmap=cmap)

            Logger.write('Registered colormaps "%s" and "%s"...'%((cmap_name,cmap_name_r)))

        except: Logger.warning('Could not register cmap "%s"!'%cmap_name)

# ----- Plotting functions

_color_index_=0
all_colors=['b','g','r','c','m','y','k','teal','gray','navy']

def next_color():
    
    global _color_index_
    color=all_colors[_color_index_%len(all_colors)]
    _color_index_+=1

    return color

def bluered_colors(N):

    return zip( np.linspace(0,1,N),\
                [0]*N,\
                np.linspace(1,0,N) )
    
###################################
#ASM# 2. function figure_list #ASM#
###################################
def figure_list():
    """
    This function uses some internal wizardry to return a list of the current figure objects.
    """
    
    import matplotlib._pylab_helpers
    
    lst = []
    for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers():
        lst.append(manager.canvas.figure)

    return lst

def get_properties(obj,verbose='yes'):
    """
    This function returns a dictionary of artist object properties and their corresponding values.
        *obj: artist object
        *verbose: set to 'yes' to spout error messages for properties whose values could not be obtained.
                  DEFAULT: 'no'
    """
    
    props_to_get=[]
    for attrib in dir(obj):
        if attrib[0:4]=='get_': props_to_get.append(attrib.replace('get_',''))
    values=[]
    props_used=[]
    for prop in props_to_get:
        ##Getp sometimes fails requiring two arguments, but these properties are not important##
        try: 
            values.append(getp(obj,prop))
            props_used.append(prop)
        except TypeError: 
            if 'y' in verbose: print('ALERT: Couldn\'t retrieve property '+prop+'.')
        
    return dict([(props_used[i],values[i]) for i in range(len(values))])
   
def set_properties(obj,prop_dict,verbose='yes'):
    """
    This function takes an object and and sets its properties according to the property dictionary input.
    If, for any entry in the dictionary, the property or method to set it does not exist, it will be skipped
    over.
        *obj: artist object
        *prop_dict: a property dictionary of the sort returned by get_properties()
        *verbose: set to 'yes' to spout error messages for properties which could not be set
                  DEFAULT: 'no'
    """
    
    misc.check_vars(prop_dict,dict)
    for key in list(prop_dict.keys()):
        try: pyplot.setp(obj,key,prop_dict[key])
        except AttributeError:
            if 'y' in verbose: Logger.warning('Property "%s" could not be set.'%key)
    return obj

def minor_ticks(nx=5,ny=5,x=True,y=True):
    """
    Sets *n* minor tick marks per major tick for the x and y axes of the current figure.
       *nx: integer number of minor ticks for x, DEFAULT: 5
       *ny: integer number of minor ticks for y, DEFAULT: 5
       *x: True/False, DEFAULT: True
       *y: True/False, DEFAULT: True
    """
    
    ax = pyplot.gca()
    
    if x:
        ax.xaxis.set_major_locator(pyplot.AutoLocator())
        x_major = ax.xaxis.get_majorticklocs()
        dx_minor =  (x_major[-1]-x_major[0])/(len(x_major)-1)/nx
        ax.xaxis.set_minor_locator(pyplot.MultipleLocator(dx_minor))
    if y:
        ax.yaxis.set_major_locator(pyplot.AutoLocator())
        y_major = ax.yaxis.get_majorticklocs()
        dy_minor =  (y_major[-1]-y_major[0])/(len(y_major)-1)/ny
        ax.yaxis.set_minor_locator(pyplot.MultipleLocator(dy_minor))
        
    pyplot.plot()
    return

def axes_limits(xlims=None,ylims=None,auto=False):
    """
    Sets limits for the x and y axes.
        *xlims: tuple of (xmin,xmax)
        *ylims: tuple of (ymin,ymax)
        *auto: set to True to turn on autoscaling for both axes
    """
    
    ax=pyplot.gca()
    
    ax.set_autoscale_on(auto)
    if xlims!=None: ax.set_xlim(xlims[0],xlims[1])
    if ylims!=None: ax.set_ylim(ylims[0],ylims[1])
    pyplot.draw();return

#totally fucked- axes heights are crazy
def grid_axes(nplots, xstart=0.15, xstop=.85, spacing=0.02,
                bottom=0.1, top = 0.85, widths=None, **kwargs):
    """
    Generates a series of plots neighboring each other horizontally and with common
    y offset and height values.

    nplots - the number of plots to create
    xstart - the left margin of the first plot
             DEFAULT: .05 <- permits visibility of axis label
    xstop - the right margin of the last plot
            DEFAULT: 1 <- entire figure
    spacing - the amount of space between plots
              DEFAULT: .075
    bottom - the bottom margin of the row of plots
    top - the top margin of the row of plots
    widths - specify the width of each plot. By default plots are evenly spaced, but
            if a list of factors is supplied the plots will be adjusted in width. Note
            that if the total adds up to more than the allotted area, RuntimeError is
            raised.
    kwargs - passed to figure.add_axes method
    """
    
    ###Check types###
    input_list=(nplots,xstart,xstop,spacing,bottom,top,widths)
    type_list=[(int,list)] #for nplots
    type_list.extend([(int,float,list)]*5) #for xstart, xstop, spacing, bottom, top
    type_list.append((list,type(None))) #for widths
    type_list=tuple(type_list)
    misc.check_vars(input_list,type_list,protect=[list])
     
    ###Grid bottom and top arguments equally for each row if necessary###
    if type(nplots)==list: #if we want more than one row
        nrows=len(nplots)
        vsize=((top-bottom+nrows*spacing))/float(nrows)
        the_bottom=bottom
        if type(bottom)!=list: #If user hasn't set widths
            bottom=[the_bottom+(vsize)*i+spacing*bool(i) for i in range(nrows)]
            bottom.reverse() #top to bottom
        if type(top)!=list:
            top=[the_bottom+vsize*(i+1) for i in range(nrows)]
            top.reverse() #top to bottom
    
    ###Make sure widths is properly formatted###
    if widths!=None:
        for i in range(len(widths)):
            if type(widths[i])==list: #specific widths for plots in row
                widths[i]=tuple(widths[i]) #turn to tuple to prevent iteration into
    
    ###Define what to do for each row###
    fig=pyplot.gcf()
    def row_of_axes(nplots_row,\
                    xstart,xstop,spacing,\
                    bottom,top,widths,kwargs,index):

        ##Check that we haven't iterated too deep##
        Logger.raiseException('Provide input values for rows and columns only (e.g. lists of depth 2).',\
                         unless=(len(index)<2),\
                         exception=IndexError)

        ##Check format of widths##
        if widths==None: widths=[1]*nplots_row
        elif hasattr(widths,'__len__'): #expect a tuple
            print(len(widths),nplots_row)
            Logger.raiseException('When providing *widths* keyword, provide a plot width for each intended sub-plot in each row.',\
                             unless=(len(widths)==nplots_row),\
                             exception=IndexError)
        else: widths=tuple(widths)*nplots_row
            
        ###Axes values###
        avg_width=(xstop-xstart-spacing*(nplots_row-1))/float(nplots_row)
        height=top-bottom
        xpos=xstart
        
        ###Weighted widths###
        weighted_widths=[]
        for j in range(nplots_row):
            weighted_width=avg_width*widths[j]
            weighted_widths.append(weighted_width)
        true_widths=[width/float(sum(weighted_widths))*(nplots_row*avg_width) \
                     for width in weighted_widths]
        
        ###Make new axes in row###
        row_axes=[]
        for j in range(nplots_row):
            width=true_widths[j]
            rect=[xpos, bottom, width, height]
            new_axis=fig.add_axes(rect, **kwargs)
            xpos+=width+spacing
            row_axes.append(new_axis)

        return row_axes
    
    ###Apply to all rows###
    new_axes=misc.apply_to_array(row_of_axes,nplots,xstart,xstop,spacing,bottom,top,widths,kwargs,\
                                 protect=[tuple,dict])
    pyplot.plot()
    
    return new_axes

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), \
              norm=plt.Normalize(0.0, 1.0),
              linewidth=3, alpha=1.0, **kwargs):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    
    import matplotlib.collections as mcoll
    
    def make_segments(x, y):
        """
        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """
    
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha, **kwargs)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def nice_colorbar(mappable,where='right',size='5%',pad=.05):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(where, size=size, pad=pad)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

#This was shamelessly harvested from:
#http://adversus.110mb.com/?cat=8
#Usage:
###########################################################################################
#Using this class is straightforward. In my case I wanted minor ticks shown on 0.5, 2 and 5:
#
#    axes.yaxis.set_minor_formatter(SelectTickDecorator(ScalarFormatter(), ticklocs=[0.5, 2, 5]))
#
#You could also modify (actually: wrap) the formatter that is already present to select some ticks:
#
#    axes.yaxis.set_major_formatter(SelectTickDecorator(axes.yaxis.get_major_formatter(),\
#                                                       ticklocs=[0.5, 2, 5]))
#
#Note that only 3 methods are passed through to the internal formatter:
#set_axis, set_locs and get_offset. Furthermore SelectTickDecorator does not 
#derive from Formatter. I don't do the latter because I want the application 
#to crash the moment matplotlib requests a method of the formatter that is 
#not handled by SelectTickDecorator. An extra method should then be written 
#passing the request on to the internal formatter.
#
#Note that using real python decorators would be much better for this problem, 
#but that would mean modifying the matplotlib source code so that is not really 
#an option. There might be other python possibilities but my python knowledge is 
#limited to doing it the way shown higher. It works anyhow.
###########################################################################################

class SelectTickDecorator():
    """
    A decorator for matplotlib tick formatters (see
    matplotlib.ticker): lets you select which ticks to plot.
 
    Use set_ticklabel_locs to set a list of tick values you want to show.
 
    Use set_ticklabel_indices to set a list of tick numbers you want
    to plot (bottommost tick: tick 0).
    """
    ## floating point comparison precision is the axis range
    ## multiplied by fractionalPrecision.
    fractionalPrecision = 1.e-8
 
    def __init__(self, formatter, **kwargs):
        """
        Constructor. Takes a Formatter as an argument.

        kwargs can contain:

        - ticklocs    : see set_ticklabel_locs
        - tickindices : see set_ticklabel_indices
        """
        self.formatter = formatter
        self.set_ticklabel_locs(kwargs.get('ticklocs',[]) )
        self.set_ticklabel_indices( kwargs.get('tickindices', []) )
 
        ## default value, is overridden in set_axis
        self.precision = SelectTickDecorator.fractionalPrecision
 
    def set_ticklabel_locs(self, tickLocs):
        """
        tickLocs should be a -possibly empty- iterable containing the
        values of the tics you actually want to show the label of.
        """
        self.tickLocs = tickLocs
 
    def set_ticklabel_indices(self, tickIndices):
        """
        Set a list if tick numbers you want to show (e.g. the 0th, the
        4th and the 5th).
        """
        self.tickIndices = tickIndices
 
    def __call__(self, x, pos=None):
        """
        The decorator intercepts these Formatter calls. They are only
        passed on to the interior Formatter class if the tick is
        requested by the user.
        """
        showtick = False
        for loc in self.tickLocs:
            if abs(x-loc) < self.precision:
                showtick = True
 
        if not showtick and pos in self.tickIndices:
            showtick = True
 
        if showtick:
            return self.formatter.__call__(x, pos)
        else:
            return ''
 
    def set_axis(self, axis):
        """
        Forward the set_axis call to the formatter. Also set the
        floating point comparison precision
        """
        self.formatter.set_axis(axis)
 
        vmin, vmax = self.formatter.axis.get_data_interval()
        self.precision = (vmax - vmin) * SelectTickDecorator.fractionalPrecision
 
    def set_locs(self, locs):
        self.formatter.set_locs(locs)
        
    def get_offset(self):
        return self.formatter.get_offset()
    
# Define the point-picking interface #
class PointPicker(object):

    def __init__(self,mousebutton=1,max_pts=1,ax=None,cbar=None,message=None,verbose=False,**kwargs):
        
        import time
        
        self.pts = []
        self.verbose=verbose
        self.kwargs=kwargs
        
        self.mousebutton=mousebutton
        self.max_pts=max_pts
        self.colors=['b','r','g','c', 'm', 'y']
        
        self.cid1=plt.gcf().canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2=plt.gcf().canvas.mpl_connect('key_press_event', self.ondelete)
        self.cid3=plt.gcf().canvas.mpl_connect('key_press_event', self.onquit)
        self.cid4=plt.gcf().canvas.mpl_connect('key_press_event', self.colorlimits)
        
        if not ax: ax=plt.gca()
        self.ax=ax
        self.cbar=cbar
        
        if verbose:
            if not message:
                message='Please click %s points on the current figure using \
                         mouse button %i.  You may at any time strike the "c" \
                         key to change color limits on an image, or the "delete" \
                         key to remove the last selected point.'%(max_pts,mousebutton)
            Logger.write(message)
            Logger.write('\tYou are now on point #1.')
        
        plt.gcf().canvas.start_event_loop(timeout=0)

    def onclick(self, event):
        
        if event.button is not self.mousebutton: return
        
        if event.xdata==None:
            event.xdata=0; event.ydata=0
        self.pts.append((event.xdata,event.ydata))

        try: color=self.kwargs['color']
        except KeyError: color=self.colors[(len(self.pts)-1)\
                                           %len(self.colors)] #Colors will cycle
        try: fontsize=self.kwargs['fontsize']
        except KeyError: fontsize=14
        
        l=self.ax.plot([event.xdata],[event.ydata],'o',color=color)[0]
        self.ax.text(event.xdata,event.ydata,len(self.pts),fontsize=fontsize,\
                     bbox=dict(facecolor='white', alpha=0.25),\
                     horizontalalignment='center',\
                     verticalalignment='bottom')
        plt.draw()
        
        # Quit the loop if we achieve the desired number of points #
        if self.max_pts and len(self.pts)==self.max_pts:
            event.key='quit_for_sure'; self.onquit(event)
            
        if self.verbose: Logger.write('You are now on point #%i'%(len(self.pts)+1))
            
    def ondelete(self,event):
        
        if event.key in ('delete','backspace'):
            if len(self.pts):
                self.pts.pop(-1)
                self.ax.lines.pop(-1)
                self.ax.texts.pop(-1)
                plt.draw()
                
    def colorlimits(self,event):
        
        if event.key=='c':
            
            clims_ok=False
            while not clims_ok:
                clims_txt=input('Type two values separated by a comma or whitespace to use '+\
                                    'for new color limits.  Or press [enter] to skip. \n'+\
                                   'New color limits:  ')
                if clims_txt:
                    try: 
                        clims=clims_txt.split()
                        clims=[clim.strip(',') for clim in clims]
                        if len(clims)==1:
                            clims=clims[0].split(',')
                        clim_min,clim_max=sorted([numpy.float(clim) for clim in clims])
                        clims_ok=True
                        break
                    except:
                        print('Colorbar limits were not formatted correctly!  Try again.')
                    
            self.ax.images[0].set_clim(clim_min,clim_max)
            plt.draw()
                
    def onquit(self,event):
        
        # Quit on "enter" only if `self.max_pts` was not defined, #
        # otherwise leave the quitting to `self.onclick`.         #
        if (event.key in ('return','enter') and not self.max_pts) \
            or event.key=='quit_for_sure':
            plt.gcf().canvas.stop_event_loop()
            plt.gcf().canvas.mpl_disconnect(self.cid1)
            plt.gcf().canvas.mpl_disconnect(self.cid2)
            plt.gcf().canvas.mpl_disconnect(self.cid3)
            plt.gcf().canvas.mpl_disconnect(self.cid4)
            
    def get_points(self): return copy.copy(self.pts)
    
class AxesDefaults(object):
    
    def __init__(self,label_fontsize=24,legend_fontsize=22,title_fontsize=26,math_fontsize_boost=1.25,\
                 tick_fontsize=20,minor_tick_fontsize=None,\
                 fontname=None,fontweight='ultralight',scale_factor=1,\
                 labelpad_x=10,labelpad_y=15,tick_pad=5,\
                 tick_width=None,tick_length=None,minor_ticks_on=True,minor_tick_width=None,minor_tick_length=None,\
                 linewidth=None,line_alpha=None,markersize=None,marker_alpha=None,\
                 subplot_params=None,figure_size=None):
    
        self.label_fontsize=label_fontsize
        self.legend_fontsize=legend_fontsize
        self.title_fontsize=title_fontsize
        self.math_fontsize_boost=math_fontsize_boost
        
        self.fontname=fontname
        self.tick_fontsize=tick_fontsize
        self.minor_tick_fontsize=minor_tick_fontsize
        
        self.fontweight=fontweight
        self.scale_factor=scale_factor
        
        self.labelpad_x=labelpad_x
        self.labelpad_y=labelpad_y
        self.tick_pad=tick_pad
        
        self.tick_width=tick_width
        self.tick_length=tick_length
        self.minor_ticks_on=minor_ticks_on
        self.minor_tick_width=minor_tick_width
        self.minor_tick_length=minor_tick_length
        
        self.linewidth=linewidth
        self.line_alpha=line_alpha
        self.markersize=markersize
        self.marker_alpha=marker_alpha
        
        self.subplot_params=subplot_params
        self.figure_size=figure_size
        
    def set_label_fontsize(self,label,fs):
        
        math_boost=self.math_fontsize_boost
        
        text=label.get_text()
        if text.startswith('$') and text.endswith('$') and math_boost: fs=fs*math_boost
        label.set_fontsize(fs*self.scale_factor)
        if self.fontname:  label.set_fontname(self.fontname)
        
    def set_fontweights(self,ax,fw):
        
        ts=[ax.title,\
            ax.xaxis.get_label(),\
            ax.xaxis.get_label()]+\
            ax.xaxis.get_majorticklabels()+\
            ax.xaxis.get_minorticklabels()+\
            [ax.yaxis.get_label(),\
            ax.yaxis.get_label()]+\
            ax.yaxis.get_majorticklabels()+\
            ax.yaxis.get_minorticklabels()+\
            ax.texts
        leg=ax.legend_
        if leg is not None: ts+=leg.texts
        
        for t in ts: t.set_fontweight(fw)
        
    def set_labels(self,ax):
                
        fs=self.title_fontsize
        if fs is not None: self.set_label_fontsize(ax.title,fs=fs)
        
        pad=self.labelpad_x
        if pad is not None: ax.xaxis.labelpad=pad*self.scale_factor
        pad=self.labelpad_y
        if pad is not None: ax.yaxis.labelpad=pad*self.scale_factor
        
        fs=self.label_fontsize
        if fs is not None:
            self.set_label_fontsize(ax.xaxis.get_label(),fs=fs)
            self.set_label_fontsize(ax.yaxis.get_label(),fs=fs)
            
        fs=self.legend_fontsize; leg=ax.legend_
        if fs is not None and leg is not None:
            for t in leg.texts:
                self.set_label_fontsize(t,fs=fs)
                
        tp=self.tick_pad
        if tp is not None:
            for t in ax.xaxis.get_major_ticks()+\
                     ax.yaxis.get_major_ticks()+\
                     ax.xaxis.get_minor_ticks()+\
                     ax.yaxis.get_minor_ticks():
                t.set_pad(tp**self.scale_factor); # t.label1 = t._get_text1()
                
        fs=self.tick_fontsize
        if fs is not None:
            for t in ax.xaxis.get_majorticklabels():
                self.set_label_fontsize(t,fs=fs)
            for t in ax.yaxis.get_majorticklabels():
                self.set_label_fontsize(t,fs=fs)
                
        fs=self.minor_tick_fontsize
        if fs is not None:
            for t in ax.xaxis.get_minorticklabels():
                self.set_label_fontsize(t,fs=fs)
            for t in ax.yaxis.get_minorticklabels():
                self.set_label_fontsize(t,fs=fs)
        
        fw=self.fontweight
        if fw is not None: self.set_fontweights(ax,fw=fw)
        
                
    def set_ticksizes(self,ax):
        
        tw=self.tick_width
        if tw is not None: plt.tick_params(which='major',width=tw*self.scale_factor)
        tl=self.tick_length
        if tl is not None: plt.tick_params(which='major',length=tl*self.scale_factor)
        
        if not self.minor_ticks_on:
            from matplotlib.ticker import NullLocator
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
        
        tw=self.minor_tick_width
        if tw is not None: plt.tick_params(which='minor',width=tw*self.scale_factor)
        tl=self.minor_tick_length
        if tl is not None: plt.tick_params(which='minor',length=tl*self.scale_factor)
        
    def set_lines_and_markers(self,ax):
        
        lw=self.linewidth
        if lw is not None:
            for l in ax.lines: l.set_linewidth(lw*self.scale_factor)
        la=self.line_alpha
        if la is not None:
            for l in ax.lines: l.set_alpha(la)
        ms=self.markersize
        if ms is not None:
            for l in ax.lines:
                l.set_markersize(ms*self.scale_factor)
        ma=self.marker_alpha
        if ma is not None:
            for l in ax.lines:
                marker=l.get_marker()
                if marker and marker!='None':
                    l.set_alpha(ma)
                
    def set_subplot_params(self,ax):
        
        subplot_params=self.subplot_params
        if subplot_params is not None:
            if isinstance(subplot_params,dict): plt.subplots_adjust(**subplot_params)
            elif hasattr(subplot_params,'__len__'): plt.subplots_adjust(*subplot_params)
                
    def set_figure_size(self,ax):
        
        figsize=self.figure_size
        if figsize is not None:
            if len(ax.figure.axes)==1:
                size=[s*self.scale_factor for s in figsize]; size[-1]+=.5
                ax.figure.set_size_inches(size,forward=True)
        
    def __call__(self,ax=None,**temp_settings):
        
        if ax is None: ax=plt.gca()
        
        ## Make some temporary settings, if instructed ##
        original_settings={}
        for key in temp_settings:
            assert hasattr(self,key),'"%s" is not a valid setting covered by this axis formatter!'
            original_settings[key]=getattr(self,key)
            setattr(self,key,temp_settings[key])
        
        ## Fontsizes ##
        self.set_labels(ax)
                
        ## Tick sizes ##
        self.set_ticksizes(ax)
        
        ## Lines and markers ##
        self.set_lines_and_markers(ax)
                
        ## Subplot parameters ##
        self.set_subplot_params(ax)
                
        ## Figure size ##
        self.set_figure_size(ax)
        
        ##  Restore original settings ##
        for key in original_settings: setattr(self,key,original_settings[key])
        

def linecut(width=1,plot=True,pts=None,data=None,\
            avg_profiles=True,mode='nearest',**kwargs):
    
    import numpy as np
    from scipy import ndimage
    
    from_image=False
    if hasattr(data,'get_array'):
        im=data
        from_image=True
    elif data is None:
        im=plt.gci()
        assert im is not None,'No current image, so provide explicit `data`!'
        from_image=True
    else:
        data=np.asarray(data)
        from_image=False
        
    if from_image: data=im.get_array().T #Transpose because axes are swtiched in image
    
    if pts is not None: pt1,pt2=pts
    else:
        PP=PointPicker(max_pts=2,verbose=True,mousebutton=3)
        pt1,pt2=PP.get_points()
    (x1,y1),(x2,y2)=pt1,pt2
    N=int(numpy.sqrt((x2-x1)**2+(y2-y1)**2))
    
    angle=numpy.arctan2(y2-y1,x2-x1)
    dx,dy=numpy.sin(angle),-numpy.cos(angle)
    
    profiles=[]
    for lineno in range(int(width)):
        xoffset=(width/2-.5-lineno)*dx
        yoffset=(width/2-.5-lineno)*dy
        xi,yi=x1+xoffset,y1+yoffset
        xf,yf=x2+xoffset,y2+yoffset
        
        xs=numpy.linspace(xi,xf,N)
        ys=numpy.linspace(yi,yf,N)
        if from_image and plot:
            X1,X2,Y1,Y2=im.get_extent()
            dX=(X2-X1)/data.shape[0]
            Xs=X1+xs*dX
            dY=(Y2-Y1)/data.shape[1]
            Ys=Y1+ys*dY
            im.axes.plot(Xs,Ys,color='k',alpha=.5)
        
        profile = ndimage.map_coordinates(data, np.vstack((xs,ys)),\
                                          mode=mode,**kwargs)
        profiles.append(profile)
    
    if avg_profiles: return numpy.mean(profiles,axis=0)
    else: return numpy.array(profiles)
    
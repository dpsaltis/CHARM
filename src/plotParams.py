## @package plotParams
#
# defines matplotlib parameters for the plots
#
#    \author UofA group
#  
#    \version 2.0
#  
#    \date October 11, 2019
#
#    \bug No known bugs
#    
#    \warning No known warnings
#    
#    \todo Nothing left

# necessary libraries
import matplotlib.pyplot as plt       # imports library for plots
from matplotlib import rcParams       # import to change plot parameters

def setPlotParams():
    """!@brief Sets matplotlib parameters

    Sets a number of matplotlib parameters (using rcParams) to
    create plots. 

    It requires latex fonts on python.
    
    @returns nothing

    \author UofA group
  
    \version 2.0
  
    \date October 11, 2019

    \bug No known bugs
    
    \warning No known warnings
    
    \todo Nothing left
    """
    
    rcParams['text.usetex']=True
    rcParams['font.family']='serif'
    #rcParams['font.sans-serif']='Latin Modern Roman'
    
    # axes and tickmarks
    rcParams['axes.labelsize']=18
    #rcParams['axes.labelweight']=600
    rcParams['axes.linewidth']=1.5
    
    rcParams['xtick.labelsize']=18
    rcParams['xtick.top']=True
    rcParams['xtick.direction']='in'
    rcParams['xtick.major.size']=6
    rcParams['xtick.minor.size']=3
    rcParams['xtick.major.width']=1.2
    rcParams['xtick.minor.width']=1.2
    rcParams['xtick.minor.visible']=True
    
    rcParams['ytick.labelsize']=18
    rcParams['ytick.right']=True
    rcParams['ytick.direction']='in'
    rcParams['ytick.major.size']=6
    rcParams['ytick.minor.size']=3
    rcParams['ytick.major.width']=1.2
    rcParams['ytick.minor.width']=1.2
    rcParams['ytick.minor.visible']=True
    
    # points and lines
    rcParams['lines.linewidth']=2.0
    rcParams['lines.markeredgewidth']=0.5
    rcParams['lines.markersize']=6


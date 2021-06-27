import numpy as np
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    

def lrtzX(f, A, f0, df, ph, bg_0, bg_1):
    phr = ph * np.pi / 180.
    bg = bg_1 * f + bg_0
    denom = (f0**2 - f**2)**2 +(f * df)**2
    ynum = (A * f0 *df) * f * df 
    xnum = (A * f0 * df) * (f0**2 - f**2)
       
    x = xnum / denom
    y = ynum / denom
        
    xCos = x * np.cos(phr)
    ySin = y * np.sin(phr)
        
    return (xCos - ySin) + bg 

#----------------------------------------------------------------------------------------

def lrtzY(f, A, f0, df, ph, bg_0, bg_1):
    phr = ph * np.pi / 180.
    bg = bg_0 + bg_1 * f
    denom = (f0**2 - f**2)**2 + (f * df)**2 
    ynum = (A * f0 * df) * f * df
    xnum = (A * f0 * df) * (f0** 2 - f** 2)
        
    x = xnum / denom
    y = ynum / denom
        
    xSin = x * np.sin(phr)
    yCos = y * np.cos(phr)
        
    return (yCos + xSin) + bg

#---------------------------------------------------------------------------------------

def lrtzRR(f, A, f0, df, bg_0 , bg_1):
    bg = bg_1 * f + bg_0
    denom=(f0**2 - f**2)**2 + (f * df)**2
    rrnum=(A * f0 * df)**2 
        
    return (rrnum / denom) + bg

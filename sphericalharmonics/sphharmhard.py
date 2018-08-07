#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:16:55 2018
 
@author: dietz
"""
 
import numba
from cmath import exp
from math import sin,cos 
 
@numba.njit(numba.complex128(numba.int64,numba.int64,numba.float64,numba.float64),nogil=True)
def sph_harm_hard(l,m,theta,phi):
    "hard coded spherical harmonics extracted from sympy"
    if l==0 and m==0: 
        return 0.282094791773878
    elif l==1 and m==-1: 
        return 0.345494149471335*exp(-1j*phi)*sin(theta)
    elif l==1 and m==0: 
        return 0.48860251190292*cos(theta)
    elif l==1 and m==1: 
        return -0.345494149471335*exp(1j*phi)*sin(theta)
    elif l==2 and m==-2: 
        return -0.38627420202319*exp(-2*1j*phi)*cos(theta)**2 + 0.38627420202319*exp(-2*1j*phi)
    elif l==2 and m==-1: 
        return 0.772548404046379*exp(-1j*phi)*sin(theta)*cos(theta)
    elif l==2 and m==0: 
        return 0.94617469575756*cos(theta)**2 - 0.31539156525252
    elif l==2 and m==1: 
        return -0.772548404046379*exp(1j*phi)*sin(theta)*cos(theta)
    elif l==2 and m==2: 
        return -0.38627420202319*exp(2*1j*phi)*cos(theta)**2 + 0.38627420202319*exp(2*1j*phi)
    elif l==3 and m==-3: 
        return 0.417223823632784*exp(-3*1j*phi)*sin(theta)**3
    elif l==3 and m==-2: 
        return -1.02198547643328*exp(-2*1j*phi)*cos(theta)**3 + 1.02198547643328*exp(-2*1j*phi)*cos(theta)
    elif l==3 and m==-1: 
        return 1.61590092057075*exp(-1j*phi)*sin(theta)*cos(theta)**2 - 0.323180184114151*exp(-1j*phi)*sin(theta)
    elif l==3 and m==0: 
        return 1.86588166295058*cos(theta)**3 - 1.11952899777035*cos(theta)
    elif l==3 and m==1: 
        return -1.61590092057075*exp(1j*phi)*sin(theta)*cos(theta)**2 + 0.323180184114151*exp(1j*phi)*sin(theta)
    elif l==3 and m==2: 
        return -1.02198547643328*exp(2*1j*phi)*cos(theta)**3 + 1.02198547643328*exp(2*1j*phi)*cos(theta)
    elif l==3 and m==3: 
        return -0.417223823632784*exp(3*1j*phi)*sin(theta)**3
    elif l==4 and m==-4: 
        return 0.442532692444983*exp(-4*1j*phi)*sin(theta)**4
    elif l==4 and m==-3: 
        return 1.25167147089835*exp(-3*1j*phi)*sin(theta)**3*cos(theta)
    elif l==4 and m==-2: 
        return -2.34166290245051*exp(-2*1j*phi)*cos(theta)**4 + 2.67618617422916*exp(-2*1j*phi)*cos(theta)**2 - 0.334523271778645*exp(-2*1j*phi)
    elif l==4 and m==-1: 
        return 3.31161143515146*exp(-1j*phi)*sin(theta)*cos(theta)**3 - 1.41926204363634*exp(-1j*phi)*sin(theta)*cos(theta)
    elif l==4 and m==0: 
        return 3.70249414203215*cos(theta)**4 - 3.17356640745613*cos(theta)**2 + 0.317356640745613
    elif l==4 and m==1: 
        return -3.31161143515146*exp(1j*phi)*sin(theta)*cos(theta)**3 + 1.41926204363634*exp(1j*phi)*sin(theta)*cos(theta)
    elif l==4 and m==2: 
        return -2.34166290245051*exp(2*1j*phi)*cos(theta)**4 + 2.67618617422916*exp(2*1j*phi)*cos(theta)**2 - 0.334523271778645*exp(2*1j*phi)
    elif l==4 and m==3: 
        return -1.25167147089835*exp(3*1j*phi)*sin(theta)**3*cos(theta)
    elif l==4 and m==4: 
        return 0.442532692444983*exp(4*1j*phi)*sin(theta)**4
    elif l==5 and m==-5: 
        return 0.464132203440858*exp(-5*1j*phi)*sin(theta)**5
    elif l==5 and m==-4: 
        return 1.46771489830575*exp(-4*1j*phi)*sin(theta)**4*cos(theta)
    elif l==5 and m==-3: 
        return 3.11349347232156*exp(-3*1j*phi)*sin(theta)**3*cos(theta)**2 - 0.34594371914684*exp(-3*1j*phi)*sin(theta)**3
    elif l==5 and m==-2: 
        return -5.0843135497827*exp(-2*1j*phi)*cos(theta)**5 + 6.7790847330436*exp(-2*1j*phi)*cos(theta)**3 - 1.6947711832609*exp(-2*1j*phi)*cos(theta)
    elif l==5 and m==-1: 
        return 6.72591462010052*exp(-1j*phi)*sin(theta)*cos(theta)**4 - 4.48394308006701*exp(-1j*phi)*sin(theta)*cos(theta)**2 + 0.320281648576215*exp(-1j*phi)*sin(theta)
    elif l==5 and m==0: 
        return 7.36787031456569*cos(theta)**5 - 8.18652257173965*cos(theta)**3 + 1.75425483680135*cos(theta)
    elif l==5 and m==1: 
        return -6.72591462010052*exp(1j*phi)*sin(theta)*cos(theta)**4 + 4.48394308006701*exp(1j*phi)*sin(theta)*cos(theta)**2 - 0.320281648576215*exp(1j*phi)*sin(theta)
    elif l==5 and m==2: 
        return -5.0843135497827*exp(2*1j*phi)*cos(theta)**5 + 6.7790847330436*exp(2*1j*phi)*cos(theta)**3 - 1.6947711832609*exp(2*1j*phi)*cos(theta)
    elif l==5 and m==3: 
        return -3.11349347232156*exp(3*1j*phi)*sin(theta)**3*cos(theta)**2 + 0.34594371914684*exp(3*1j*phi)*sin(theta)**3
    elif l==5 and m==4: 
        return 1.46771489830575*exp(4*1j*phi)*sin(theta)**4*cos(theta)
    elif l==5 and m==5: 
        return -0.464132203440858*exp(5*1j*phi)*sin(theta)**5
    elif l==6 and m==-6: 
        return 0.483084113580066*exp(-6*1j*phi)*sin(theta)**6
    elif l==6 and m==-5: 
        return 1.6734524581001*exp(-5*1j*phi)*sin(theta)**5*cos(theta)
    elif l==6 and m==-4: 
        return 3.92459389139398*exp(-4*1j*phi)*sin(theta)**4*cos(theta)**2 - 0.356781262853998*exp(-4*1j*phi)*sin(theta)**4
    elif l==6 and m==-3: 
        return 7.16529534454487*exp(-3*1j*phi)*sin(theta)**3*cos(theta)**3 - 1.95417145760315*exp(-3*1j*phi)*sin(theta)**3*cos(theta)
    elif l==6 and m==-2: 
        return -10.7479430168173*exp(-2*1j*phi)*cos(theta)**6 + 16.6104573896268*exp(-2*1j*phi)*cos(theta)**4 - 6.1882096157433*exp(-2*1j*phi)*cos(theta)**2 + 0.325695242933858*exp(-2*1j*phi)
    elif l==6 and m==-1: 
        return 13.5951920379376*exp(-1j*phi)*sin(theta)*cos(theta)**5 - 12.3592654890342*exp(-1j*phi)*sin(theta)*cos(theta)**3 + 2.0598775815057*exp(-1j*phi)*sin(theta)*cos(theta)
    elif l==6 and m==0: 
        return 14.6844857238222*cos(theta)**6 - 20.024298714303*cos(theta)**4 + 6.67476623810098*cos(theta)**2 - 0.317846011338142
    elif l==6 and m==1: 
        return -13.5951920379376*exp(1j*phi)*sin(theta)*cos(theta)**5 + 12.3592654890342*exp(1j*phi)*sin(theta)*cos(theta)**3 - 2.0598775815057*exp(1j*phi)*sin(theta)*cos(theta)
    elif l==6 and m==2: 
        return -10.7479430168173*exp(2*1j*phi)*cos(theta)**6 + 16.6104573896268*exp(2*1j*phi)*cos(theta)**4 - 6.1882096157433*exp(2*1j*phi)*cos(theta)**2 + 0.325695242933858*exp(2*1j*phi)
    elif l==6 and m==3: 
        return -7.16529534454487*exp(3*1j*phi)*sin(theta)**3*cos(theta)**3 + 1.95417145760315*exp(3*1j*phi)*sin(theta)**3*cos(theta)
    elif l==6 and m==4: 
        return 3.92459389139398*exp(4*1j*phi)*sin(theta)**4*cos(theta)**2 - 0.356781262853998*exp(4*1j*phi)*sin(theta)**4
    elif l==6 and m==5: 
        return -1.6734524581001*exp(5*1j*phi)*sin(theta)**5*cos(theta)
    else:
        return 0.483084113580066*exp(6*1j*phi)*sin(theta)**6

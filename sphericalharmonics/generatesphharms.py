#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:39:05 2018
 
@author: dietz
"""
 
from sympy import Ynm, Symbol,N
 
theta=Symbol("theta")
phi=Symbol("phi")
 
generated_code=''
 
first_iter=True
for n in range(0,7):
    for m in range(-n,n+1):
        if n==6 and m==n:
            generated_code += 'else:\n'
        else:
            if first_iter:
                generated_code += 'if '
            else:
                generated_code += 'elif '
            generated_code +='l == {:d} and m == {:d}:\n'.format(n,m)
            
        
        
        sph_harm=N(Ynm(n,m,theta,phi).expand(func=True))
        sph_harm=str(sph_harm).replace('I','1j')
        generated_code+='    return '+sph_harm+'\n'
        first_iter=False
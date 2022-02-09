#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:31:55 2021

@author: kwkwon
"""

'''
    Main reference is "Riemann Solvers and Numerical Methods for Fluid Dynamics" by E. F. Toro.
    The Primitive Formulation of the Euler Equations
    W_t + A(W)W_x = 0
    W=[rho, u, p], A(W) = [[u, rho 0], [0, u , 1/rho], [0, rho * a^2, u]]
    W(x,0) = W_L = [rho_L, u_L, p_L] if x < 0
             W_R = [rho_R, u_R, p_R] if x > 0
    gamma = c_p / c_nu is the ratio of specific heats
    a = sqrt(gamma*p / rho) is the sound of speed
    We restrict our attention to ideal gases obeying the caloric EOS(equation of states)
    e = p / ((gamma-1) * rho)
    Analysis of eigenvalue structure of the equations shows that
    in the star region, u_*, p_* are constant while rho_*L and rho_*R have different values in general.
    We only consider no vacuum case.
'''

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation

def case(test):
    if test == 'test1': # Sod test problem
        DL = 1.0  # Initial density on left state
        UL = 0.0  # Initial velocity on left state
        PL = 1.0  # Initial pressure on left state
        DR = 0.125  # Initial density on right state
        UR = 0.0  # Initial velocity on right state
        PR = 0.1  # Initial pressure on right state
        return DL, UL, PL, DR, UR, PR
    elif test == 'test2': # the 123 problem
        DL = 1.0  
        UL = -2.0  
        PL = 0.4  
        DR = 1.0  
        UR = 2.0  
        PR = 0.4  
        return DL, UL, PL, DR, UR, PR
    elif test == 'test3':   
        DL = 1.0  
        UL = 0.0  
        PL = 1000.0  
        DR = 1.0  
        UR = 0.0  
        PR = 0.01  
        return DL, UL, PL, DR, UR, PR
    elif test == 'test4':
        DL = 1.0  
        UL = 0.0  
        PL = 0.01  
        DR = 1.0  
        UR = 0.0  
        PR = 100.0  
        return DL, UL, PL, DR, UR, PR
    elif test == 'test5': # the right and left shocks emerging from test3 and test4
        DL = 5.99924  
        UL = 19.5975  
        PL = 460.894  
        DR = 5.99242
        UR = -6.19633
        PR = 46.0950
        return DL, UL, PL, DR, UR, PR


# Declaration of Variables
DL, UL, PL, DR, UR, PR = case('test3')
GAMMA = 1.4 # The ratio of specific heats. This value must be more than 1.


# Compute gamma related constants
G1 = (GAMMA - 1.0) / (2.0*GAMMA)
G2 = (GAMMA + 1.0) / (2.0*GAMMA)
G3 = 2.0*GAMMA/(GAMMA - 1.0)
G4 = 2.0/(GAMMA - 1.0)
G5 = 2.0/(GAMMA + 1.0)
G6 = (GAMMA - 1.0)/(GAMMA + 1.0)
G7 = (GAMMA - 1.0)/2.0
G8 = GAMMA - 1.0

# Compute sound speeds
aL = np.sqrt(GAMMA*PL/DL)   
aR = np.sqrt(GAMMA*PR/DR)
UD = UR - UL    # Difference of velocities


# Test for the pressure positivity condition
if 2.0/(GAMMA - 1.0)*(aL + aR) <= UR-UL:
    print("The initial data is such that vacuum is generated.")
    print("Program stopped.")
    exit()



# The pressure function
def pressFuncLeft(p, DL, PL, G1, G2, G3, G4, G5, G6, G7, G8):
    AL = G5/DL
    BL = G6*PL
    aL = np.sqrt(GAMMA*PL/DL)   
    if p > PL:  # Shock wave
        FL = (p - PL)*np.sqrt(AL/(p + BL))
    else:  # Rarefaction wave
        FL = G4*aL*((p/PL)**G1 - 1.0)
    return FL

def pressFuncRight(p, DR, PR, G1, G2, G3, G4, G5, G6, G7, G8):
    AR = G5/DR
    BR = G6*PR
    aR = np.sqrt(GAMMA*PR/DR)
    if p > PR:  # Shock wave
        FR = (p - PR)*np.sqrt(AR/(p + BR))
    else:  # Rarefaction wave
        FR = G4*aR*((p/PR)**G1 - 1.0)
    return FR

def pressFunc(p, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8):
    FL = pressFuncLeft(p, DL, PL, G1, G2, G3, G4, G5, G6, G7, G8)
    FR = pressFuncRight(p, DR, PR, G1, G2, G3, G4, G5, G6, G7, G8)
    UD = UR - UL
    F = FL + FR + UD
    return F

# Find the pressure PS on the star region by solving the pressure equation.
PMax = max(PL, PR)
PMin = min(PL, PR)
FMax = pressFunc(PMax, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8)
FMin = pressFunc(PMin, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8)

def derivPress(p, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8):
    AL = G5/DL
    BL = G6*PL
    AR = G5/DR
    BR = G6*PR
    if p > PL:  # Shock wave
        DFL = np.sqrt(AL/(BL + p))*(1.0 - 1.0/2.0*(p - PL)/(BL + p))
    else:   # Rarefaction wave
        DFL = 1.0/(DL*aL)*(p/PL)**(-G2)
    if p > PR:  # Shock wave
        DFR = np.sqrt(AR/(BR + p))*(1.0 - 1.0/2.0*(p - PR)/(BR + p))
    else:   # Rarefaction wave
        DFR = 1.0/(DR*aR)*(p/PR)**(-G2)
    DF = DFL + DFR
    return DF

def newtonBisect(f, df, a, b, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8, tol=1.0e-9):
    from numpy import sign

    fa = f(a, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8)
    if fa == 0.0: 
        return a
    fb = f(b, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8)
    if fb == 0.0: 
        return b
    if sign(fa) == sign(fb):
        print('Root is not bracketed')
    x = 0.5 * (a + b)
    for i in range(30):
        fx = f(x, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8)
        if fx == 0.0:
            return x
        # Tighten the brackets on the root
        if sign(fa) != sign(fx):
            b = x
        else:
            a = x
        # Try a Newton-Raphson step
        dfx = df(x, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8)
        # If division by zero, push x out of bounds
        try:
            dx = -fx / dfx
        except ZeroDivisionError:
            dx = b - a
        x = x + dx
        # If the result is outside the brackets, use bisection
        if (b - x) * (x - a) < 0.0:
            dx = 0.5 * (b - a)
            x = a + dx
        # Check for convergence
        if abs(dx) < tol * max(abs(b), 1.0): return x
    print('Too many iterations in Newton-Raphson')

def newton(f, df, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8, tol=1.0e-9):
    P1 = 1/2*(PL + PR)
    for i in range(30):
        Pn = P1 - f(P1, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8)/df(P1, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8)
        if abs(2*(Pn - P1)/(Pn + P1)) < tol:
                return Pn
        P1 = Pn
        
if FMax > 0.0 and FMin > 0.0:
    a = 0.0
    b = PMin
    PS = newtonBisect(pressFunc, derivPress, a, b, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8, tol=1.0e-9)
elif FMax >= 0.0 and FMin <= 0.0:
    a = PMin
    b = PMax
    PS = newtonBisect(pressFunc, derivPress, a, b, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8, tol=1.0e-9)
elif FMax < 0.0 and FMin < 0.0:
    PS = newton(pressFunc, derivPress, DL, UL, PL, DR, UR, PR, G1, G2, G3, G4, G5, G6, G7, G8, tol=1.0e-9)
print(PS)

# Find the velocity US on the star region.
FL = pressFuncLeft(PS, DL, PL, G1, G2, G3, G4, G5, G6, G7, G8)
FR = pressFuncRight(PS, DR, PR, G1, G2, G3, G4, G5, G6, G7, G8)
US = 1.0/2.0*(UL + UR) + 1.0/2.0*(FR - FL)
print(US)

# Construct Mesh and time interval
Nx = 500
x = np.linspace(-2, 2, Nx)
Nt = 200
t = np.linspace(0, 1, Nt)

# Exact solution of the Euler equations.
D = np.zeros([Nt, Nx])  # The density
U = np.zeros([Nt, Nx])  # The velocity
P = np.zeros([Nt, Nx])  # The pressure

# Compute the density DSL and DSR on the star region.
AL = G5/DL
BL = G6*PL
if PS > PL: # The left shock wave
    DSL = DL*(PS/PL + G6)/(G6*PS/PL + 1.0)
    SL = UL - aL*np.sqrt(G2*PS/PL + G1)     # The left shock speed
    for i in range(1, Nt):
        for j in range(0, Nx):
            if SL <= x[j]/t[i] and x[j]/t[i] <= US:
                D[i][j] = DSL
                U[i][j] = US
                P[i][j] = PS
            elif x[j]/t[i] <= SL:
                D[i][j] = DL
                U[i][j] = UL
                P[i][j] = PL

else:   # The left rarefaction wave
    DSL = DL*(PS/PL)**(1.0/GAMMA)
    aSL = aL*(PS/PL)**G1   # The sound speed behind rarefaction wave
    SHL = UL - aL   # The speed of the head of the wave
    STL = US - aSL  # The speed of the tail of the wave
    # Density, Velocity, Pressure inside the fan wave
    DLF = np.zeros([Nt, Nx])
    ULF = np.zeros([Nt, Nx])
    PLF = np.zeros([Nt, Nx])
    for i in range(1, Nt):
        for j in range(0, Nx):
            if x[j]/t[i] <= SHL:
                D[i][j] = DL
                U[i][j] = UL
                P[i][j] = PL
            elif SHL <= x[j]/t[i] and x[j]/t[i] <= STL:
                D[i][j] = DL*(G5 + G6/aL*(UL - x[j]/t[i]))**G4
                U[i][j] = G5*(aL + G7*UL + x[j]/t[i])
                P[i][j] = PL*(G5 + G6/aL*(UL - x[j]/t[i]))**G3
            elif STL <= x[j]/t[i] and x[j]/t[i] <= US:
                D[i][j] = DSL
                U[i][j] = US
                P[i][j] = PS

AR = G5/DR
BR = G6*PR
if PS > PR: # The right shock wave
    DSR = DR*(PS/PR + G6)/(G6*PS/PR + 1.0)
    SR = UR + aR*np.sqrt(G2*PS/PR + G1)     # The right shock speed
    for i in range(1, Nt):
        for j in range(0, Nx):
            if US <= x[j]/t[i] and x[j]/t[i] <= SR:
                D[i][j] = DSR
                U[i][j] = US
                P[i][j] = PS
            elif x[j]/t[i] >= SR:
                D[i][j] = DR
                U[i][j] = UR
                P[i][j] = PR

else:   # The right rarefaction wave
    DSR = DR*(PS/PR)**(1.0/GAMMA)
    aSR = aR*(PS/PR)**G1   # The sound speed behind rarefaction wave
    SHR = UR + aR   # The speed of the head of the wave
    STR = US + aSR  # The speed of the tail of the wave
    # Density, Velocity, Pressure inside the fan wave
    DRF = np.zeros([Nt, Nx])
    URF = np.zeros([Nt, Nx])
    PRF = np.zeros([Nt, Nx])
    for i in range(1, Nt):
        for j in range(0, Nx):
            if x[j]/t[i] >= SHR:
                D[i][j] = DR
                U[i][j] = UR
                P[i][j] = PR
            elif STR <= x[j]/t[i] and x[j]/t[i] <= SHR:
                D[i][j] = DR*(G5 - G6/aR*(UR - x[j]/t[i]))**G4
                U[i][j] = G5*(-aR + G7*UR + x[j] / t[i])
                P[i][j] = PR*(G5 - G6/aR*(UR - x[j]/t[i]))**G3
            elif US <= x[j]/t[i] and x[j]/t[i] <= STR:
                D[i][j] = DSR
                U[i][j] = US
                P[i][j] = PS

for i in range(1, Nt):
    plt.figure(1)
    # plt.axis([-1, 1, 0, 1.5])
    # plt.plot(x, D[i])
    # plt.figure(2)
    plt.axis([0, 1, 0, 1])
    plt.plot(x, U[:][i])
    # plt.figure(3)
    # plt.axis([0, 1, 0, 1])
    # plt.plot(x, P[:][i])
    plt.show()
    

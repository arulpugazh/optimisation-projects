import numpy as np
from scipy.optimize import minimize, Bounds
import math
from prettytable import PrettyTable
import seaborn as sns

banana_pt = PrettyTable(["x1","x2"])
egg_pt = PrettyTable(["x1","x2"])
gol_pt = PrettyTable(["x1","x2","x3","x4","x5","x6","x7"])

# Rosenbrok function

def banana(x):
    print(x)
    x1, x2 = list(x)
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

def banana_callback(xp):
    x1, x2 = list(xp)
    banana_pt.add_row([x1, x2])

def eggcrate(x):
    x1, x2 = list(x)
    return x1 ** 2 + x2 ** 2 + 25 * (math.sin(x1) ** 2 + math.sin(x2) ** 2)

def egg_callback(xp):
    x1, x2 = list(xp)
    egg_pt.add_row([x1, x2])

def golinksi(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 0.7854 * x1 * (x2 ** 2) * (3.3333 * (x3 ** 2) + 14.9334 * x3 - 43.0934) \
           - 1.5079 * x1 * ((x6 ** 2) + (x7 ** 2)) + 7.477 * ((x6 ** 3) + (x7 ** 3)) \
           + 0.7854 * (x4 * (x6 ** 2) + (x5 * (x7 ** 2)))

def gol_cons1(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return - (27 * (x1 ** -1) * (x2 ** -2) * (x3 ** -1) - 1)


def gol_cons2(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return - (397.5 * (x1 ** -1) * (x2 ** -2) * (x3 ** -2) - 1)


def gol_cons3(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return - (1.93 * (x2 ** -1) * (x3 ** -1) * (x4 ** 3) * (x6 ** -4) - 1)

def gol_cons4(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return - (1.93 * (x2 ** -1) * (x3 ** -1) * (x5 ** 3) * (x7 ** -4) - 1)

def gol_cons5(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    numerator = ((745 * x4 * (x2**-1) *(x3**-1))**2 + 16.9* (10**6)) ** 0.5
    denominator = (110.0 * (x6**3))
    return 1 - (numerator/denominator)

def gol_cons6(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    numerator = ((745 * x5 * (x2**-1) *(x3**-1))**2 + 157.5* (10**6)) ** 0.5
    denominator = (85.0 * (x7**3))
    return 1 - (numerator/denominator)

def gol_cons7(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - (x2*x3/40)

def gol_cons8(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - (5*x2/x1)

def gol_cons9(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - (x1/(12*x2))

def gol_cons24(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - ((1.5 * x6 + 1.9) * (x4**-1))

def gol_cons25(x):
    x1, x2, x3, x4, x5, x6, x7 = list(x)
    return 1 - ((1.1 * x7 + 1.9) * (x5**-1))

def gol_callback(xk):
    x1, x2, x3, x4, x5, x6, x7 = list(xk)
    gol_pt.add_row([x1, x2, x3, x4, x5, x6, x7])

def run_exercise2():
    con1 = {'type': 'ineq', 'fun': gol_cons1}
    con2 = {'type': 'ineq', 'fun': gol_cons2}
    con3 = {'type': 'ineq', 'fun': gol_cons3}
    con4 = {'type': 'ineq', 'fun': gol_cons4}
    con5 = {'type': 'ineq', 'fun': gol_cons5}
    con6 = {'type': 'ineq', 'fun': gol_cons6}
    con7 = {'type': 'ineq', 'fun': gol_cons7}
    con8 = {'type': 'ineq', 'fun': gol_cons8}
    con9 = {'type': 'ineq', 'fun': gol_cons9}
    con24 = {'type': 'ineq', 'fun': gol_cons24}
    con25 = {'type': 'ineq', 'fun': gol_cons25}
    cons = ([con1, con2, con3, con4, con5, con6, con7, con8, con9, con24, con25])
    #bounds
    b1=(2.6,3.6)
    b2=(0.7,0.8)
    b3=(17,28)
    b4=(7.3,8.3)
    b5=(7.3,8.3)
    b6=(2.9,3.9)
    b7=(5,5.5)
    bounds = (b1,b2,b3,b4,b5,b6,b7)

    banana_bounds = Bounds(-5, 5)
    bsol = minimize(banana, x0=[0.0, 0.0], bounds=banana_bounds, callback=banana_callback, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    print(banana_pt)

    eggcrate_bounds = Bounds(0 - (2 * math.pi), int(2 * math.pi))
    eggsol = minimize(eggcrate, x0=[1.0, 1.0], bounds=eggcrate_bounds, callback=egg_callback, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
    print(egg_pt)

    methods = ['Nelder-Mead', 'Powell','CG','BFGS','L-BFGS-B','TNC','COBYLA','SLSQP']
    for method in methods:
        gol_sol = minimize(golinksi, x0=[3.0, 0.75,23,8.0,8.0,3.5, 5.25], callback=gol_callback, method=method, bounds=bounds, constraints=cons, options={'xtol': 1e-8, 'disp': True})
        print(f"{method}: {gol_sol.x}")

    #SLSQP
    gol_sol = minimize(golinksi, x0=[3.0, 0.75,23,8.0,8.0,3.5, 5.25], callback=gol_callback, method='SLSQP', bounds=bounds, constraints=cons, options={'xtol': 1e-8, 'disp': True})
    print(gol_sol)
    print(gol_pt)

run_exercise2()
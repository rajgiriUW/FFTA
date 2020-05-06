"""parab.py: Parabola fit around three points to find a true vertex."""

import numpy as np

def fit_new(f, x):
    '''
    Uses solution to parabola to fit peak and two surrounding points
    This assumes there is a peak (i.e. parabola second deriv is negative)
    
    f : array f(x)
    x : array x with the indices corresponding to f
    
    
    If interested, this is educational to see with sympy
     import sympy
     y1, y2, y3 = sympy.symbols('y1 y2 y3')
     A = sympy.Matrix([[(-1)**2, -1, 1],[0**2, 0, 1],[(1)**2,1,1]]) 
     C = sympy.Matrix([[y1],[y2],[y3]])
     D = AA.inv().multiply(CC)
     D contains the values of a, b, c in ax**2 + bx + c
     Peak position is at x = -D[1]/(2D[0])   
    '''

    pk = np.argmax(np.abs(f))
    
    y1 = f[pk-1]
    y2 = f[pk]
    y3 = f[pk+1]
    
    a = 0.5 * y1 - y2 + 0.5 * y3
    b = -0.5 * y1 + 0.5 * y3
    c = y2
    
    xindex = pk + -b/(2*a)
    yindex = a*(xindex-pk)**2 + b*(xindex-pk) + c
    findex = xindex * (x[1] - x[0]) - 1 
    
    return findex, yindex

def fit(f, x):
    """
    f = array
    x = index of peak, typically just argmax

    Uses parabola equation to fit to the peak and two surrounding points
    """

    x1 = x - 1
    x2 = x
    x3 = x + 1

    y1 = f[x - 1]
    y2 = f[x]
    y3 = f[x + 1]

    d = (x1 - x3) * (x1 - x2) * (x2 - x3)

    A = (x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)) / d

    B = (x1 ** 2.0 * (y2 - y3) +
         x2 ** 2.0 * (y3 - y1) +
         x3 ** 2.0 * (y1 - y2)) / d

    C = (x2 * x3 * (x2 - x3) * y1 +
         x3 * x1 * (x3 - x1) * y2 +
         x1 * x2 * (x1 - x2) * y3) / d

    xindex = -B / (2.0 * A)
    yindex = C - B ** 2.0 / (4.0 * A)

    return xindex, yindex
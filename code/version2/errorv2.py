import cmath
import numpy as np

def error(x, xb, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    a = (g0 - g0.mean())
    b = (g1 - g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    y = xb
    error = (p - 2*p*y + y**2)/(1 - 2*y + 2*y**2)
    return error

def deriv(p, xb):
    y = xb
    deriv = ((2*y**2 - 2*y)*(2*p - 1))/(1 - 2*y + 2*y**2)**2
    return deriv




# def errorMenWom(x, xb, g0, g1):
#     x = x.reshape(-1,1)
#     g0 = g0.reshape(1,-1)
#     g1 = g1.reshape(1,-1)
#     denom = (1 + xb**2)
#     c = (1 + x*xb)/denom
#     bi_men = x * g1
#     bi_men_post = (x-c*xb)*g1.mean()
#     bi_wom = g0
#     bi_wom_post = (1-c)*g0.mean()
#     error = (bi_men + bi_wom - (bi_men_post + bi_wom_post))**2
#     return error.mean(axis=1).mean()



def errorMenWom2(x, xb, g0, g1):
    x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    c = (1 + x*xb)/(1 + xb**2)
    w_men = (x - c*xb)*(g1 - g1.mean()) 
    w_wom = (1 - c)*(g0 - g0.mean()) 
    error = (w_men + w_wom)**2
    return error.mean(axis=1).mean()


def errorMenWom3(x, xb, g0, g1):
    x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    c = (1 + x*xb)/(1 + xb**2)
    w_men = (x - c*xb)*(g1 - g1.mean()) 
    w_wom = (1 - c)*(g0 - g0.mean()) 
    error = w_men**2 + w_wom**2
    return error.mean(axis=1).mean()

def errorMenWom4(x, xb, g0, g1):
    x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    c = (1 + x*xb)/(1 + xb**2)
    w_men = (x - c*xb)*(g1 - g1.mean()) 
    w_wom = (1 - c)*(g0 - g0.mean()) 
    error = w_men**2*np.var(g1) + w_wom**2*np.var(g0)
    return error.mean()


def errorMenWom5(p, xb, v0, v1):
    # g0 = g0.reshape(1,-1)
    # g1 = g1.reshape(1,-1)
    denom = (v0 + xb**2*v1)
    # w_men = (x - c*xb)*(g1 - g1.mean()) 
    # w_wom = (1 - c)*(g0 - g0.mean()) 
    # error = (xb*(xb-2*p)/denom**2)*np.var(g1) + (xb**2*(xb**2 - 2*p*xb + p)/denom**2)*np.var(g0)
    error = v0*v1*((1-p)*xb**2 + p*(xb-1)**2)/denom
    return error

# def errorMenWom5Rev(p, xb, g0, g1):
#     g0 = g0.reshape(1,-1)
#     g1 = g1.reshape(1,-1)
#     yb = (1-xb)
#     denom = (yb**2 + xb**2)
#     w_wom = 1 - (yb**2)/denom
#     w_men = 1 - (xb**2)/denom
#     error = ((1-p)*w_wom*np.var(g0) + p*w_men*np.var(g1))
#     return error



# def errorMenWom6(p, xb, g0, g1):
#     # x = x.reshape(-1,1)
#     g0 = g0.reshape(1,-1)
#     g1 = g1.reshape(1,-1)
#     # a = (g0 - g0.mean())
#     # b = (g1 - g1.mean())
#     yb = (1-xb)
#     # b2 = (b**2).mean()
#     # a2 = (a**2).mean()
#     w_men = (yb**2)/(yb**2 + xb**2)
#     w_wom = (xb**2)/(yb**2 + xb**2)
#     error = ((1-p)*w_wom*np.var(g0) + p*w_men*np.var(g1))
#     return error


# def errorMenWom7(p, xb, g0, g1):
#     # x = x.reshape(-1,1)
#     g0 = g0.reshape(1,-1)
#     g1 = g1.reshape(1,-1)
#     # a = (g0 - g0.mean())
#     # b = (g1 - g1.mean())
#     yb = (1-xb)
#     # b2 = (b**2).mean()
#     # a2 = (a**2).mean()
#     denom = (yb**2 + xb**2)
#     w_men = p*(yb**4 )/denom**2 +(1-p)*yb**2*xb**2/denom**2
#     w_wom = (1-p)*(xb**4 )/denom**2 +p*yb**2*xb**2/denom**2
#     error = (w_wom*np.var(g0) + w_men*np.var(g1))
#     # error = ((1-p)*(xb**2/denom) + p*(yb**2/denom))
#     return error


def splitderiv2(p, xb, v0, v1):
    # g0 = g0.reshape(1,-1)
    # g1 = g1.reshape(1,-1)
    # v0 = np.var(g0)
    # v1 = np.var(g1)
    denom = (v0 + xb**2*v1)
    deriv = 2*v0*v1*(v0*(xb - p) + v1*p*(xb-1)*xb)/denom**2
    return deriv



def opt_prop(p, v0, v1):
    opt1 = ((v1*p - v0) + cmath.sqrt(v0**2 + 2*v0*v1*p*(2*p -1) + v1**2 * p**2))/(2*v1*p)
    opt2 = ((v1*p - v0) - cmath.sqrt(v0**2 + 2*v0*v1*p*(2*p -1) + v1**2 * p**2))/(2*v1*p)
    return (opt1, opt2)

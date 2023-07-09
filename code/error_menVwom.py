import numpy as np

def error(x, xb, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    a = (g0 - g0.mean())
    b = (g1 - g1.mean())
    # b2_prev = 
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    y = xb
    error = (p - 2*p*y + y**2)/(1 - 2*y + 2*y**2)
    return error

def deriv(p, xb):
    y = xb
    deriv = ((2*y**2 - 2*y)*(2*p - 1))/(1 - 2*y + 2*y**2)**2
    return deriv




def errorMenWom(x, xb, g0, g1):
    x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    a = (g0 - g0.mean())
    b = (g1 - g1.mean())
    y = (1-x)
    yb = (1-xb)
    # b2_prev = 
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    c = (1 - x - xb + 2*x*xb)/(1 - 2*xb + 2*xb**2)
    bi_men = x * g1
    bi_men_post = (x-c*xb)*g1.mean() + c*xb*g1
    bi_wom = y*g0
    bi_wom_post = (y-c*yb)*g0.mean() + c*yb*g0
    error = (bi_men + bi_wom - (bi_men_post + bi_wom_post))**2
    return error.mean(axis=1).mean()




def errorMenWom2(x, xb, g0, g1):
    x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    a = (g0 - g0.mean())
    b = (g1 - g1.mean())
    y = (1-x)
    yb = (1-xb)
    # b2_prev = 
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    c = (1 - x - xb + 2*x*xb)/(1 - 2*xb + 2*xb**2)
    w_men = (x - c*xb)*(g1 - g1.mean()) 
    w_wom = (y - c*yb)*(g0 - g0.mean()) 
    error = (w_men + w_wom)**2
    return error.mean(axis=1).mean()


def errorMenWom3(x, xb, g0, g1):
    x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    a = (g0 - g0.mean())
    b = (g1 - g1.mean())
    y = (1-x)
    yb = (1-xb)
    # b2_prev = 
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    c = (1 - x - xb + 2*x*xb)/(1 - 2*xb + 2*xb**2)
    w_men = (x - c*xb)*(g1 - g1.mean()) 
    w_wom = (y - c*yb)*(g0 - g0.mean()) 
    error = (w_men + w_wom)**2
    return error.mean(axis=1).mean()

def errorMenWom4(x, xb, g0, g1):
    x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    # a = (g0 - g0.mean())
    # b = (g1 - g1.mean())
    y = (1-x)
    yb = (1-xb)
    # b2_prev = 
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    # c = (1 - x - xb + 2*x*xb)/(1 - 2*xb + 2*xb**2)
    c = (y*yb + x*xb)/(yb**2 + xb**2)
    w_men = (x - c*xb)**2*np.var(g1)
    w_wom = (y - c*yb)**2*np.var(g0)
    error = (w_men + w_wom)
    return error.mean()


def errorMenWom5(p, xb, g0, g1):
    # x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    # a = (g0 - g0.mean())
    # b = (g1 - g1.mean())
    yb = (1-xb)
    # b2_prev = 
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    w_wom = 1 - (yb**2)/(yb**2 + xb**2)
    w_men = 1 - (xb**2)/(yb**2 + xb**2)
    error = ((1-p)*w_wom*np.var(g0) + p*w_men*np.var(g1))
    return error




def errorMenWom6(p, xb, g0, g1):
    # x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    # a = (g0 - g0.mean())
    # b = (g1 - g1.mean())
    yb = (1-xb)
    # b2_prev = 
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    w_men = (yb**2)/(yb**2 + xb**2)
    w_wom = (xb**2)/(yb**2 + xb**2)
    error = ((1-p)*w_wom*np.var(g0) + p*w_men*np.var(g1))
    return error




def splitderiv2(p, xb, g0, g1):
    # x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    # a = (g0 - g0.mean())
    # b = (g1 - g1.mean())
    yb = (1-xb)
    # b2_prev = 
    # b2 = (b**2).mean()
    # a2 = (a**2).mean()
    w_men = (2*xb*(1-xb))/(2*xb**2-2*xb+1)**2
    w_wom = -(2*xb*(1-xb))/(2*xb**2-2*xb+1)**2
    deriv = (1-p)*w_wom + p*w_men
    return deriv









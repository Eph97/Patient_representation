import numpy as np
import pandas as pd
# def deriv(p, xb):
#     p1 = p
#     p2 = 1  - p
#     deriv_res = ((4 - 8*xb) * ((p2 * xb**4 - 2 * p * p2 * xb**3 + 2 * p*p2*xb**4 + p*xb**2 -2*p*xb**3 + p*xb**4) +
#         (p*(1-xb)**4 - 2*(xb - xb**2 - 2*xb**2 + 2*xb**3 - xb**4)*p2*p + p2*(xb**2 - 2*xb**3 + p*xb**4)) +
#         2*(p*xb**2 - xb*p**2 - p*xb**3 + p*xb**2 - xb**3 + p*xb**2 + xb**4 - p*xb**3)) - (2*xb**2 - 2*xb + 1)*((p2 * xb**4 - 2 * p * p2 * xb**3 + 2 * p*p2*xb**4 + p*xb**2 -2*p*xb**3 + p*xb**4) +
#             (p2*(2*xb - 6*xb**2 + 4*xb**3) - 4*p*(1 - xb)**3 - (2 + 4*xb + 8*xb - 12*xb**2 + 16*xb**3 )*p*p2 )+
#             2*(2*p*xb - p**2 - 3*p*xb**2 + 2*p*xb - 3*xb**2 + 2*p*xb + 4*xb**3 - 3*p*xb**2)))
#     return deriv_res


def deriv(p, xb):
    p1 = p
    p2 = 1  - p
    deriv_res = (4 - 8*xb) * (xb**4 *(4) + (xb**3 *(-6* p2* p - 8*p - 4) ) + 
            (xb**2 *(18*p - 8*p**2)) + 
            (xb *(-4*p -2 *p2* p - 2 *p**2)) + 
            p) - (2*xb**2 - 2*xb +1) *((xb**3 *(4 - 8* p *p2 )) + (xb**2*(-18*p *p2 - 24*p - 12)) + xb*(24* p - 12 *p* p2 + 2) + (-4*p -2 *p*p2))
    return deriv_res




def error(p, xb, x, v1, v2):
    xb2 = xb
    xb1 = 1 - xb2
    x1 = x[:,0]
    x2 = x[:,1]
    # p1 = 1-p
    # p2 = p
    # x1 = np.random.binomial(1, p, size=num_ind)
    # x2 = 1 - x1
    # error = ((x2 * xb**4 - 2 * x1 * x2 * xb**3 + 2 * x1*x2*xb**4 + x1*xb**2 -2*x1*xb**3 + x1*xb**4) +
    #         (x1*(1-xb)**4 - 2*(xb - xb**2 - 2*xb**2 + 2*xb**3 - xb**4)*x2*x1 + x2*(xb**2 - 2*xb**3 + x1*xb**4)) +
    #         2*(x1*xb**2 - xb*x1**2 - x1*xb**3 + x1*xb**2 - xb**3 + x1*xb**2 + xb**4 - x1*xb**3))/(2*xb**2 - 2*xb + 1)**2
    # error = (((1-p)*xb**4  - 2* (1-p)* (xb**3 )*p* (1 - xb) + p* (1 - xb)**2 * xb**2)
    # + (p* (1 - xb)**4  - 2 *p *(1 - xb)**3 *(1-p)*xb + (1-p)*(1 - xb)**2 * xb**2)
    # +  2 *((1-p)* xb**2 - p *(1 - xb)* xb)* (p *(1 - xb)**2 - (1-p)* (1 - xb)* xb)) /((1 - xb)**4 + 2*(1 - xb)**2* xb**2 + xb**4 )
    error = ((x1**2* xb2**4  - 2 *x1* xb2**3 *x2* xb1 + x2**2*xb1**2 *xb2**2)*v1 + 
            (x2**2 *xb1**4  - 2* x2 *xb1**3* x1* xb2 + x1**2 *xb1**2 *xb2**2)*v2 + 
            2*v1*v2*(x1* xb2**2 - x2* xb1 *xb2) *(x2* xb1**2 - x1*xb1* xb2)) / (xb1**4 + 2*xb1**2 *xb2**2 + xb2**4)
    return error.mean()



def error2(x, xb, g0, g1):
    x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - (g0.mean() + g1.mean()))
    c = (1 - x - xb + 2*x*xb)/(1 - 2*xb + 2*xb**2)
    # error = (g0 + x*(g1 - g0) - ( 1 -c )*(g0.mean() + g1.mean()) - c*(g0 + xb * (g1 - g0)))**2
    error = (g0 + x*(g1 - g0) - ((1 -x - c*(1-xb))*g0.mean() + (x - c*xb)*g1.mean() + c*(g0 + xb * (g1 - g0))))**2
    # error = (g0 + x*(g1 - g0) -  c*(g0 + xb * (g1 - g0)))**2
    return error.mean(axis=1).mean(axis=0)


def error3(x, xb, g0, g1):
    x = x.reshape(-1,1)
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - (g0.mean() + g1.mean()))
    c = (1 - x - xb + 2*x*xb)/(1 - 2*xb + 2*xb**2)
    error = b**2 * (x**2 + 2*c*x + xb**2*c**2) + 2*a*b*(x + c*x + c*xb + c**2 * xb) + a**2*(1 + 2*c+ c**2)
    return error.mean(axis=1).mean(axis=0)


def error4(p, xb, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    denom = (1 - 2*xb + 2*xb**2)
    # error = (b2) * (p - 2*p*xb**2/denom + xb**2*(p*(2*xb -1) + (xb -1)**2)/denom**2) + \
    error = (b2) * (p - 2*p*xb**2/denom + (xb**4 + 2*xb**3*(p-1) + xb**2*(1-p))/denom**2) + \
    (2*ab)*(p - p*xb / denom - xb*(xb*(2*p -1) + (1-p)) / denom + xb*(p*(2*xb -1) + (xb - 1)**2)/denom**2) + (a2)*(1 - 2*(xb*(2*p - 1) + (1 - p))/denom + (p*(2*xb - 1) + (xb - 1)**2)/denom**2)
    return error


def error5(p, xb, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    denom = (1 - 2*xb + 2*xb**2)
    # error = (b2) * (p - 2*p*xb**2/denom + xb**2*(p*(2*xb -1) + (xb -1)**2)/denom**2) + \
    error = (b2) * (p - 2*p*xb**2/denom + (xb**4 + 2*xb**3*(p-1) + xb**2*(1-p))/denom**2) + \
    (2*ab)*(p - (xb**2*(2*p - 1) + xb) / denom + (xb**3 + 2*xb**2*(p-1) + xb*(1-p))/denom**2) + (a2)*(1 - 2*(xb*(2*p - 1) + (1 - p))/denom + (p*(2*xb - 1) + (xb - 1)**2)/denom**2)
    # (2*ab)*(p - (xb**2*(2*p - 1) + xb) / denom + xb*(p*(2*xb -1) + (xb - 1)**2)/denom**2) + (a2)*(1 - 2*(xb*(2*p - 1) + (1 - p))/denom + (p*(2*xb - 1) + (xb - 1)**2)/denom**2)
    return error

def error6(p, xb, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    denom = (1 - 2*xb + 2*xb**2)
    error = (b2) * (p - 2*p*xb**2/denom + (xb**4 + 2*xb**3*(p-1) + xb**2*(1-p))/denom**2) + \
    (2*ab)*(p - (xb**2*(2*p - 1) + xb) / denom + (xb**3 + 2*xb**2*(p-1) + xb*(1-p))/denom**2) + (a2)*(1 - 2*(xb*(2*p - 1) + (1 - p))/denom + (xb**2 + 2*xb*(p - 1) + (1-p))/denom**2)
    return error

def error7(p, xb, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    denom = (1 - 2*xb + 2*xb**2)
    # error = (b2*p + 2*ab*p + a2) - (xb**2*(2*p*b2 + 2*ab * (2*p-1)) + xb*(2*ab + 2*a2*(2*p-1))+2*a2*(1-p) )/denom  + (b2*xb**4 + 2*xb**3*(b2*(p-1) + ab) + xb**2*(b2*(1-p) + 4*ab*(p-1) + a2) + \
    #         2*xb*( ab*(1-p) + a2*(p-1)) + a2*(1-p))/(denom**2)
    error2 = (b2*p + 2*ab*p + a2) - (xb**2*(2*p*b2 + 2*ab*(2*p-1)) + xb*(2*ab + 2*a2*(2*p-1))+2*a2*(1-p) )/denom  + (b2*xb**4 + 2*xb**3*(b2*(p-1) + ab) + xb**2*((b2 - 4*ab)*(1-p) + a2) + \
            2*xb*((ab-a2)*(1-p)) + a2*(1-p))/(denom**2)
    # assert np.abs(error - error2) < 0.0001
    # print(error, error2)
    return error2

def error8(p, xb, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - (g0.mean() - g1.mean()))
    b2 = (b**2).mean()
    a2 = g0.var() +(g1.mean())**2
    c = (g0.mean())**2 - g0.var() - (g0*g1).mean()
    ab = c
    denom = (1 - 2*xb + 2*xb**2)
    # error = (b2*p + 2*ab*p + a2) - (xb**2*(2*p*b2 + 2*ab * (2*p-1)) + xb*(2*ab + 2*a2*(2*p-1))+2*a2*(1-p) )/denom  + (b2*xb**4 + 2*xb**3*(b2*(p-1) + ab) + xb**2*(b2*(1-p) + 4*ab*(p-1) + a2) + \
    #         2*xb*( ab*(1-p) + a2*(p-1)) + a2*(1-p))/(denom**2)
    error2 = (b2*p + 2*ab*p + a2) - (xb**2*(2*p*b2 + 2*ab*(2*p-1)) + xb*(2*ab + 2*a2*(2*p-1))+2*a2*(1-p) )/denom  + (b2*xb**4 + 2*xb**3*(b2*(p-1) + ab) + xb**2*((b2 - 4*ab)*(1-p) + a2) + \
            2*xb*((ab-a2)*(1-p)) + a2*(1-p))/(denom**2)
    # assert np.abs(error - error2) < 0.0001
    # print(error, error2)
    return error2



def deriv2(p, xb, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    denom = (1 - 2*xb + 2*xb**2)
    deriv =  (xb**2*(2*p*b2 + 2*ab*(2*p-1)) + xb*(2*ab + 2*a2*(2*p-1))+2*a2*(1-p) )*(4*xb -2 )/denom**2  -\
              (2*xb*(2*p*b2 + 2*ab*(2*p-1)) + (2*ab + 2*a2*(2*p-1)))/denom  -\
            (b2*xb**4 + 2*xb**3*(b2*(p-1) + ab) + xb**2*((b2 - 4*ab)*(1-p) + a2) + 2*xb*((ab-a2)*(1-p)) + a2*(1-p))*(8*xb -4 )/denom**3  + \
            (4*b2*xb**3 + 6*xb**2*(b2*(p-1) + ab) + 2*xb*((b2 - 4*ab)*(1-p) + a2) + 2*((ab-a2)*(1-p)))/denom**2
    return deriv


def deriv3(p, x, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    c = (a*b).mean()
    ab = c
    denom = (1 - 2*x + 2*x**2)
    deriv =  (x**2*(2*p*b2 + 2*ab*(2*p-1)) + x*(2*ab + 2*a2*(2*p-1))+2*a2*(1-p) )*(4*x -2 )/denom  -\
              (2*x*(2*p*b2 + 2*ab*(2*p-1)) + (2*ab + 2*a2*(2*p-1)))  -\
            (b2*x**4 + 2*x**3*(b2*(p-1) + ab) + x**2*((b2 - 4*ab)*(1-p) + a2) + 2*x*((ab-a2)*(1-p)) + a2*(1-p))*(8*x -4 )/denom**2  + \
            (4*b2*x**3 + 6*x**2*(b2*(p-1) + ab) + 2*x*((b2 - 4*ab)*(1-p) + a2) + 2*((ab-a2)*(1-p)))/denom
    return deriv


def deriv4(p, x, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    c = ab
    denom = (1 - 2*x + 2*x**2)
    deriv =  (x**5*(16*b2*p+32*c*p-16*c)+\
                x**4*(32*a2*p-16*a2-24*b2*p-48*c*p+40*c) +\
                x**3*(-64*a2*p+40*a2+16*b2*p+32*c*p-40*c) +\
                x**2*(56*a2*p-40*a2-4*b2*p-8*c*p+20*c) +\
                x*(-24*a2*p+20*a2-4*c)+(4*a2*p-4*a2))/denom**2 -\
            (x**5*(16*b2*p+32*c*p-16*c) +\
                x**4*(16*a2*p-8*a2 -32*b2*p-64*c*p +40*c) +\
                x**3*(-32*a2*p+16*a2+32*b2*p+64*c*p -48*c) +\
                x**2*(32*a2*p-16*a2-16*b2*p-32*c*p+32*c) +\
                x*(-16*a2*p +8*a2 +4*b2*p+8*c*p -12*c) +\
            (4*a2*p-2*a2+2*c))/denom**2 -\
                (8*x**3*(a2+(1-p)*(b2-4*c))-4*x**2*(a2+(1-p)*(b2-4*c)) +\
                16*(1-p)*x**2*(c-a2)-\
                8*(1-p)*x*(c-a2)+8*a2*(1-p)*x-\
                4*a2*(1-p)+16*x**4*(b2*(p-1)+c)-\
                8*x**3*(b2*(p-1)+c)+8*b2*x**5-4*b2*x**4)/denom**2  + \
        (x**5*(8*b2) +\
            x**4*(12*ab+12*b2*p-20*b2) +\
            x**3*(4*a2+16*ab*p-28*ab-16*b2*p+20*b2) +\
            x**2*(4*a2*p-8*a2-20*ab*p+26*ab+10*b2*p-10*b2) +\
            x*(-4*a2*p+6*a2+12*ab*p-12*ab-2*b2*p+2*b2) +\
            (2*a2*p-2*a2-2*ab*p+2*ab))/denom**2
            # (4*b2*x**3 + 6*x**2*(b2*(p-1) + ab) + 2*x*((b2 - 4*ab)*(1-p) + a2) + 2*((ab-a2)*(1-p)))/denom
    return deriv

def deriv5(p, x, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    c = ab
    denom = (1 - 2*x + 2*x**2)
    deriv =  (x**5*(16*b2*p+32*c*p-16*c)+\
                x**4*(32*a2*p-16*a2-24*b2*p-48*c*p+40*c) +\
                x**3*(-64*a2*p+40*a2+16*b2*p+32*c*p-40*c) +\
                x**2*(56*a2*p-40*a2-4*b2*p-8*c*p+20*c) +\
                x*(-24*a2*p+20*a2-4*c)+(4*a2*p-4*a2)) -\
            (x**5*(16*b2*p+32*c*p-16*c) +\
                x**4*(16*a2*p-8*a2 -32*b2*p-64*c*p +40*c) +\
                x**3*(-32*a2*p+16*a2+32*b2*p+64*c*p -48*c) +\
                x**2*(32*a2*p-16*a2-16*b2*p-32*c*p+32*c) +\
                x*(-16*a2*p +8*a2 +4*b2*p+8*c*p -12*c) +\
                (4*a2*p-2*a2+2*c)) -\
            (8*x**3*(a2+(1-p)*(b2-4*c))-4*x**2*(a2+(1-p)*(b2-4*c)) +\
                16*(1-p)*x**2*(c-a2)-\
                8*(1-p)*x*(c-a2)+8*a2*(1-p)*x-\
                4*a2*(1-p)+16*x**4*(b2*(p-1)+c)-\
                8*x**3*(b2*(p-1)+c)+8*b2*x**5-4*b2*x**4)  + \
        (x**5*(8*b2) +\
            x**4*(12*ab+12*b2*p-20*b2) +\
            x**3*(4*a2+16*ab*p-28*ab-16*b2*p+20*b2) +\
            x**2*(4*a2*p-8*a2-20*ab*p+26*ab+10*b2*p-10*b2) +\
            x*(-4*a2*p+6*a2+12*ab*p-12*ab-2*b2*p+2*b2) +\
            (2*a2*p-2*a2-2*ab*p+2*ab))
            # (4*b2*x**3 + 6*x**2*(b2*(p-1) + ab) + 2*x*((b2 - 4*ab)*(1-p) + a2) + 2*((ab-a2)*(1-p)))/denom
    return deriv


def deriv6(p, x, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    c = ab
    denom = (1 - 2*x + 2*x**2)
    deriv =  (x**5*(16*b2*p+32*c*p-16*c)+\
                x**4*(32*a2*p-16*a2-24*b2*p-48*c*p+40*c) +\
                x**3*(-64*a2*p+40*a2+16*b2*p+32*c*p-40*c) +\
                x**2*(56*a2*p-40*a2-4*b2*p-8*c*p+20*c) +\
                x*(-24*a2*p+20*a2-4*c)+(4*a2*p-4*a2)) -\
            (x**5*(16*b2*p+32*c*p-16*c) +\
                x**4*(16*a2*p-8*a2 -32*b2*p-64*c*p +40*c) +\
                x**3*(-32*a2*p+16*a2+32*b2*p+64*c*p -48*c) +\
                x**2*(32*a2*p-16*a2-16*b2*p-32*c*p+32*c) +\
                x*(-16*a2*p +8*a2 +4*b2*p+8*c*p -12*c) +\
                (4*a2*p-2*a2+2*c)) -\
            (x**5*(8*b2)+\
                x**4*(16*b2*p-16*b2+16*c-4*b2)+\
                x**3*(8*a2-8*b2*p+8*b2+32*c*p-32*c-8*b2*p+8*b2-8*c)+\
                x**2*(-4*a2+4*b2*p-4*b2-16*c*p+16*c+16*a2*p-16*a2-16*c*p+16*c)+\
                x*(-16*a2*p+ 16*a2+ 8*c*p- 8*c)+\
                (4*a2*p-4*a2)) +\
        (x**5*(8*b2) +\
            x**4*(12*ab+12*b2*p-20*b2) +\
            x**3*(4*a2+16*ab*p-28*ab-16*b2*p+20*b2) +\
            x**2*(4*a2*p-8*a2-20*ab*p+26*ab+10*b2*p-10*b2) +\
            x*(-4*a2*p+6*a2+12*ab*p-12*ab-2*b2*p+2*b2) +\
            (2*a2*p-2*a2-2*ab*p+2*ab))
            # (4*b2*x**3 + 6*x**2*(b2*(p-1) + ab) + 2*x*((b2 - 4*ab)*(1-p) + a2) + 2*((ab-a2)*(1-p)))/denom
    return deriv



def deriv7(p, x, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    c = ab
    denom = (1 - 2*x + 2*x**2)
    deriv = x**5*(16*b2*p+32*c*p-16*c)+\
                x**4*(32*a2*p-16*a2-24*b2*p-48*c*p+40*c) +\
                x**3*(-64*a2*p+40*a2+16*b2*p+32*c*p-40*c) +\
                x**2*(56*a2*p-40*a2-4*b2*p-8*c*p+20*c) +\
                x*(-24*a2*p+20*a2-4*c)+(4*a2*p-4*a2) -\
            x**5*(16*b2*p+32*c*p-16*c) -\
                x**4*(16*a2*p-8*a2 -32*b2*p-64*c*p +40*c) -\
                x**3*(-32*a2*p+16*a2+32*b2*p+64*c*p -48*c) -\
                x**2*(32*a2*p-16*a2-16*b2*p-32*c*p+32*c) -\
                x*(-16*a2*p +8*a2 +4*b2*p+8*c*p -12*c) -\
                (4*a2*p-2*a2+2*c) -\
            x**5*(8*b2)-\
                x**4*(16*b2*p-16*b2+16*c-4*b2)-\
                x**3*(8*a2-8*b2*p+8*b2+32*c*p-32*c-8*b2*p+8*b2-8*c)-\
                x**2*(-4*a2+4*b2*p-4*b2-16*c*p+16*c+16*a2*p-16*a2-16*c*p+16*c)-\
                x*(-16*a2*p+ 16*a2+ 8*c*p- 8*c)-\
                (4*a2*p-4*a2) +\
            x**5*(8*b2) +\
                x**4*(12*ab+12*b2*p-20*b2) +\
                x**3*(4*a2+16*ab*p-28*ab-16*b2*p+20*b2) +\
                x**2*(4*a2*p-8*a2-20*ab*p+26*ab+10*b2*p-10*b2) +\
                x*(-4*a2*p+6*a2+12*ab*p-12*ab-2*b2*p+2*b2) +\
                (2*a2*p-2*a2-2*ab*p+2*ab)
    return deriv



def deriv8(p, x, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - g0.mean() + g1.mean())
    b2 = (b**2).mean()
    a2 = (a**2).mean()
    ab = (a*b).mean()
    c = ab
    deriv = x**4*(16*a2*p-8*a2+4*b2*p+16*c*p-4*c) +\
            x**3*(-32*a2*p+20*a2-16*b2*p+4*b2-48*c*p+20*c) +\
            x**2*(12*a2*p-12*a2+18*b2*p-6*b2+36*c*p-18*c) +\
            x*(4*a2*p+2*a2-6*b2*p+2*b2-4*c*p+4*c) +\
            -2*a2*p-2*c*p
    return deriv



def deriv9(p, x, g0, g1):
    g0 = g0.reshape(1,-1)
    g1 = g1.reshape(1,-1)
    b = (g1 - g0)
    a = (g0 - (g0.mean() - g1.mean()))
    # b2_prev = 
    b2 = (b**2).mean()
    a2 = g0.var() +(g1.mean())**2
    a2_prev =(a**2).mean()
    c = (g0.mean())**2 - g0.var() - (g0*g1).mean()
    c_prev = (a*b).mean()
    # c = c_prev
    # print(c,c_prev)
    assert (np.abs(a2 - a2_prev) < 0.005)
    assert np.abs(c - c_prev) < 0.005
    deriv = x**4*(16*a2*p-8*a2+4*b2*p+16*c*p-4*c) +\
            x**3*(-32*a2*p+20*a2-16*b2*p+4*b2-48*c*p+20*c) +\
            x**2*(12*a2*p-12*a2+18*b2*p-6*b2+36*c*p-18*c) +\
            x*(4*a2*p-6*b2*p-4*c*p+2*a2+2*b2+4*c) +\
            -2*a2*p-2*c*p
    return deriv


def deriv10(p, x):
    # c = c_prev
    # print(c,c_prev)
    # assert (np.abs(c - c_prev) < 0.005)
    # deriv = x**4*(8*p - 4) + x**3*(8 - 16*p) + x**2*(12*p-6) + x*(4 -4*p)
    # deriv = x*(x**3*(4*p - 2) + x**2*(4 - 8*p) + x*(6*p-3) + (2 -2*p))
    deriv = x**3*(4*p - 2) + x**2*(4 - 8*p) + x*(6*p-3) + (2 -2*p)
    return deriv


def f_prime(p, x):
    # sol = x*(x**3*(4*p - 2) + x**2*(4 - 8*p) + x*(6*p-3) + (2 -2*p))
    deriv = 4*x**3*(4*p - 2) + 3*x**2*(4 - 8*p) + 2*x*(6*p-3) + (2 -2*p)
    return deriv

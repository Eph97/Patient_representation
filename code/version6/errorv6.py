import cmath
import numpy as np

def MSE(x, x_bar, c_i, g):
    b = g - g.mean(axis=0)
    xb = c_i*x_bar
    x_scaled = (x - xb)
    error = (np.dot(x_scaled, b.T)**2).mean()
    return error


def partials(x,p_1, p_2, p_3, b1, b2, b3):
    eqn1 = p_1*(x*(2*x-2*(1-x-y))*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2)-2*x*(b1-b3)+y*(b2-b3)+b3)*(b1-x*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2))+p_2*(y*(2*x-2*(1-x-y))*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2)-y*(b1-b3))*(b2-y*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2))+(p_3)*((1-x-y)*(2*x-2*(1-x-y))*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2)+x*(b1-b3)+y*(b2-b3)+b3-(b1-b3)*(1-x-y))*(b3-(1-x-y)*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2))
    eqn2 = p_1*(x*(2*y-2*(1-x-y))*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2)-x*(b2-b3))*(b1-x*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2))+p_2*(y*(2*y-2*(1-x-y))*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2)-x*(b1-b3)+2*y*(b2-b3)+b3)*(b2-y*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2))+(p_3)*((2*y-2*(1-x-y))*(1-x-y)*(x*(b1-b3)+y*(b2-b3)+b3)/(x^2+(1-x-y)^2+y^2)+x*(b1-b3)+y*(b2-b3)+b3-(b2-b3)*(1-x-y))*(b3-((1-x-y)*(x*(b1-b3)+y*(b2-b3)+b3))/(x^2+(1-x-y)^2+y^2)) == 0;
    return eqn1, eqn2


def anal_error(x, x_bar, g):
    b = gammas - gammas.mean(axis=0)
    p_1, p_2, p_3 = x.mean(axis=0)
    xb, yb, zb = x_bar[0]
    b = gammas - gammas.mean(axis=0)
    b1 = b[:,0]
    b2 = b[:,1]
    b3 = b[:,2]
    denom = np.dot(x_bar, x_bar.T)
    xi = (xb*b1+yb*b2+zb*b3)/denom
    w_1 = (b1-xb*xi)
    w_2 = (b2-yb*xi)
    w_3 = (b3-zb*xi)
    weighted_mean = p_1*w_1**2 + p_2*w_2**2 + p_3*w_3**2
    return (weighted_mean).mean()

def anal_error2(x, x_bar, gammas):
    # b = g - g.mean(axis=0)
    v1,v2,v3 = gammas.var(axis=0)
    p_1, p_2, p_3 = x.mean(axis=0)
    xb, yb, zb = x_bar[0]
    b = gammas - gammas.mean(axis=0)
    b1 = b[:,0]
    b2 = b[:,1]
    b3 = b[:,2]
    # denom = np.dot(x_bar, x_bar.T)
    denom = v1*xb**2 + v2*yb**2 + v3*zb**2
    numer = (xb*b1+yb*b2+zb*b3)
    xi = numer/denom
    w_1 = v1*(1-v1*xb**2/denom)
    w_2 = v2*(1-v2*yb**2/denom)
    w_3 = v3*(1-v3*zb**2/denom)
    weighted_mean = p_1*w_1 + p_2*w_2 + p_3*w_3
    return weighted_mean




def test(x, x_bar, g):
    b = g - g.mean(axis=0)
    p_1, p_2, p_3 = x.mean(axis=0)
    xb, yb, zb = x_bar[0]
    b = g - g.mean(axis=0)
    b1 = b[:,0]
    b2 = b[:,1]
    b3 = b[:,2]
    denom = np.dot(x_bar, x_bar.T)
    xi = (xb*b1+yb*b2+zb*b3)/denom
    # weighted_mean = p_1*(b1 - xb * xi)**2 + p_2*(b2 - yb*xi)**2 + (1 - p_1 - p_2)*(b3-(1 - xb - yb)*xi)**2
    weighted_mean = p_1*(1-2*xb**2/denom +xb**2*xi**2) + p_2*(1-2*yb**2/denom+yb**2*xi**2) + (1 - p_1 - p_2)*(1-2*(1-xb-yb)**2/denom+(1-xb-yb)**2*xi**2)
    # weighted_mean = ((1-p_2)*xb**2 + (1 - p_1)*yb**2 + (1 - p_1 - p_2)*(2*xb*yb - 2*xb - 2*yb + 1))*xi**2
    return (weighted_mean).mean()



def test_sol(x, xb,yb,zb):
    p1, p2,p3 = x.mean(axis=0)
    numer =(1-p1)*xb**2 + (1-p2)*yb**2 + (1-p3)*zb**2
    denom = xb**2 + yb**2 + zb**2
    return numer/denom


def sol_cov(x,x_bar, g):
    b = g - g.mean(axis=0)
    v1,v2,v3 = g.var(axis=0)
    p1, p2,p3 = x.mean(axis=0)
    xb, yb, zb = x_bar[0]
    denom = v1*xb**2 + v2*yb**2 + v3*zb**2
    weighted_mean =(p1*v1)*(1-v1*xb**2/denom)+ (p2*v2)*(1-v2*yb**2/denom) + (p3*v3)*(1-v3*zb**2/denom)
    return weighted_mean

def sol_cov2(x,x_bar, g):
    v1,v2,v3 = g.var(axis=0)
    p1, p2,p3 = x.mean(axis=0)
    xb, yb, zb = x_bar[0]
    numer =(p2*v2+ p3*v3)*v1*xb**2 + (p1*v1+ p3*v3)*v2*yb**2*v2 + (p1*v1+ p2*v2)*v3*zb**2*v3
    denom = v1*xb**2 + v2*yb**2 + v3*zb**2
    return numer/denom




def sol_cov3(x,x_bar, gammas, beta_ate):
    v1,v2,v3 = gammas.var(axis=0)
    g1 = gammas[:,0]
    g2 = gammas[:,1]
    g3 = gammas[:,2]
    p1, p2,p3 = x.mean(axis=0)
    xb, yb, zb = x_bar[0]
    cov1 = np.cov(g1,beta_ate)[0,1]
    cov2 = np.cov(g2,beta_ate)[0,1]
    cov3 = np.cov(g3,beta_ate)[0,1]
    vb = np.var(beta_ate)
    # weighted_mean =(p1*v1+p2*v2 +p3*v3)+(p1*cov1**2 + p2*cov2**2 +p3*cov3**2)/np.var(beta_ate)
    weighted_mean =(p1*v1+p2*v2 +p3*v3)-(p1*(cov1)**2 + p2*(cov2)**2 +p3 *(cov3)**2)/vb
    return weighted_mean

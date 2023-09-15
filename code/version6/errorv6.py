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
    b = g - g.mean(axis=0)
    p_1, p_2, p_3 = x.mean(axis=0)
    xb, yb, zb = x_bar[0]
    b = g - g.mean(axis=0)
    b1 = b[:,0]
    b2 = b[:,1]
    b3 = b[:,2]
    w_1 = (b1-xb*(xb*b1+yb*b2+zb*b3)/np.dot(x_bar, x_bar.T))
    w_2 = (b2-yb*(xb*b1+yb*b2+zb*b3)/np.dot(x_bar, x_bar.T))
    w_3 = (b3-zb*(xb*b1+yb*b2+zb*b3)/np.dot(x_bar, x_bar.T))
    weighted_mean = p_1*w_1**2 + p_2*w_2**2 + p_3*w_3**2
    return (weighted_mean).mean()





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



def test_sol(p1,p2,p3, x,y,z):

    numer =(1-p1)*x**2 + (1-p2)*y**2 + (1-p3)*z**2
    denom = x**2 + y**2 + z**2
    return numer/denom

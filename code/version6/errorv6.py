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



# def test(x, x_bar, g):
#     b = g - g.mean(axis=0)
#     p_1, p_2, p_3 = x.mean(axis=0)
#     xb, yb, zb = x_bar[0]
#     b = g - g.mean(axis=0)
#     b1 = b[:,0]
#     b2 = b[:,1]
#     b3 = b[:,2]
#     w_0 = (x[8,0]*b1 + x[8,1]*b2 + x[8,2]*b3 - np.dot(x[8],b.T) *(xb*b1+yb*b2+zb*b3)/np.dot(x_bar, x_bar.T))**2

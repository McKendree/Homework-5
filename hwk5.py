import numpy as np
import matplotlib.pyplot as plt

#base function that will be integrated
def function_to_integrate(x):
    return x**2

#exact integral of f(x) function
def analytic_integral_of_f(a=None, b=None):
    lowerLim = (1/3)*a**3
    upperLim = (1/3)*b**3
    return upperLim - lowerLim

#trapezoidal rule function where f=func, a=lower lim, b=upper lim, and n=number of datapoints
def trapezoidal_rule(f, a=None, b=None, n=None):
    #generate linear set of n points starting at a, ending at b
    points = np.linspace(a,b,n)
    h = (b-a)/(n-1)
    area = 0
    for i in range(len(points)-1):
        area += (1/2)*(f(points[i])+f(points[i+1]))*h
    return area

#lefthand riemann function where f=func, a=lower lim, b=upper lim, and n=number of datapoints
def lefthand_riemann(f, a=None, b=None, n=None):
    #generate linear set of n points starting at a, ending at b
    points = np.linspace(a,b,n)
    h = points[1]-points[0]
    integral_approx = 0
    for point in points:
        integral_approx += f(point)*h
    return integral_approx

#simpson rule function where f=func, a=lower lim, b=upper lim, and n=number of datapoints
def simpson_rule(f, a=None, b=None, n=None):
    assert n >= 3
    remainder = (n-3)%2
    #generate linear set of n points starting at a, ending at b
    points = np.linspace(a,b,n)
    area = 0
    for i in range(int((len(points)-remainder-1)/2)):
        #finds 3 base coordinates for each parabola
        coord1 = (points[i*2],f(points[i*2]))
        coord2 = (points[(i*2)+1],f(points[(i*2)+1]))
        coord3 = (points[(i*2)+2],f(points[(i*2)+2]))

        #finds values of the parts of the parabola integral function
        xNumerator = (-coord1[0]**2+coord2[0]**2+((coord1[1]-coord2[1])/
                (coord2[1]-coord3[1]))*(coord2[0]**2-coord3[0]**2))
        xDenominator = (2*(-coord1[0]+coord2[0]+((coord1[1]-coord2[1])/(coord2[1]-coord3[1]))
                *coord2[0]-((coord1[1]-coord2[1])/(coord2[1]-coord3[1]))*coord3[0]))
        x = xNumerator/xDenominator
        a_parabola = (coord1[1]-coord2[1])/((coord1[0]-x)**2-(coord2[0]-x)**2)
        y = coord1[1]-a_parabola*(coord1[0]-x)**2

        #computes parabola integral function over specific range defined by initial 3 coordinates
        lowerLim = (a_parabola/3)*points[i*2]**3+x*points[i*2]**2+(x**2+y)*points[i*2]
        upperLim = (a_parabola/3)*points[(i*2)+2]**3+x*points[(i*2)+2]**2+(x**2+y)*points[(i*2)+2]
        area += upperLim - lowerLim
        
    #if there's an extra step interval, uses trapezoidal rule to compute its integral
    if remainder == 1:
        h = (b-a)/(n-1)
        area += (1/2)*(f(points[-2])+f(points[-1]))*h
    return area

#relative error function
def relative_error(true=None, estimate=None):
    return np.abs((true-estimate)/true)

if __name__ == "__main__":
    a = 0
    b = 1
    n = 10
    print("Lefthand riemann approx to f, between",a,b," steps=",n)
    print(lefthand_riemann(function_to_integrate, a=a, b=b, n=n))

    '''creates list of relative errors for each integration method with step sizes
    of 0.1, 0.01, 0.001, 0.0001, 0.00001, and 0.000001'''
    stepSizes = [0.1,0.01,0.001,0.0001,0.00001,0.000001]
    percentError_RiemannSum = []
    percentError_TrapezoidalRule = []
    percentError_SimpsonsRule = []
    for size in stepSizes:
        percentError_RiemannSum.append(relative_error(analytic_integral_of_f(0,1),
                    lefthand_riemann(function_to_integrate,0,1,int(1/size))))
        percentError_TrapezoidalRule.append(relative_error(analytic_integral_of_f(0,1),
                    trapezoidal_rule(function_to_integrate,0,1,int(1/size))))
        percentError_SimpsonsRule.append(relative_error(analytic_integral_of_f(0,1),
                    simpson_rule(function_to_integrate,0,1,int(1/size))))

    #converts lists to arrays for plotting
    stepSizes = np.array(stepSizes)
    percentError_RiemannSum = np.array(percentError_RiemannSum)
    percentError_TrapezoidalRule = np.array(percentError_TrapezoidalRule)
    percentError_SimpsonsRule = np.array(percentError_SimpsonsRule)

    #plots the relative errors vs stepsize
    plt.plot(stepSizes,percentError_RiemannSum,color='blue', alpha=0.6)
    plt.plot(stepSizes,percentError_TrapezoidalRule,color='green', alpha=0.6)
    plt.plot(stepSizes,percentError_SimpsonsRule,color='tomato', alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Relative Error vs. Step Size for Integration Methods')
    plt.xlabel('Step Size')
    plt.ylabel('Relative Error %')
    plt.legend(['Lefthand Riemann Sum','Trapezoidal Method',"Simpson's Rule"])
    plt.savefig('IntegrationMethodErrors.png')
    plt.show()

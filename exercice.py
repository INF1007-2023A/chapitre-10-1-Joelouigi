#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import matplotlib.pyplot as plt
import scipy


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(start=-1.3, stop=2.5, num=64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([(np.sqrt(c[0]**2 + c[1]**2), np.arctan2(c[1], c[0])) for c in cartesian_coordinates])


def find_closest_index(values: np.ndarray, number: float) -> int:
    try:
        min_diff = np.absolute(values-number)
        close_index = min_diff.argmin()
        return close_index
    except:
        return ('Erreur dans le remplacement de caractère')

def plot(x):
    #Cree une fct pour pouvoir faire l'equation
    # def f(x):
    #     return ((x**2)*np.sin(1/x**2)+x)
    #Specifie l'intervalle demande
    #x = np.linspace(-1, 1, 250)
    equation = ((x**2)*np.sin(1/x**2)+x)
    plt.plot(x, equation)
    plt.show()

def estimate_pi_monte_carlo(iteration: int=5000):
    x_inside_dots = []
    x_outside_dots = []
    y_inside_dots = []
    y_outside_dots = []

    for i in range(iteration):
        x = np.random.random()
        y = np.random.random()
        if np.sqrt(x**2 + y**2) <=1:
            x_inside_dots.append(x)
            y_inside_dots.append(y)
        else:
            x_outside_dots.append(x)
            y_outside_dots.append(y)
    
    plt.scatter(x_inside_dots, y_inside_dots)
    plt.scatter(x_outside_dots, y_outside_dots)
    plt.title("Calcul de n par la methode de Monte Carlo")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    

def integral_evaluation():
    x = np.linspace(start=-4, stop=4)
    def function(x):
        return np.exp((-x)**2)
    y = [(scipy.integrate.quad(function, 0, value))[0] for value in x]
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y)
    plt.show()
    return integral

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    #plot(np.linspace(-1, 1, 250))
    estimate_pi_monte_carlo()
    integral_evaluation()
    pass

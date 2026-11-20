import numpy as np
from scipy.integrate import dblquad


def cylinder_front(radius, distance, theta):
    """
    Function to evalutate the G factor for the front of the cylinder.
    """
    # This is the saem as your analytic version Cody
    # I just thought I'd implement it for consistency
    d = distance
    r = radius

    def integrand(phi, rho):
        # This is waht we're actually integrating
        numerator = max(0, d * np.cos(theta)) * rho

        denominator = (
            d**2 + rho**2
            - 2*d*rho*np.sin(theta)*np.cos(phi)
        )**1.5

        return numerator / (4*np.pi*denominator)
    # This uses scipy's dblquad for a double integral, and we just pass in the bounds 0-r and 0-2Pi
    result, error = dblquad(
        integrand,
        0, r,
        0, 2 * np.pi
    )
    # Inspired by source - https://stackoverflow.com/q/4035369
    # Posted by Jason, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-11-16, License - CC BY-SA 2.5

    # I also learned about dblquad at https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html

    return result

def cylinder_side(radius, length, distance, theta):
    """
    Function to evaluate the Geometric factor for the side of the cylinder.
    """
    r = radius
    h = length
    d = distance

    def integrand(phi, z):

        numerator = max(
            0,
            d*abs(np.sin(theta))*np.cos(phi) - r
        )

        denominator = (
            d**2 * np.sin(theta)**2
            + r**2
            - 2*d*r*abs(np.sin(theta))*np.cos(phi)
            + (d*np.cos(theta) - z)**2
        )**1.5

        return r * numerator / (4*np.pi*denominator)

    result, error = dblquad(
        integrand,
        -h, 0,
        0, 2*np.pi
    )

    return result

def cylinder_geometric_factor(radius, length, distance, theta):
    """
    Fcuntion to evaluate the Geometric factor for the enitre cylinder.
    """
    G_front = cylinder_front(radius, distance, theta)

    G_side = cylinder_side(radius, length, distance, theta)

    return G_front + G_side

def geometric_factor_onaxis(radius, distance):
    """
    Geometric factor for a circular front face
    viewed by an on-axis point source.

    Parameters
    ----------
    radius : float
        Crystal radius in metres.

    distance : float
        Source-to-front-face distance in metres.
    """

    return 0.5 * (
        1 - distance / np.sqrt(distance**2 + radius**2)
    )

def square_front(width, distance, theta):
    """
    Function to get the G factor for a square crystal.
    """
    w = width
    d = distance

    def integrand(y, x):

        numerator = max(0, d * np.cos(theta))

        denominator = (
            (d*np.sin(theta) - x)**2
            + y**2
            + (d*np.cos(theta))**2
        )**1.5

        return numerator / (4*np.pi*denominator)

    result, error = dblquad(
        integrand,
        -w/2, w/2,
        -w/2, w/2
    )

    return result

def square_side(width, thickness, distance, theta):

    w = width
    t = thickness
    d = distance

    def integrand(y, z):

        numerator = max(
            0,
            d * abs(np.sin(theta)) - w/2
        )

        denominator = (
            (d * abs(np.sin(theta)) - w/2)**2
            + y**2
            + (d*np.cos(theta) - z)**2
        )**1.5

        return numerator / (4*np.pi*denominator)

    result, error = dblquad(
        integrand,
        -t, 0,
        -w/2, w/2
    )

    return result

def square_geometric_factor(width, thickness, distance, theta):

    G_front = square_front(width, distance, theta)

    G_side = square_side(width, thickness, distance, theta)

    return G_front + G_side

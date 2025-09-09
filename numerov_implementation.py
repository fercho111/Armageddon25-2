import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

# ================================================================
# Potential definition: V(r) = -V0 exp(-r^2/a^2) + λ / (r^2+b^2)^(p/2)
# ================================================================
def potential(r, V0=1.0, a=1.0, b=0.5, p=3.0, lam=0.5):
    """Central potential V(r)."""
    return -V0 * np.exp(-r**2 / a**2) + lam / (r**2 + b**2)**(p/2)


# ================================================================
# Numerov integration for radial Schrödinger equation
# ================================================================
def numerov(u0, u1, r, k, ell, V, m=1.0, hbar=1.0):
    """
    Numerov integration for one partial wave.
    
    Parameters:
        u0, u1 : initial values of the radial wavefunction
        r      : array of radii
        k      : wave number (sqrt(2mE)/ħ)
        ell    : angular momentum quantum number
        V      : potential function V(r)
        m, hbar: mass and Planck's constant
    
    Returns:
        u(r) array (solution of radial Schr eqn)
    """
    h = r[1] - r[0]  # step size
    u = np.zeros_like(r)
    u[0], u[1] = u0, u1

    def f(r):
        """Effective potential term f(r)."""
        return k**2 - 2*m*V(r)/hbar**2 - ell*(ell+1)/r**2

    for i in range(1, len(r) - 1):
        f_im1, f_i, f_ip1 = f(r[i-1]), f(r[i]), f(r[i+1])
        u[i+1] = ( (2*(1 - 5*h**2*f_i/12)*u[i] - (1 + h**2*f_im1/12)*u[i-1])
                   / (1 + h**2*f_ip1/12) )
    return u


# ================================================================
# Extract phase shift δ_l by matching to asymptotic form
# ================================================================
def phase_shift(u, r, k, ell):
    """
    Extract phase shift δ_l using logarithmic derivative matching.

    Formula:
    tan δ_l = [k j'_l(kR) u(R) - j_l(kR) u'(R)] /
              [k y'_l(kR) u(R) - y_l(kR) u'(R)]
    """
    R = r[-1]            # matching radius (end of integration domain)
    uR, uR_prev = u[-1], u[-2]
    h = r[1] - r[0]
    uR_prime = (uR - uR_prev) / h  # finite-diff derivative

    j_l = spherical_jn(ell, k*R)
    y_l = spherical_yn(ell, k*R)
    j_l_p = spherical_jn(ell, k*R, derivative=True)
    y_l_p = spherical_yn(ell, k*R, derivative=True)

    num = k * j_l_p * uR - j_l * uR_prime
    den = k * y_l_p * uR - y_l * uR_prime

    return np.arctan2(num, den)  # δ_l


# ================================================================
# Main driver
# ================================================================
def compute_phase_shift(E_over_V0=1.0, ell=0,
                        V0=1.0, a=1.0, b=0.5, p=3.0, lam=0.5,
                        r_max=20.0, N=2000, m=1.0, hbar=1.0):
    """
    Compute phase shift δ_l for chosen energy ratio E/V0 and partial wave l.
    """
    E = E_over_V0 * V0
    k = np.sqrt(2*m*E)/hbar

    r = np.linspace(1e-5, r_max, N)

    # Initial conditions near r=0
    u0 = r[0]**(ell+1)
    u1 = r[1]**(ell+1)

    V_func = lambda x: potential(x, V0=V0, a=a, b=b, p=p, lam=lam)
    u = numerov(u0, u1, r, k, ell, V_func, m=m, hbar=hbar)

    delta = phase_shift(u, r, k, ell)
    return delta


def total_cross_section(deltas, k):
    """
    Compute total scattering cross section from phase shifts.

    Parameters:
        deltas : list or array of phase shifts δ_l (in radians)
        k      : wave number (sqrt(2mE)/ħ)

    Returns:
        sigma_tot : total cross section
    """
    sigma = 0.0
    for ell, delta in enumerate(deltas):
        sigma += (2*ell + 1) * np.sin(delta)**2
    return (4*np.pi / k**2) * sigma

# ================================================================
# Example usage: multiple l values
# ================================================================
if __name__ == "__main__":
    # Parameters
    # resonant s wave
    V0, a, b, p, lam = 6.0, 1.0, 0.1, 2.0, 0.0
    # V0, a, b, p, lam = 0.8, 1.0, 0.1, 3, 0.2
    energies = np.linspace(0.1, 10, 200)  # E/V0 range
    # ells = [0, 1, 2, 3, 4]  # s, p,d, f, g waves
    ells = [0]
    m, hbar = 1.0, 1.0

    # ----------------------
    # Phase shifts vs E
    # ----------------------
    plt.figure(figsize=(8, 5))

    for ell in ells:
        deltas = []
        for E_over_V0 in energies:
            delta = compute_phase_shift(E_over_V0, ell, V0, a, b, p, lam, m=m, hbar=hbar)
            deltas.append(delta)
        plt.plot(energies, deltas, marker='o', label=fr"$\ell={ell}$")

    plt.xlabel(r"$E/V_0$")
    plt.ylabel(r"$\delta_\ell$ [rad]")
    plt.title("Phase shifts for different partial waves")
    plt.legend()
    plt.grid(True)
    plt.savefig("phase_shifts.png")

    # ----------------------
    # Total cross section vs E
    # ----------------------
    sigma_vals = []
    for E_over_V0 in energies:
        E = E_over_V0 * V0
        k = np.sqrt(2*m*E)/hbar
        deltas = [compute_phase_shift(E_over_V0, ell, V0, a, b, p, lam, m=m, hbar=hbar) for ell in ells]
        sigma_tot = total_cross_section(deltas, k)
        sigma_vals.append(sigma_tot)

    plt.figure(figsize=(8, 5))
    plt.plot(energies, sigma_vals, marker='s', color="darkred")
    plt.xlabel(r"$E/V_0$")
    plt.ylabel(r"$\sigma_{\text{tot}}$")
    plt.title("Total cross section vs energy")
    plt.grid(True)
    plt.savefig("cross_section.png")

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn


def potential(r, V0=1.0, a=1.0, b=0.5, p=3.0, lam=0.5):
    return -V0 * np.exp(-r**2 / a**2) + lam / (r**2 + b**2)**(p/2)


def numerov(u0, u1, r, k, ell, V, m=1.0, hbar=1.0):
    h = r[1] - r[0]  # step size
    u = np.zeros_like(r)
    u[0], u[1] = u0, u1

    def f(r):
        return k**2 - 2*m*V(r)/hbar**2 - ell*(ell+1)/r**2

    for i in range(1, len(r) - 1):
        f_im1, f_i, f_ip1 = f(r[i-1]), f(r[i]), f(r[i+1])
        u[i+1] = ( (2*(1 - 5*h**2*f_i/12)*u[i] - (1 + h**2*f_im1/12)*u[i-1])
                   / (1 + h**2*f_ip1/12) )
    return u


def phase_shift(u, r, k, ell):
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


def better_deltas(deltas, flip=False):
    deltas = np.unwrap(deltas)
    if flip:
        deltas = np.negative(deltas)
    shift = np.round(deltas[0] / np.pi) * np.pi
    deltas = deltas - shift

    return deltas

if __name__ == "__main__":
    # Mass and hbar
    m, hbar = 1.0, 1.0

    # Parameter sets: (label, V0, a, b, p, lam)
    param_sets = [
        ("A (Born-friendly)", 0.8, 1.0, 0.1, 3.0, 0.2),
        ("B (resonant s-wave)", 6.0, 1.0, 0.1, 2.0, 0.0),
        ("C (mixed tail)", 3.0, 1.0, 0.5, 1.5, -1.5),
    ]

    # E/V0 goes from 0.1 to 10
    E_over_V0 = np.linspace(0.1, 10.0, 50)
    ells = [0]  # only s-wave in this comparison

    # Now: rows = param sets, cols = graph types
    fig, axes = plt.subplots(len(param_sets), 3, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for row, (title, V0, a, b, p, lam) in enumerate(param_sets):
        # Compute phase shifts, sin^2, and cross sections
        deltas, sin2_vals, sigma_vals = [], [], []
        for ratio in E_over_V0:
            E = ratio * V0
            k = np.sqrt(2*m*E)/hbar
            delta = compute_phase_shift(ratio, 0, V0, a, b, p, lam, m=m, hbar=hbar)
            deltas.append(delta)
            sin2_vals.append(np.sin(delta)**2)
            sigma_vals.append(total_cross_section([delta], k))

        # Beautify deltas
        deltas = better_deltas(deltas, flip=True)

        # --- Col 0: Phase shifts
        ax1 = axes[row, 0]
        ax1.plot(E_over_V0, deltas, color="c", marker=".")
        if row == 0:
            ax1.set_title("Desplazamiento de fase s-wave")
        ax1.set_xlabel(r"$E/V_0$")
        ax1.set_ylabel(fr"{title}\n$\delta_0$ (rad)")
        ax1.grid(True)

        # --- Col 1: Total cross section
        ax2 = axes[row, 1]
        ax2.plot(E_over_V0, sigma_vals, color="y", marker=".")
        if row == 0:
            ax2.set_title("Sección eficaz total")
        ax2.set_xlabel(r"$E/V_0$")
        ax2.set_ylabel(r"$\sigma_{\text{tot}}$")
        ax2.grid(True)

        # --- Col 2: Resonance sin^2
        ax3 = axes[row, 2]
        ax3.plot(E_over_V0, sin2_vals, color="m", marker=".")
        if row == 0:
            ax3.set_title("Resonancias s-wave")
        ax3.set_xlabel(r"$E/V_0$")
        ax3.set_ylabel(r"$\sin^2(\delta_0)$")
        ax3.set_ylim(-0.1, 1.1)  # full resonance range
        ax3.grid(True)

    plt.tight_layout()
    plt.savefig("scattering_comparison.png", dpi=300)

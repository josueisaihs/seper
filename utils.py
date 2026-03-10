import numpy as np
import matplotlib.pyplot as plt



def get_complex_current(U_g, P_g, Q_g) -> complex:
    """Calcula la corriente compleja conjugada a partir de S y U."""
    S_g = P_g + 1j * Q_g
    return np.conj(S_g / U_g)


def get_current_magnitude(U_g, P_g, Q_g) -> float:
    """Magnitud de la corriente en por unidad."""
    return np.abs(get_complex_current(U_g, P_g, Q_g))


def get_current_angle(U_g, P_g, Q_g) -> float:
    """Ángulo de fase de la corriente en radianes."""
    return np.angle(P_g + 1j * Q_g) - np.angle(U_g)


def get_power_factor(U_g, P_g, Q_g) -> float:
    """Factor de potencia (coseno del ángulo de fase)."""
    return np.cos(get_current_angle(U_g, P_g, Q_g))


def get_E_delta(U_g, X_t, I_m, phi):
    a = X_t * I_m * np.cos(phi)
    b = U_g + X_t * I_m * np.sin(phi)
    delta = np.arctan2(a, b)
    E = np.sqrt(a**2 + b**2)
    return E, delta


def plot_phasor_diagram(U_g, I_m, phi, E, delta, X_t, scale_I=None, title=''):
    """Dibuja el diagrama fasorial: U_g, I, E y jX_t·I."""
    fig, ax = plt.subplots(figsize=(8, 8))

    if scale_I is None:
        scale_I = U_g / I_m * 0.5

    quiver_kw = dict(angles='xy', scale_units='xy', scale=1, width=0.008)

    def draw_phasor(ox, oy, dx, dy, color, label, ls=None):
        ax.quiver(ox, oy, dx, dy, color=color, label=label, **quiver_kw)
        if ls == '--':
            ax.plot([ox, ox + dx], [oy, oy + dy], color=color, ls='--', lw=1.5)
        ax.text(ox + dx * 1.05, oy + dy * 1.05, label, color=color,
                fontsize=13, ha='left', va='bottom')

    # U_g en eje real (referencia)
    draw_phasor(0, 0, U_g, 0, 'blue', r'$U_g$')

    # Corriente I con ángulo -phi (atraso)
    I_scaled = I_m * scale_I
    draw_phasor(0, 0, I_scaled * np.cos(-phi), I_scaled * np.sin(-phi),
                'red', r'$I$')

    # E con ángulo delta
    Ex = E * np.cos(delta)
    Ey = E * np.sin(delta)
    draw_phasor(0, 0, Ex, Ey, 'green', r'$E$')

    # Caída jX_t·I (de U_g a E)
    draw_phasor(U_g, 0, Ex - U_g, Ey, 'orange', r'$jX_t I$', ls='--')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    margin = max(E, U_g, I_scaled) * 0.15
    ax.set_xlim(-margin, max(E, U_g) + margin)
    ax.set_ylim(-(I_scaled + margin), max(Ey, 0) + margin)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginario')
    ax.set_title(f'Diagrama Fasorial {title}', pad=10)
    ax.legend(loc='best')
    return fig, ax

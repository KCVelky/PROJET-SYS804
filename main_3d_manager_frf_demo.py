from __future__ import annotations

import matplotlib.pyplot as plt

from core import SimulationManager3D


def main() -> None:
    manager = SimulationManager3D(load_defaults=True)

    result = manager.run_frf_direct(
        sensor_name="S1",
        use_black_hole=True,
        damping_ratio=0.01,
        damping_freq1_hz=140.0,
        damping_freq2_hz=500.0,
        rebuild=False,
    )

    frf = result["frf_result"]
    print("=== Manager FRF directe 3D ===")
    print("Noeud excitation :", frf.excitation_node_id)
    print("Noeud capteur    :", frf.sensor_node_id)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(frf.frequencies_hz, frf.magnitude)
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("|H(ω)|")
    ax.set_title("FRF directe 3D via SimulationManager3D")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

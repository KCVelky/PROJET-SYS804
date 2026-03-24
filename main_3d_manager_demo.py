from __future__ import annotations

from core.simulation_manager_3d import SimulationManager3D
from solid3d import ModalComparison3D


def main() -> None:
    manager = SimulationManager3D(load_defaults=True)

    print("=== Résumé du cas 3D ===")
    print(manager.get_case_summary())

    modal = manager.run_modal_analysis(n_modes=6, use_black_hole=True)
    print("\n=== Fréquences propres 3D avec ABH ===")
    for i, f in enumerate(modal["frequencies_hz"], start=1):
        print(f"Mode {i:02d} : {f:.3f} Hz")

    validation = manager.validate_modal_analysis(n_modes=6, use_black_hole=True)
    print("\nValidation OK ?", validation["is_valid"])

    comparison = manager.compare_modal(n_modes=6)
    ModalComparison3D.print_summary(comparison["comparison"])


if __name__ == "__main__":
    main()

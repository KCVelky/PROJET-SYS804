from core import SimulationManager

manager = SimulationManager(load_defaults=True)

modal = manager.run_modal_analysis(n_modes=6, use_black_hole=True)
print(modal["frequencies_hz"])

frf = manager.run_frf_modal(
    sensor_name="S1",
    use_black_hole=True,
    n_modes=30,
    damping_ratio=0.01,
)

result = frf["frf_result"]
print(result.magnitude[:10])
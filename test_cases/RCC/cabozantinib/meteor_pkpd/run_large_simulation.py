import io
import contextlib
import os

import cabozantinib_pkpd_simulation as m

def main() -> None:
    n = 300
    n_jobs = max(1, (os.cpu_count() or 1) - 1)
    res = m.run_full_simulation(
        n_patients=n,
        starting_doses=[20, 40, 60],
        n_jobs=n_jobs,
        validate_pk_first=True,
        pk_validation_n=300,
    )

    m.plot_results(res, save_path="cabozantinib_pkpd_results.png")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.print_summary_table(res)

    with open("simulation_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Simulation N per group: {n}\n\n")
        f.write(buf.getvalue())

    print("Done")


if __name__ == "__main__":
    main()

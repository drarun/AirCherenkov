"""Compatibility entry point for the renamed gamma camera pipeline.

Prefer ``python run_gamma_camera_pipeline.py`` for new usage. This command is
retained so existing scripts continue to work.
"""


def _run_gamma_camera_pipeline():
    from run_gamma_camera_pipeline import main

    return main()


def main():
    print(
        "[deprecated name] run_full_sim.py runs only the gamma camera pipeline; "
        "use run_gamma_camera_pipeline.py instead."
    )
    return _run_gamma_camera_pipeline()


if __name__ == "__main__":
    main()

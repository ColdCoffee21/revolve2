import argparse


class Config():

    def _get_params(self):
        parser = argparse.ArgumentParser()

        # EA params

        parser.add_argument(
            "--population_size",
            required=False,
            default=100,
            type=int,
        )

        parser.add_argument(
            "--offspring_size",
            required=False,
            default=100,
            type=int,
        )

        parser.add_argument(
            "--num_generations",
            required=False,
            default=200,
            type=int,
        )

        parser.add_argument(
            "--max_modules",
            required=False,
            default=30,
            type=int,
            help="",
        )

        parser.add_argument(
            "--substrate_radius",
            required=False,
            default=15,
            type=int,
            help="",
        )

        parser.add_argument(
            "--plastic_body",
            required=False,
            default=0,
            type=int,
            help="0 is not plastic, 1 is plastic",
        )

        parser.add_argument(
            "--plastic_brain",
            required=False,
            default=0,
            type=int,
            help="0 is not plastic, 1 is plastic",
        )

        parser.add_argument(
            "--body_substrate_dimensions",
            required=False,
            default='2d',
            type=str,
            help="2d or 3d",
        )

        parser.add_argument(
            "--num_initial_mutations",
            required=False,
            default=10,
            type=int,
        )  # number of initial mutations for body and brain CPPNWIN networks

        parser.add_argument(
            "--crossover_prob",
            required=False,
            default=0,
            type=float,
        )

        parser.add_argument(
            "--mutation_prob",
            required=False,
            default=1,
            type=float,
        )

        parser.add_argument(
            "--fitness_measure",
            required=False,
            default="speed_y",
            type=str,
        )

        # Simulation and experiment params

        parser.add_argument(
            "--study_name",
            required=False,
            default="default_study",
            type=str,
            help="",
        )

        parser.add_argument(
            "--experiment_name",
            required=False,
            default="defaultexperiment",
            type=str,
            help="Name of the experiment.",
        )

        parser.add_argument(
            "--run",
            required=False,
            default=1,
            type=int,
            help="",
        )

        parser.add_argument(
            "--simulation_time",
            required=False,
            default=30,
            type=int,
        )

        parser.add_argument(
            "--simulator",
            required=False,
            default="isaac",
            type=str,
            help="isaac or mujoco",
        )

        parser.add_argument(
            "--mainpath",
            required=False,
            default="/home",
            type=str,
            help="path results files"
        )

        parser.add_argument(
            "--sampling_frequency",
            required=False,
            default=5,
            type=int,
        )  # number of samples per sec from batch (snapshots of sim)

        parser.add_argument(
            "--run_simulation",
            required=False,
            default=1,
            type=int,
            help="If 0, runs optimizer without simulating robots, so behavioral measures are none."
        )

        parser.add_argument(
            "--control_frequency",
            required=False,
            default=20,
            type=int,
        )

        # provides params that define environmental conditions and/or task
        parser.add_argument(
            "--seasons_conditions",
            required=False,
            default='1.0_1.0_0_0_0',
            type=str,
            # seasons separated by '#' and their params separated by '_': params order matters!
            # order: staticfriction, dynamicfriction, yrotationdegrees, platform, direction
            help="param1_param2...#"
                 "param1_param2...#...",
        )
        args = parser.parse_args()

        return args


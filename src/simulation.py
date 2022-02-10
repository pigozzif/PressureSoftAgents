import argparse

from src.frameworks import RenderFramework, NoRenderFramework
from src.utils import set_seed


def simulation(args, render):
    if render:
        framework = RenderFramework(args.body, args.task)
    else:
        framework = NoRenderFramework(args.body, args.task)
    while framework.get_step_count() < args.timesteps:
        framework.step()
    framework.reset()
    return framework.get_reward() / (args.timestep / 60.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument("--body", type=str, default="pressure", help="kind of soft body to simulate")
    parser.add_argument("--task", type=str, default="obstacles", help="task to simulate")
    parser.add_argument("--timesteps", type=int, default=900, help="number of time steps to simulate")
    parser.add_argument("--render", type=bool, default=True, help="render simulation on screen")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()
    set_seed(args.seed)
    simulation(args, args.render)

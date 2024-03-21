"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Pong Game Playback
"""

import datetime
import glob
import os
import pickle
import shutil
from pathlib import Path

import imageio
import jsonargparse
import pandas as pd
import pygame
from pygifsicle import optimize
from tqdm import tqdm

from corl.simulators.pong.pong import GameStatus, PongRender


def create_gif(image_directory: str, output_filename: str, fps: int):
    """Creates a gif of all images in the `image_directory` and saves it
    out to `output_filename`.

    Parameters
    ----------
    image_directory : str
        The directory containing the individual image frames.
    output_filename : str
        Output filename.
    fps : int
        Frames per second.
    """

    writer: imageio.plugins.pillowmulti.GIFFormat.Writer

    image_files = glob.glob(f"{image_directory}/*")
    image_files.sort()
    print("Creating a gif...")
    # Reads one frame in at a time while creating the gif
    with imageio.v2.get_writer(output_filename, duration=1000 / fps) as writer:
        for file in tqdm(image_files):
            image = imageio.v2.imread(file)
            writer.append_data(image)

    # Attempts lossless compression
    optimize(output_filename)


def main(pickle_files: list, parse_args: jsonargparse.Namespace):  # noqa: PLR0915
    """Episode playback. Renders every episode in the list of pickle files
    sequentially. Displays the simulation time, accumulated awards and episode
    file as text.

    Parameters
    ----------
    pickle_files : typing.List
        A list of files to iterate through.
    """

    pygame.init()

    clock = pygame.time.Clock()
    fps = 60
    left_score = 0
    right_score = 0
    pong_render = None
    tmp_directory = "/tmp/pong_frames/"  # noqa: S108
    frame = 0

    # Make sure this directory is empty
    os.makedirs(tmp_directory, exist_ok=False)

    for file in tqdm(pickle_files):
        with open(file, "rb") as file:  # noqa: PLW2901
            episode_states = pickle.load(file)  # noqa: S301

        pong = episode_states[0]["pong_game"]
        pong_render = PongRender(pong.screen_width, pong.screen_height)
        file_to_display = str(file.name).replace(parse_args.input, "...") if os.path.isdir(parse_args.input) else parse_args.input
        text_block = []

        episode_length = len(episode_states)
        # Iterates through an episode
        for state in episode_states:
            clock.tick(fps)

            pong = state["pong_game"]
            # Construct extra text information to show
            text_block.append(file_to_display)
            for key, value in state["rewards_accumulator"].items():
                text_block.append(f"{key}:{value}")
            sim_time = state["sim_time"]
            text_block.append(f"sim time: {sim_time}")

            # Renders each frame
            pong_render.draw(pong, left_score, right_score, text_block)
            game_status = state["game_status"]

            if game_status is GameStatus.RIGHT_WIN:
                right_score += 1
            elif game_status is GameStatus.LEFT_WIN:
                left_score += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            # zero pad the frame count
            frame_count = str(frame).zfill(len(str(episode_length) * len(pickle_files)))
            pygame.image.save(pong_render.display_window, f"{tmp_directory}/{frame_count}.jpg")
            text_block = []
            frame += 1

        # Creates the gif for the episode
        if not parse_args.one_gif_all_episodes:
            output_filename = Path(file.name).name.replace(".pkl", ".gif")
            create_gif(image_directory=tmp_directory, output_filename=f"{parse_args.gif_output_directory}/{output_filename}", fps=fps)
            frame = 0
            # Removes the directory and recreates it
            shutil.rmtree(tmp_directory)
            os.makedirs(tmp_directory, exist_ok=False)

        # Updates the final score
        pong_render.draw(pong, left_score, right_score, text_block)

    # Creates one gif for all the episodes
    if parse_args.one_gif_all_episodes:
        filename_start = Path(pickle_files[0].name).name.replace(".pkl", "")
        filename_end = Path(pickle_files[-1].name).name.replace(".pkl", "")
        output_filename = f"{filename_start}_{filename_end}-{datetime.datetime.now().isoformat()}.gif"

        create_gif(image_directory=tmp_directory, output_filename=f"{parse_args.gif_output_directory}/{output_filename}", fps=fps)
        # Remove the temporary directory
        shutil.rmtree(tmp_directory)

    if pong_render:
        pong_render.draw_win("Finished playback.")

    pygame.quit()
    shutil.rmtree(tmp_directory)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--input", help="Top level directory to where the episode states are saved or a single episode.")
    parser.add_argument("--later-episodes-first", help="Plays episodes in reverse order, with newer ones first. ", action="store_true")
    parser.add_argument("--one-gif-all-episodes", action="store_true", help="Creates a single GIF for all episodes in the input.")
    parser.add_argument("--gif-output-directory", default="gifs/")
    args = parser.parse_args()
    if os.path.isdir(args.input):
        episode_files = list(Path(args.input).rglob("episode*.pkl"))
        file_df = pd.DataFrame(episode_files, columns=["full_filepath"])

        file_df["filename"] = file_df["full_filepath"].apply(lambda x: x.name)
        file_df = file_df.sort_values(by=["filename"], ascending=not args.later_episodes_first)
        episode_files = file_df["full_filepath"].tolist()
    else:
        episode_files = [Path(args.input)]

    print(f"{len(episode_files)} episodes found.")
    os.makedirs(args.gif_output_directory, exist_ok=True)
    main(pickle_files=episode_files, parse_args=args)

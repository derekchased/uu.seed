import argparse
from dataaugmentations.YMLLoader import YMLLoader
from ast import literal_eval
import numpy as np
import glob
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import pickle

class AnalysisShell:
    """
    1. get arguments
    2. load config
    3. inject
    """

    def __init__(self):

        LOAD_PICKLE = True
        SAVE_PICKLE = not LOAD_PICKLE
        SHOW_TIME = True

        TITLE = "Gen Test 10 Exp3: Some obstacles & Garage back and rotated"

        ID = "kiwi_1205_gen10_exp3"
        PICKLE_NAME = ID+".pickle"
        BASE_PATH = "../unity-testing-demo-env/Assets/Resources/DaggerDemonstrations"
        FOLDER = BASE_PATH + "/" + ID + "/"
        FIG = FOLDER + ID + "_success.png"
        FIG_TIME = FOLDER + ID + "_wtime_success.png"
        BASELINE = "base"


        if LOAD_PICKLE:
            experiments = pickle.load(open(PICKLE_NAME, 'rb'))
        else:

            experiments = []

            g = glob.glob(FOLDER + "*.txt")
            # g.extend(glob.glob('../unity-testing-demo-env/Assets/Resources/DaggerDemonstrations/exp1_train/221129*.txt'))

            for index, path in enumerate(g):
                episodes = {"model":path.split("/")[-1],"success": [], "length":[], "x": [], "y": [], "z": []}

                experiments.append(episodes)

                with open(path) as f:
                    for line in f:
                        lit = list(literal_eval(line.strip()))
                        episodes["success"].append(lit.pop(0))
                        x = [tup[0] for tup in lit]
                        episodes["x"].append(x)
                        episodes["y"].append([tup[1] for tup in lit])
                        episodes["z"].append([tup[2] for tup in lit])
                        episodes["length"].append(len(x))

                len_np = np.asarray(episodes["length"])
                successful = len_np[episodes["success"]]

                # episodes["avg"] = np.mean(episodes["length"])
                episodes["avg_success"] = np.mean( successful)
                # episodes["std"] = np.std(len_np, dtype=np.float64)
                episodes["std_success"] = np.std(successful, dtype=np.float64)
                episodes["success_rate"] = len(successful)/len(len_np)

                # for key, val in episodes.items():
                #     if(key != "x" and key != "y" and key != "z" and key != "success" and key != "length"):
                #         print(f"{key} : {val}")
                # print("\n")

        if SAVE_PICKLE:
            pickle.dump(experiments, open(PICKLE_NAME, 'wb'))

        dff = [{"name": " ".join(experiment["model"].split("_")[2:-3]),"success rate":experiment["success_rate"], "avg success time": experiment["avg_success"]}   for experiment in experiments]

        # print(dff)
        df = pd.DataFrame(dff)

        # print(df.columns)
        df.set_index(["name"], inplace=True)
        index = df.index

        # print(index)

        sns.set_theme(style="white", context="talk")
        ax = sns.barplot(data=df, x=df.index, y="success rate", palette="rocket")
        ax.title.set_text(TITLE)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-85)
        plt.ylim(df["success rate"].min()-.03, df["success rate"].max()+.1)







        ax.axhline(y=df.loc[BASELINE,"success rate"], color='green', linewidth=1, linestyle="--" )
        plt.tight_layout(h_pad=.5)
        fig = ax.get_figure()
        fig.savefig(FIG)
        if not SHOW_TIME:
            plt.show()

        times = df.loc[:, "avg success" \
                          " time"].tolist()
        formatted_times = ["" if math.isnan(val) else str(math.ceil(val)) for val in times]
        ax.bar_label(ax.containers[0], formatted_times, fontsize=8)
        fig = ax.get_figure()
        fig.savefig(FIG_TIME)

        if SHOW_TIME:
            plt.show()

        plt.show()






        #
        # rs = np.random.RandomState(8)
        #
        # # Set up the matplotlib figure
        # # f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
        # f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        #
        # # Generate some sequential data
        # # x = np.array(list("ABCDEFGHIJ"))
        # # y1 = np.arange(1, 11)
        # # sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
        # sns.barplot(data=df, x="name", y="success rate", palette="rocket", ax=ax1)
        # ax1.axhline(0, color="k", clip_on=False)
        # ax1.set_ylabel("Sequential")
        #
        # # Center the data to make it diverging
        # # y2 = y1 - 5.5
        # # sns.barplot(x=x, y=y2, palette="vlag", ax=ax2)
        # sns.barplot(data=df, x="name", y="avg success time", palette="rocket", ax=ax2)
        # ax2.axhline(0, color="k", clip_on=False)
        # ax2.set_ylabel("Diverging")
        #
        # # Randomly reorder the data to make it qualitative
        # # y3 = rs.choice(y1, len(y1), replace=False)
        # # sns.barplot(x=x, y=y3, palette="deep", ax=ax3)
        # # ax3.axhline(0, color="k", clip_on=False)
        # # ax3.set_ylabel("Qualitative")
        #
        # # Finalize the plot
        # sns.despine(bottom=True)
        # # plt.setp(f.axes, yticks=[])
        # plt.tight_layout(h_pad=2)
        #
        # ####################################################################
        #
        # # df = pd.DataFrame(experiments)
        # # print(df["length"])
        #
        # ####################################################################
        #
        # # sns.set_theme(style="ticks")
        # #
        # # # Initialize the figure with a logarithmic x axis
        # # f, ax = plt.subplots(figsize=(7, 6))
        # # ax.set_xscale("log")
        # #
        # # # Load the example planets dataset
        # # # planets = sns.load_dataset("planets")
        # # # print(planets)
        # # # Plot the orbital period with horizontal boxes
        # # sns.boxplot(x="frames", y="model", data=df["length"], whis=[0, 2000], width=.6, palette="vlag")
        # #
        # # # Add in points to show each observation
        # # # sns.stripplot(x="distance", y="method", data=planets,
        # # #               size=4, color=".3", linewidth=0)
        # #
        # # # Tweak the visual presentation
        # # ax.xaxis.grid(True)
        # # ax.set(ylabel="")
        # # sns.despine(trim=True, left=True)
        #
        # plt.show()

if __name__ == "__main__":
    shell = AnalysisShell()


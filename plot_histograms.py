import json

import fire

import plotly.graph_objects as go

import numpy as np


def main(filename_1: str = "data/outputs/bert_big_00010000.json", filename_2: str = None, label_1: str = None, label_2: str = None):
    """
    Plots comparative histograms between multiple sets of simulated scores
    """

    # Load and add the first histogram
    if not label_1:
        label_1 = filename_1

    # Load the data
    with open(filename_1, 'r') as f:
        data_1 = json.load(f)

    output_filename_part_1 = filename_1.split(".")[0].split("/")[-1]
    output_filename = f"data/images/{output_filename_part_1}"

    # Plot the histograms
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data_1, name=label_1, histnorm='percent', texttemplate="%{y:.3}%"))



    fig.update_layout(
        xaxis_title_text='Score',
        yaxis_title_text='Frequency (%)',
        bargap=0.1
    )

    if filename_2:
        if not label_2:
            label_2 = filename_2

        with open(filename_2, 'r') as f:
            data_2 = json.load(f)

        output_filename_part_2 = filename_2.split(".")[0].split("/")[-1]
        output_filename += f"_{output_filename_part_2}"

        fig.add_trace(go.Histogram(x=data_2, name=label_2, histnorm='percent', texttemplate="%{y:.3}%"))

    fig.show()

    output_filename += ".png"
    fig.write_image(output_filename)


if __name__ == "__main__":
    fire.Fire(main)
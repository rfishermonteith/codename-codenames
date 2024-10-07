# codename-codenames

This is an exploratory repo to try play optimal Codenames. I've written about this here.

Most of the functions can be called with optional parameters, which can be used to specify the model (bert or hash), 
size (small or big) or file format (json or pkl). You can see the help by running `python <function_name>.py --help`.

To produce the final visualisations, or to simulate a game, you'll need to run the following (you can switch the
parameters out for whichever you'd like):

```commandline
python create_embeddings.py bert big
```

```commandline
python calculate_distances.py bert big json
```

Then, either play a game with:

```commandline
python play_game.py bert big json
```

Or simulate many games:
```commandline
python play_n_games.py bert big json 10
```
This will create a file in data/outputs/

You can create a histogram of this using:
```commandline
python plot_histograms.py "data/outputs/bert_big_00000010.json"
```
This will create a plot in data/images/
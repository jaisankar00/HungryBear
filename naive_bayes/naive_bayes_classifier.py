import pandas as pd
file = pd.read_csv("../training_data/term_frequency_matrix.csv")
CUISINES = ["Indian","Mexican","Chinese","American","Mediterranean","Japanese","Thai","Italian"]
DISHES = list(file['Dish'])

def naive_helper():
    cuisine_total = {}
    for cuisine in CUISINES:
        total = file.sum()
        cuisine_total[cuisine] = total[cuisine]
    return cuisine_total

CUISINE_TOTAL = naive_helper()
print(CUISINE_TOTAL)

""" Takes DISH and returns a string - the classified cuisine."""
def naive_bayes(dish):

    words = dish.split()

    highest_cuisine = ""
    highest_prob = 0
    for cuisine in CUISINES:
        prior_cuisine_probability = CUISINE_TOTAL[cuisine]/sum(CUISINE_TOTAL.values())
        final_prob = prior_cuisine_probability
        for word in words:
            found = file.loc[word]
            word_prob = (found[cuisine] + 1)/ (file["Dish"] + sum(file[cuisine].values()))
            final_prob *= word_prob
        if (final_prob > highest_prob):
            highest_prob = final_prob
            highest_cuisine = cuisine

    return highest_cuisine

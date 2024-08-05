import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.io import show


# many custom function to plot things

def get_ratio(db):
    ratio = sum(db['is_fraud']) / len(db.index)
    print(100 * ratio)


def plot_lat_lon_category(longitude, latitude, category):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a color map
    cmap = plt.cm.get_cmap('YlOrBr')

    colors = ['blue', 'red']
    color_values = [colors[cat] for cat in category]
    # Plot the longitude and latitude values as a scatterplot, coloring the dots according to the category
    ax.scatter(longitude, latitude, c=color_values, alpha=0.3)

    # Add axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Show the plot
    plt.show()


def plot_pca_scatter(x, y):
    # Create a PCA object and fit it to the data
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)

    # Set the colors for each point based on the category
    categories = list(set(y))
    colors = ['blue', 'red']
    color_values = [colors[cat] for cat in y]

    # Create a ColumnDataSource with the PCA data and colors
    source = ColumnDataSource(data=dict(x=x_pca[:, 0], y=x_pca[:, 1], color=color_values))

    # Create the scatter plot
    p = figure(title='PCA Scatter Plot')
    p.circle(x='x', y='y', color='color', source=source, size=10)

    show(p)


def plot_histogram(data, x, y, title, width=100):
    # Group the data by the category
    groups = data.groupby(x)

    # Get the counts for each class in each category
    counts = groups[y].value_counts().unstack()

    # Calculate the ratio of class 0 to class 1 in each category
    for index, num in enumerate(counts[0]):
        if np.isnan(counts[0][index]):
            counts[0][index] = 1
            print(counts[1][index])
            counts[1][index] = 0
        if np.isnan(counts[1][index]):
            counts[1][index] = 0

    ratio = counts[1] / counts[0]
    counts['ratio'] = ratio

    # Sort the categories by the ratio
    #ratio = ratio.sort_values()

    # Set the figure width
    plt.figure(figsize=(width, 10))

    # Plot the histogram
    plt.bar(ratio.index, ratio, width=0.8)
    plt.title(title, fontsize=30)
    plt.xlabel(x, fontsize=30)
    plt.xticks(rotation=0, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Ratio of class 0 to class 1', fontsize=20)

    # # Add the ratio values inside the bars
    # for i, val in enumerate(ratio):
    #     plt.text(i, val, f'{val:.4f}', ha='center', rotation=90)
    plt.show()


def dist_same_interval(doc, to_check, threshold=[0, 30000], bins_num=[10]):
    # check input
    if len(threshold) - 1 != len(bins_num):
        print("bins and interval don't match")
        return

    original = doc
    for i in range(0, len(threshold) - 1):
        # create bins
        bins = np.linspace(threshold[i], threshold[i + 1], bins_num[i])

        # round, and get only values between thresholds
        doc = doc[doc[to_check] > threshold[i]]
        doc = doc[doc[to_check] < threshold[i + 1]]

        # split data
        fraud_train = doc[doc['is_fraud'] == 1]

        # create histograms
        title = "Density between " + str(threshold[i]) + " to " + \
                str(threshold[i + 1])

        ns, bins, patches = plt.hist(x=[fraud_train[to_check], doc[to_check]],
                                     bins=bins, color=['r', 'g'],
                                     label=['fraud', 'all'])
        del patches

        plt.close()

        for i, item in enumerate(ns[1]):
            if ns[1][i] == 0:
                ns[1][i] = 1

        ratios = ns[0] / ns[1]
        plt.bar(x=bins, height=ratios, width=bins[1] - bins[0])
        plt.gca().set(title=title, ylabel='Density')
        plt.show()
        doc = original


def plot_countries(df):
    # Group the data by country
    grouped = df.groupby('state')

    # Iterate over the groups
    for name, group in grouped:
        # Get the longitude and latitude for each group
        lon = group['long'] + np.random.normal(0, 0.1, size=len(group['long']))
        lat = group['lat'] + np.random.normal(0, 0.1, size=len(group['lat']))

        # Get the binary variable for each group
        binary = group['is_fraud']

        colors = ['blue', 'red']
        color_values = [colors[cat] for cat in binary]

        # Create a scatter plot
        plt.scatter(lon, lat, c=color_values, alpha=0.5)

        # Set the title to the name of the country
        plt.title(name)

        # Show the plot
        plt.show()


def plot_histograms_with_intervals(doc, to_check, target, intervals, bins):
    for i, interval in enumerate(intervals):
        # Get the data for the current interval
        data = doc[(doc[to_check] >= interval[0]) & (doc[to_check] < interval[1])]

        # Make the histogram
        plt.hist(data[to_check], bins=bins[i])
        plt.close()

        # Calculate the ratio of 1s to all samples for each bin
        for j in range(bins[i]):
            bin_data = data[(data[to_check] >= interval[0] + j * (interval[1] - interval[0]) / bins[i]) &
                            (data[to_check] < interval[0] + (j + 1) * (interval[1] - interval[0]) / bins[i])]
            counts = bin_data[target].value_counts()
            total = len(bin_data)
            if 1 in counts:
                ratio = counts[1] / total
            else:
                ratio = 0
            width = (interval[1] - interval[0]) / bins[i]
            # Add a bar plot to show the ratio of 1s to all samples
            plt.bar(bin_data[to_check].mean(), ratio, width=width, color='blue')

        # Show the plot
        plt.show()

import matplotlib.pyplot as plt

def visualize_word2vec(w2v):
    plt.scatter(result[:, 0], result[:, 1])
    words = list(w2v.wv.vocab)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()

def draw_word_cloud(wordcloud):
    fig = plt.figure(1, figsize=(10, 5))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

def plot_confusion_matrix_graph(confMatrix):
    # print the confusion matrix & plot graph
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confMatrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(confMatrix.shape[0]):
        for j in range(confMatrix.shape[1]):
            ax.text(x=j, y=i, s=confMatrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

def visualize_corex(model):
    plt.figure(figsize=(10,5))
    plt.bar(range(model.tcs.shape[0]), model.tcs, color='#4e79a7', width=0.5)
    plt.xlabel('Topic', fontsize=16)
    plt.ylabel('Total Correlation (nats)', fontsize=16);

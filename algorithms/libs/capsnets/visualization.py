import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets, interactive


class AffineVisualizer:
    # only MNIST
    def __init__(self, model, x, y, hist=True):
        self.min_value = - 0.30
        self.max_value = + 0.30
        self.step = 0.05
        self.sliders = {str(i): widgets.FloatSlider(min=self.min_value, max=self.max_value, step=self.step) for i in
                        range(16)}
        self.text = widgets.IntText()
        self.sliders['index'] = self.text
        self.model = model
        self.X = x
        self.y = y
        self.hist = hist

    def affine_transform(self, **info):

        index = abs(int(info['index']))
        tmp = np.zeros([1, 10, 16])

        for d in range(16):
            tmp[:, :, d] = info[str(d)]

        y_pred, X_gen = self.model.predict([self.X[index:index + 1], self.y[index:index + 1], tmp])

        if self.hist:
            fig, ax = plt.subplots(1, 3, figsize=(15, 3))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(12, 12))
        ax[0].imshow(self.X[index, ..., 0], cmap='gray')
        ax[0].set_title('Input Digit')
        ax[1].imshow(X_gen[0, ..., 0], cmap='gray')
        ax[1].set_title('Output Generator')
        if self.hist:
            ax[2].set_title('Output Caps Length')
            ax[2].bar(range(10), y_pred[0])
        plt.show()

    def on_button_clicked(self, k):
        for i in range(16):
            self.sliders[str(i)].value = 0

    def start(self):
        button = widgets.Button(description="Reset")
        button.on_click(self.on_button_clicked)

        main = widgets.HBox([self.text, button])
        u1 = widgets.HBox([self.sliders[str(i)] for i in range(0, 4)])
        u2 = widgets.HBox([self.sliders[str(i)] for i in range(4, 8)])
        u3 = widgets.HBox([self.sliders[str(i)] for i in range(8, 12)])
        u4 = widgets.HBox([self.sliders[str(i)] for i in range(12, 16)])

        out = widgets.interactive_output(self.affine_transform, self.sliders)

        # Only IPython
        display(main, u1, u2, u3, u4, out)

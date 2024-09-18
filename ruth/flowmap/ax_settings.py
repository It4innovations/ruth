class AxSettings:
    def __init__(self, ax):
        self.ylim = ax.get_ylim()
        self.xlim = ax.get_xlim()
        self.aspect = ax.get_aspect()

    def apply(self, ax):
        ax.set_ylim(self.ylim)
        ax.set_xlim(self.xlim)
        ax.set_aspect(self.aspect)

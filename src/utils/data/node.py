from utils.data.datasetinit import get_loaders


class BaseNodes:
    def __init__(self, channels, n_nodes, dirs, data_folder, label_folder, weights, normMaxValue, normMinValue, test_size, val_size, csv_file=None, batch_size=8, num_workers=4, backbone='enet'):
        self.backbone = backbone
        self.channels = channels
        self.n_nodes = n_nodes
        self.dirs = dirs
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.weights = weights
        self.normMaxValue = normMaxValue
        self.normMinValue = normMinValue
        self.test_size = test_size
        self.val_size = val_size
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_loaders, self.val_loaders, self.test_loaders = None, None, None
        self._init_dataloaders()

    def _init_dataloaders(self):
        # get_loaders returns (train, test, val); keep that order explicit here
        self.train_loaders, self.test_loaders, self.val_loaders = get_loaders(
            self.channels,
            self.n_nodes,
            self.dirs,
            self.data_folder,
            self.label_folder,
            self.weights,
            self.normMaxValue,
            self.normMinValue,
            self.test_size,
            self.val_size,
            csv_file=self.csv_file,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            backbone=self.backbone,
        )

    def __len__(self):
        return self.n_nodes


def get_data(n, dirs, data_folder, label_folder, channels, csv_file=None, batch_size=8, num_workers=4, backbone='enet'):
    nodes = BaseNodes(channels, n, dirs, data_folder, label_folder, weights=[0.5, 0.3, 0.2], normMaxValue=[255, 255, 255], normMinValue=[0, 0, 0], test_size=16, val_size=16, csv_file=csv_file, batch_size=batch_size, num_workers=num_workers, backbone=backbone)
    return nodes.train_loaders, nodes.val_loaders, nodes.test_loaders

import os
import pandas as pd


class DataLoader:
    def __init__(self):
        self.data = None

    def get_data(self, data_dir):
        # Get a list of all jpg files in the data directory
        data = []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if ".jpg" in file:
                    path = os.path.join(root, file)
                    label = file.split("+")[0]
                    data.append({"file": path, "label": label})

        self.data = pd.DataFrame(data)

    def get_classes(self):
        return self.data['label'].unique()

    def get_number_of_classes(self):
        return len(self.get_classes())

    def check_counts(self):
        return self.data['label'].value_counts()


if __name__ == '__main__':
    loader = DataLoader()
    loader.get_data('../data/google_api_images')
    print(loader.get_classes())
    print(loader.data.sample(10))
    print(loader.check_counts())


from sentence_transformers import InputExample
from torch.utils.data import Dataset


# Class implementation for a triplet data set contain positive and negative examples
class TripletDataset(Dataset):
    # -------------------------------------------------------------------------
    def __init__(self, triplet_list) -> None:
        super().__init__()
        self.triplets = []
        self.build_triplets_input_examples(triplet_list)

    # -------------------------------------------------------------------------
    def build_triplets_input_examples(self, triplet_list):

        for triplet in triplet_list:
            if isinstance(triplet, tuple):
                self.triplets.append(InputExample(texts=[triplet[0], triplet[1], triplet[2], triplet[3]]))
            else:
                self.triplets.append(InputExample(texts=[triplet['anchor'], triplet['positive'], triplet['negative'], triplet['anchor_group']]))

    # # -------------------------------------------------------------------------
    # def __init__(self, large_groups, small_words) -> None:
    #     super().__init__()
    #     self.triplets = []
    #     self.small_words = small_words
    #     self.build_triplets(large_groups)

    # # -------------------------------------------------------------------------
    # def build_triplets(self, groups):
    #     for group in groups:
    #         for i in range(len(group)):
    #             for j in range(i + 1, len(group)):
    #                 anchor = group[i]
    #                 positive = group[j]
    #                 negative = random.choice(self.small_words)
    #                 self.triplets.append(InputExample(texts=[anchor, positive, negative]))

    # -------------------------------------------------------------------------
    def __len__(self):
        return (len(self.triplets))

    # -------------------------------------------------------------------------
    def __getitem__(self, index):
        return self.triplets[index]

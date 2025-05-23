from src.dataset import CarDataModule


asdf = CarDataModule(data_dir='data/train', transform=None, batch_size=32, mode='train')

print(type(asdf))
print(asdf)
# print(asdf[1].shape)
print(asdf[0])